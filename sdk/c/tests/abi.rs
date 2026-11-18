//! The C ABI, driven from Rust exactly as a C caller would drive it: raw
//! pointers, NUL-terminated strings, status codes, and `flats_last_error`.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use flats_c::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn last_error() -> Option<String> {
    // SAFETY: flats_last_error returns NULL or a NUL-terminated string that
    // lives until the next flats_* call on this thread; it is copied at once.
    unsafe {
        let p = flats_last_error();
        (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

struct Db(*mut flats_db);

impl Db {
    fn open(dir: &std::path::Path) -> Db {
        let path = c(dir.to_str().unwrap());
        let mut db = ptr::null_mut();
        // SAFETY: valid C string and out-pointer.
        let rc = unsafe { flats_open(path.as_ptr(), &mut db) };
        assert_eq!(rc, FLATS_OK, "open: {:?}", last_error());
        assert!(!db.is_null());
        Db(db)
    }

    fn exec(&self, sql: &str) -> i32 {
        let sql = c(sql);
        // SAFETY: live handle, valid C string, NULL out discards the result.
        unsafe { flats_execute(self.0, sql.as_ptr(), ptr::null_mut()) }
    }

    fn query(&self, sql: &str) -> Res {
        let sql = c(sql);
        let mut res = ptr::null_mut();
        // SAFETY: live handle, valid C string and out-pointer.
        let rc = unsafe { flats_execute(self.0, sql.as_ptr(), &mut res) };
        assert_eq!(rc, FLATS_OK, "{sql:?}: {:?}", last_error());
        Res(res)
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        // SAFETY: the handle is live and released exactly once.
        unsafe { flats_close(self.0) };
    }
}

struct Res(*mut flats_result);

#[derive(Debug, PartialEq)]
enum Cell {
    Int(i64),
    Float(f64),
    Text(String),
    Vector(Vec<f32>),
}

impl Res {
    fn columns(&self) -> Vec<String> {
        // SAFETY: live result; names live as long as it does and are copied.
        unsafe {
            (0..flats_result_column_count(self.0))
                .map(|i| {
                    CStr::from_ptr(flats_result_column_name(self.0, i))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        }
    }

    fn rows(&self) -> Vec<Vec<Cell>> {
        // SAFETY: live result, indices in range, every pointer copied at once.
        unsafe {
            let cols = flats_result_column_count(self.0);
            (0..flats_result_row_count(self.0))
                .map(|r| {
                    (0..cols)
                        .map(|col| match flats_result_type(self.0, r, col) {
                            FLATS_TYPE_INT => Cell::Int(flats_result_int(self.0, r, col)),
                            FLATS_TYPE_FLOAT => Cell::Float(flats_result_float(self.0, r, col)),
                            FLATS_TYPE_TEXT => {
                                let mut len = 0;
                                let p = flats_result_text(self.0, r, col, &mut len);
                                let bytes = std::slice::from_raw_parts(p as *const u8, len);
                                assert_eq!(*p.add(len), 0, "NUL-terminated");
                                Cell::Text(String::from_utf8(bytes.to_vec()).unwrap())
                            }
                            FLATS_TYPE_VECTOR => {
                                let mut len = 0;
                                let p = flats_result_vector(self.0, r, col, &mut len);
                                Cell::Vector(std::slice::from_raw_parts(p, len).to_vec())
                            }
                            other => panic!("unexpected cell type {other}"),
                        })
                        .collect()
                })
                .collect()
        }
    }
}

impl Drop for Res {
    fn drop(&mut self) {
        // SAFETY: released exactly once.
        unsafe { flats_result_free(self.0) };
    }
}

const SCHEMA: &str = "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), \
                      year INT, score FLOAT) WITH (capacity = 1000);";

#[test]
fn version_is_the_engine_version() {
    // SAFETY: static string.
    let v = unsafe { CStr::from_ptr(flats_version()) };
    assert_eq!(v.to_str().unwrap(), "0.1.0");
}

#[test]
fn a_round_trip_through_execute() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    assert_eq!(db.exec(SCHEMA), FLATS_OK);
    assert_eq!(
        db.exec(
            "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 1e-5], 2024, 0.5);"
        ),
        FLATS_OK
    );
    let res = db.query("SELECT id, year, score, vector FROM docs;");
    assert_eq!(res.columns(), vec!["id", "year", "score", "vector"]);
    assert_eq!(
        res.rows(),
        vec![vec![
            Cell::Text("a".into()),
            Cell::Int(2024),
            Cell::Float(0.5),
            Cell::Vector(vec![1.0, 0.0, 1e-5]),
        ]]
    );
    let hits = db.query("SEARCH TOP 1 NEAREST TO [1, 0, 0] FROM docs RETURNING id, score;");
    assert_eq!(hits.columns(), vec!["id", "score"]);
    assert_eq!(hits.rows().len(), 1);
}

#[test]
fn a_write_returns_an_empty_result() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    let res = db.query(SCHEMA);
    assert!(res.columns().is_empty());
    assert!(res.rows().is_empty());
}

#[test]
fn a_bad_statement_is_a_query_error_with_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    assert_eq!(db.exec("SELEC nope;"), FLATS_ERR_QUERY);
    assert!(last_error().is_some());
    assert_eq!(db.exec("SELECT id FROM missing;"), FLATS_ERR_QUERY);
    assert!(last_error().unwrap().contains("missing"));
}

#[test]
fn an_engine_refusal_is_an_engine_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let ins = "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);";
    assert_eq!(db.exec(ins), FLATS_OK);
    assert_eq!(db.exec(ins), FLATS_ERR_ENGINE, "duplicate key");
    assert!(last_error().unwrap().contains("'a'"), "{:?}", last_error());
}

#[test]
fn null_and_invalid_arguments_are_refused_not_crashed_on() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    // SAFETY: every call below is passed NULL or invalid bytes ON PURPOSE —
    // the contract is a refusal, never a dereference.
    unsafe {
        let mut out = ptr::null_mut();
        assert_eq!(
            flats_open(ptr::null(), &mut out),
            FLATS_ERR_INVALID_ARGUMENT
        );
        assert!(out.is_null());
        let p = c(dir.path().to_str().unwrap());
        assert_eq!(
            flats_open(p.as_ptr(), ptr::null_mut()),
            FLATS_ERR_INVALID_ARGUMENT
        );
        let sql = c("SELECT id FROM docs;");
        assert_eq!(
            flats_execute(ptr::null_mut(), sql.as_ptr(), ptr::null_mut()),
            FLATS_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            flats_execute(db.0, ptr::null(), ptr::null_mut()),
            FLATS_ERR_INVALID_ARGUMENT
        );
        let bad_utf8 = [0xFFu8, 0xFE, 0];
        assert_eq!(
            flats_execute(db.0, bad_utf8.as_ptr() as *const c_char, ptr::null_mut()),
            FLATS_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().unwrap().contains("UTF-8"));
        // Releasing NULL is a no-op.
        assert_eq!(flats_close(ptr::null_mut()), FLATS_OK);
        flats_result_free(ptr::null_mut());
        flats_batch_free(ptr::null_mut());
    }
}

#[test]
fn out_of_range_accessors_return_zero_or_null() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    db.exec("INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);");
    let res = db.query("SELECT id, year FROM docs;");
    // SAFETY: live result; out-of-range indices are part of the contract.
    unsafe {
        assert_eq!(flats_result_type(res.0, 5, 0), 0);
        assert_eq!(flats_result_type(res.0, 0, 9), 0);
        assert!(flats_result_column_name(res.0, 9).is_null());
        assert!(
            flats_result_text(res.0, 0, 1, ptr::null_mut()).is_null(),
            "not TEXT"
        );
        assert!(
            flats_result_vector(res.0, 0, 0, ptr::null_mut()).is_null(),
            "not VECTOR"
        );
        assert_eq!(flats_result_int(res.0, 0, 0), 0, "not INT");
        assert_eq!(flats_result_int(res.0, 0, 1), 1);
        // NULL result handles answer like an empty result.
        assert_eq!(flats_result_row_count(ptr::null()), 0);
        assert_eq!(flats_result_column_count(ptr::null()), 0);
    }
}

#[test]
fn the_last_error_belongs_to_the_thread_that_failed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    assert_eq!(db.exec("nonsense;"), FLATS_ERR_QUERY);
    assert!(last_error().is_some());
    std::thread::spawn(|| assert!(last_error().is_none(), "another thread's error leaked"))
        .join()
        .unwrap();
}

#[test]
fn one_handle_serves_many_threads() {
    struct Shared(*mut flats_db);
    // SAFETY: the contract says a flats_db may be shared across threads.
    unsafe impl Sync for Shared {}
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let shared = Shared(db.0);
    std::thread::scope(|s| {
        for t in 0..4 {
            let shared = &shared;
            s.spawn(move || {
                for i in 0..25 {
                    let sql = c(&format!(
                        "INSERT INTO docs (id, vector, year, score) VALUES ('t{t}-{i}', [1, 0, 0], {i}, 0.0);"
                    ));
                    // SAFETY: shared live handle, valid string.
                    let rc = unsafe { flats_execute(shared.0, sql.as_ptr(), ptr::null_mut()) };
                    assert_eq!(rc, FLATS_OK);
                }
            });
        }
    });
    assert_eq!(db.query("SELECT id FROM docs;").rows().len(), 100);
}

// -- batch ---------------------------------------------------------------

struct Batch(*mut flats_batch);

impl Batch {
    fn new(db: &Db, collection: &str) -> Result<Batch, i32> {
        let name = c(collection);
        let mut b = ptr::null_mut();
        // SAFETY: live handle, valid string and out-pointer.
        match unsafe { flats_batch_new(db.0, name.as_ptr(), &mut b) } {
            FLATS_OK => Ok(Batch(b)),
            rc => Err(rc),
        }
    }
    fn vector(&self, v: &[f32]) -> i32 {
        // SAFETY: live batch; `v` is valid for `v.len()` floats.
        unsafe { flats_batch_vector(self.0, v.as_ptr(), v.len()) }
    }
    fn int(&self, col: &str, v: i64) -> i32 {
        let col = c(col);
        // SAFETY: live batch, valid string.
        unsafe { flats_batch_int(self.0, col.as_ptr(), v) }
    }
    fn float(&self, col: &str, v: f64) -> i32 {
        let col = c(col);
        // SAFETY: live batch, valid string.
        unsafe { flats_batch_float(self.0, col.as_ptr(), v) }
    }
    fn text(&self, col: &str, v: &str) -> i32 {
        let col = c(col);
        // SAFETY: live batch; `v` is valid for `v.len()` bytes (no NUL needed).
        unsafe { flats_batch_text(self.0, col.as_ptr(), v.as_ptr() as *const c_char, v.len()) }
    }
    fn end_row(&self) -> i32 {
        // SAFETY: live batch.
        unsafe { flats_batch_end_row(self.0) }
    }
    fn len(&self) -> usize {
        // SAFETY: live batch.
        unsafe { flats_batch_len(self.0) }
    }
    fn commit(&self) -> (i32, usize) {
        let mut n = usize::MAX;
        // SAFETY: live batch, valid out-pointer.
        let rc = unsafe { flats_batch_commit(self.0, &mut n) };
        (rc, n)
    }
    fn row(&self, id: &str, year: i64) {
        assert_eq!(self.vector(&[year as f32, 1.0, 0.0]), FLATS_OK);
        assert_eq!(self.text("id", id), FLATS_OK);
        assert_eq!(self.int("year", year), FLATS_OK);
        assert_eq!(self.float("score", year as f64 / 2.0), FLATS_OK);
        assert_eq!(self.end_row(), FLATS_OK, "{:?}", last_error());
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: released exactly once.
        unsafe { flats_batch_free(self.0) };
    }
}

#[test]
fn a_batch_inserts_every_row_as_one_statement() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let b = Batch::new(&db, "docs").unwrap();
    for i in 0..100 {
        b.row(&format!("d{i}"), i);
    }
    assert_eq!(b.len(), 100);
    assert_eq!(b.commit(), (FLATS_OK, 100));
    assert_eq!(b.len(), 0, "emptied");
    let res = db.query("SELECT id, year, score, vector FROM docs WHERE year = 7;");
    assert_eq!(
        res.rows(),
        vec![vec![
            Cell::Text("d7".into()),
            Cell::Int(7),
            Cell::Float(3.5),
            Cell::Vector(vec![7.0, 1.0, 0.0]),
        ]]
    );
    // Reusable after a commit.
    b.row("again", 1000);
    assert_eq!(b.commit(), (FLATS_OK, 1));
    assert_eq!(db.query("SELECT id FROM docs;").rows().len(), 101);
}

#[test]
fn a_batch_checks_each_value_as_it_is_added() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let b = Batch::new(&db, "docs").unwrap();
    assert_eq!(
        b.int("nope", 1),
        FLATS_ERR_INVALID_ARGUMENT,
        "unknown column"
    );
    assert!(last_error().unwrap().contains("nope"));
    assert_eq!(
        b.text("year", "x"),
        FLATS_ERR_INVALID_ARGUMENT,
        "wrong type"
    );
    assert_eq!(
        b.int("id", 1),
        FLATS_ERR_INVALID_ARGUMENT,
        "wrong type for the key"
    );
    assert_eq!(
        b.vector(&[1.0, 2.0]),
        FLATS_ERR_INVALID_ARGUMENT,
        "wrong dimension"
    );
    assert_eq!(
        b.float("score", f64::NAN),
        FLATS_ERR_INVALID_ARGUMENT,
        "NaN"
    );
    assert_eq!(
        b.vector(&[1.0, f32::INFINITY, 0.0]),
        FLATS_ERR_INVALID_ARGUMENT,
        "inf"
    );
    // A row missing a column cannot be finished.
    assert_eq!(b.vector(&[1.0, 0.0, 0.0]), FLATS_OK);
    assert_eq!(b.text("id", "x"), FLATS_OK);
    assert_eq!(b.end_row(), FLATS_ERR_INVALID_ARGUMENT);
    assert!(last_error().unwrap().contains("year"), "{:?}", last_error());
    assert_eq!(b.len(), 0);
}

#[test]
fn a_batch_with_a_duplicate_key_lands_nothing_and_is_emptied() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let b = Batch::new(&db, "docs").unwrap();
    b.row("a", 1);
    b.row("b", 2);
    b.row("a", 3);
    let (rc, n) = b.commit();
    assert_eq!(rc, FLATS_ERR_ENGINE);
    assert_eq!(n, 0);
    assert!(last_error().unwrap().contains("'a'"));
    assert_eq!(b.len(), 0, "emptied on failure too");
    assert!(db.query("SELECT id FROM docs;").rows().is_empty());
}

#[test]
fn a_batch_for_an_unknown_collection_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    assert_eq!(Batch::new(&db, "missing").err(), Some(FLATS_ERR_QUERY));
}

#[test]
fn backup_then_open_the_copy() {
    let dir = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    {
        let db = Db::open(dir.path());
        db.exec(SCHEMA);
        db.exec("INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);");
        let dest = c(copy.to_str().unwrap());
        // SAFETY: live handle, valid string.
        assert_eq!(unsafe { flats_backup(db.0, dest.as_ptr()) }, FLATS_OK);
        // A second backup into the now non-empty directory is refused.
        // SAFETY: as above.
        assert_eq!(
            unsafe { flats_backup(db.0, dest.as_ptr()) },
            FLATS_ERR_ENGINE
        );
    }
    let restored = Db::open(&copy);
    assert_eq!(restored.query("SELECT id FROM docs;").rows().len(), 1);
}

#[test]
fn close_checkpoints_and_reports_ok() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let raw = db.0;
    std::mem::forget(db);
    // SAFETY: released exactly once (the wrapper was forgotten).
    assert_eq!(unsafe { flats_close(raw) }, FLATS_OK);
    let db = Db::open(dir.path());
    assert_eq!(db.query("SELECT id FROM docs;").rows().len(), 0);
}
