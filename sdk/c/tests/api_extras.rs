//! `flats_batch_upsert`, `flats_collections` and `flats_describe`, driven as a
//! C caller would.

use std::ffi::{CStr, CString};
use std::ptr;

use flats_c::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn last_error() -> String {
    // SAFETY: NULL or a NUL-terminated string, copied at once.
    unsafe {
        let p = flats_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

struct Db(*mut flats_db);
impl Db {
    fn open(dir: &std::path::Path) -> Db {
        let p = c(dir.to_str().unwrap());
        let mut db = ptr::null_mut();
        // SAFETY: valid string and out-pointer.
        assert_eq!(unsafe { flats_open(p.as_ptr(), &mut db) }, FLATS_OK);
        Db(db)
    }
    fn exec(&self, sql: &str) {
        let s = c(sql);
        // SAFETY: live handle, valid string, NULL discards the result.
        assert_eq!(
            unsafe { flats_execute(self.0, s.as_ptr(), ptr::null_mut()) },
            FLATS_OK,
            "{sql}: {}",
            last_error()
        );
    }
    /// Every cell of a result, rendered.
    fn table(res: *mut flats_result) -> (Vec<String>, Vec<Vec<String>>) {
        // SAFETY: a live result, freed once here.
        unsafe {
            let cols = (0..flats_result_column_count(res))
                .map(|i| {
                    CStr::from_ptr(flats_result_column_name(res, i))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            for r in 0..flats_result_row_count(res) {
                let mut row = Vec::new();
                for col in 0..cols.len() {
                    row.push(match flats_result_type(res, r, col) {
                        FLATS_TYPE_INT => flats_result_int(res, r, col).to_string(),
                        FLATS_TYPE_FLOAT => flats_result_float(res, r, col).to_string(),
                        FLATS_TYPE_TEXT => {
                            let mut len = 0;
                            let p = flats_result_text(res, r, col, &mut len);
                            CStr::from_ptr(p).to_string_lossy().into_owned()
                        }
                        other => format!("type{other}"),
                    });
                }
                rows.push(row);
            }
            flats_result_free(res);
            (cols, rows)
        }
    }
    fn query(&self, sql: &str) -> Vec<Vec<String>> {
        let s = c(sql);
        let mut res = ptr::null_mut();
        // SAFETY: live handle, valid string and out-pointer.
        assert_eq!(
            unsafe { flats_execute(self.0, s.as_ptr(), &mut res) },
            FLATS_OK,
            "{}",
            last_error()
        );
        let mut rows = Db::table(res).1;
        rows.sort();
        rows
    }
}
impl Drop for Db {
    fn drop(&mut self) {
        // SAFETY: live, released once.
        unsafe { flats_close(self.0) };
    }
}

const SCHEMA: &str = "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(2), n INT, f FLOAT) WITH (capacity = 100);";

#[test]
fn a_batch_can_commit_as_an_upsert() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    db.exec("INSERT INTO docs (id, vector, n, f) VALUES ('a', [1, 0], 1, 0.5);");
    let coll = c("docs");
    let (id, n, f) = (c("id"), c("n"), c("f"));
    // SAFETY: live handles, valid strings and buffers; the batch is freed once.
    unsafe {
        let mut b = ptr::null_mut();
        assert_eq!(flats_batch_new(db.0, coll.as_ptr(), &mut b), FLATS_OK);
        for (key, value) in [("a", 10), ("b", 20)] {
            let k = c(key);
            assert_eq!(flats_batch_vector(b, [0.0f32, 1.0].as_ptr(), 2), FLATS_OK);
            assert_eq!(
                flats_batch_text(b, id.as_ptr(), k.as_ptr(), key.len()),
                FLATS_OK
            );
            assert_eq!(flats_batch_int(b, n.as_ptr(), value), FLATS_OK);
            assert_eq!(flats_batch_float(b, f.as_ptr(), 2.0), FLATS_OK);
            assert_eq!(flats_batch_end_row(b), FLATS_OK);
        }
        let mut written = 0;
        assert_eq!(
            flats_batch_upsert(b, &mut written),
            FLATS_OK,
            "{}",
            last_error()
        );
        assert_eq!(written, 2);
        // A plain commit of an existing key is still refused.
        let k = c("a");
        assert_eq!(flats_batch_vector(b, [0.0f32, 1.0].as_ptr(), 2), FLATS_OK);
        assert_eq!(flats_batch_text(b, id.as_ptr(), k.as_ptr(), 1), FLATS_OK);
        assert_eq!(flats_batch_int(b, n.as_ptr(), 99), FLATS_OK);
        assert_eq!(flats_batch_float(b, f.as_ptr(), 2.0), FLATS_OK);
        assert_eq!(flats_batch_end_row(b), FLATS_OK);
        assert_eq!(flats_batch_commit(b, &mut written), FLATS_ERR_ENGINE);
        flats_batch_free(b);
    }
    assert_eq!(
        db.query("SELECT id, n FROM docs;"),
        vec![
            vec!["a".to_string(), "10".into()],
            vec!["b".into(), "20".into()]
        ]
    );
}

#[test]
fn collections_lists_every_collection_with_its_size() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    db.exec("CREATE COLLECTION other (k INT PRIMARY KEY, emb VECTOR(3)) WITH (capacity = 7);");
    db.exec(
        "INSERT INTO docs (id, vector, n, f) VALUES ('a', [1, 0], 1, 0.5), ('b', [0, 1], 2, 1);",
    );
    db.exec("DELETE FROM docs WHERE id = 'b';");
    let mut res = ptr::null_mut();
    // SAFETY: live handle and out-pointer.
    assert_eq!(unsafe { flats_collections(db.0, &mut res) }, FLATS_OK);
    let (cols, mut rows) = Db::table(res);
    rows.sort();
    assert_eq!(
        cols,
        ["name", "dimension", "capacity", "rows", "primary_key"]
    );
    assert_eq!(
        rows,
        vec![
            vec![
                "docs".to_string(),
                "2".into(),
                "100".into(),
                "1".into(),
                "id".into()
            ],
            vec![
                "other".into(),
                "3".into(),
                "7".into(),
                "0".into(),
                "k".into()
            ],
        ]
    );
}

#[test]
fn describe_lists_columns_in_declaration_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path());
    db.exec(SCHEMA);
    let name = c("docs");
    let mut res = ptr::null_mut();
    // SAFETY: live handle, valid string and out-pointer.
    assert_eq!(
        unsafe { flats_describe(db.0, name.as_ptr(), &mut res) },
        FLATS_OK
    );
    let (cols, rows) = Db::table(res);
    assert_eq!(cols, ["column", "type", "primary_key"]);
    assert_eq!(
        rows,
        vec![
            vec!["id".to_string(), "TEXT".into(), "1".into()],
            vec!["vector".into(), "VECTOR(2)".into(), "0".into()],
            vec!["n".into(), "INT".into(), "0".into()],
            vec!["f".into(), "FLOAT".into(), "0".into()],
        ]
    );
    let nope = c("nope");
    let mut res = ptr::null_mut();
    // SAFETY: as above.
    assert_eq!(
        unsafe { flats_describe(db.0, nope.as_ptr(), &mut res) },
        FLATS_ERR_QUERY
    );
    assert!(res.is_null());
    assert!(last_error().contains("nope"));
}

#[test]
fn the_new_calls_check_their_arguments() {
    let mut res = ptr::null_mut();
    let mut n = 0;
    // SAFETY: NULL handles are part of each contract (refused).
    unsafe {
        assert_eq!(
            flats_collections(ptr::null(), &mut res),
            FLATS_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            flats_describe(ptr::null(), c("x").as_ptr(), &mut res),
            FLATS_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            flats_batch_upsert(ptr::null_mut(), &mut n),
            FLATS_ERR_INVALID_ARGUMENT
        );
    }
}
