//! `flats_open` refusing a database whose store file is damaged, and
//! `flats_open_salvage` getting past it — driven as a C caller would.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use flats_c::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn last_error() -> String {
    // SAFETY: NULL or a NUL-terminated string valid until this thread's next
    // flats_* call; copied at once.
    unsafe {
        let p = flats_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

type Opener = unsafe extern "C" fn(*const std::ffi::c_char, *mut *mut flats_db) -> i32;

fn try_open(open: Opener, dir: &Path) -> Result<*mut flats_db, (i32, String)> {
    let path = c(dir.to_str().unwrap());
    let mut db = ptr::null_mut();
    // SAFETY: a valid C string and out-pointer.
    let rc = unsafe { open(path.as_ptr(), &mut db) };
    if rc == FLATS_OK {
        assert!(!db.is_null());
        Ok(db)
    } else {
        assert!(db.is_null(), "a failed open hands back no handle");
        Err((rc, last_error()))
    }
}

fn exec(db: *mut flats_db, sql: &str) {
    let sql = c(sql);
    // SAFETY: a live handle and a valid C string; NULL discards the result.
    let rc = unsafe { flats_execute(db, sql.as_ptr(), ptr::null_mut()) };
    assert_eq!(rc, FLATS_OK, "{sql:?}: {}", last_error());
}

fn count(db: *mut flats_db, sql: &str) -> usize {
    let sql = c(sql);
    let mut res = ptr::null_mut();
    // SAFETY: a live handle, a valid C string and out-pointer; the result is
    // freed once.
    unsafe {
        let rc = flats_execute(db, sql.as_ptr(), &mut res);
        assert_eq!(rc, FLATS_OK, "{sql:?}: {}", last_error());
        let n = flats_result_row_count(res);
        flats_result_free(res);
        n
    }
}

fn close(db: *mut flats_db) {
    // SAFETY: a live handle, released once.
    assert_eq!(unsafe { flats_close(db) }, FLATS_OK);
}

/// Ten rows, checkpointed by a clean close; then the tuple store damaged.
fn damaged(dir: &Path) {
    let db = try_open(flats_open, dir).unwrap();
    exec(
        db,
        "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);",
    );
    for i in 0..10 {
        exec(
            db,
            &format!("INSERT INTO t (id, vector) VALUES ({i}, [1, 0]);"),
        );
    }
    close(db);
    let file = dir.join("collection-0").join("tuples.snap");
    let mut bytes = std::fs::read(&file).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&file, bytes).unwrap();
}

#[test]
fn open_refuses_a_damaged_store_and_says_how_to_salvage() {
    let dir = tempfile::tempdir().unwrap();
    damaged(dir.path());
    let (rc, msg) = try_open(flats_open, dir.path()).unwrap_err();
    assert_eq!(rc, FLATS_ERR_ENGINE);
    assert!(msg.contains("tuples.snap"), "{msg}");
    assert!(msg.contains("salvage"), "{msg}");
}

#[test]
fn open_salvage_opens_it_and_the_database_is_healthy_after() {
    let dir = tempfile::tempdir().unwrap();
    damaged(dir.path());

    let db = try_open(flats_open_salvage, dir.path()).expect("salvage opens");
    // Every row was checkpointed, so the log proves none of them.
    assert_eq!(count(db, "SELECT id FROM t;"), 0);
    exec(db, "INSERT INTO t (id, vector) VALUES (1, [1, 0]);");
    close(db);

    let db = try_open(flats_open, dir.path()).expect("a normal open works again");
    assert_eq!(count(db, "SELECT id FROM t;"), 1);
    close(db);
}

#[test]
fn open_salvage_on_a_healthy_database_is_a_plain_open() {
    let dir = tempfile::tempdir().unwrap();
    let db = try_open(flats_open, dir.path()).unwrap();
    exec(
        db,
        "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);",
    );
    exec(db, "INSERT INTO t (id, vector) VALUES (1, [1, 0]);");
    close(db);
    let db = try_open(flats_open_salvage, dir.path()).unwrap();
    assert_eq!(count(db, "SELECT id FROM t;"), 1);
    close(db);
}

#[test]
fn open_salvage_checks_its_arguments() {
    let mut db = ptr::null_mut();
    // SAFETY: NULL path is part of the contract (refused); out is valid.
    assert_eq!(
        unsafe { flats_open_salvage(ptr::null(), &mut db) },
        FLATS_ERR_INVALID_ARGUMENT
    );
    let path = c("/tmp/never-opened");
    // SAFETY: NULL out is part of the contract (refused).
    assert_eq!(
        unsafe { flats_open_salvage(path.as_ptr(), ptr::null_mut()) },
        FLATS_ERR_INVALID_ARGUMENT
    );
}
