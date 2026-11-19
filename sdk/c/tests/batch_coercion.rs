//! A batch takes values the way V-SQL does: an integer into a FLOAT column
//! is coerced, exactly as `INSERT ... VALUES (1)` and a `?` parameter are.
//! Found by `fuzz/sqlite_diff.py`: Python's `insert({"score": 1})` was refused
//! with "column is Float, not Int".

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

#[test]
fn an_int_goes_into_a_float_column_as_a_float() {
    let dir = tempfile::tempdir().unwrap();
    let path = c(dir.path().to_str().unwrap());
    // SAFETY: valid strings and out-pointers throughout; every handle is
    // released once.
    unsafe {
        let mut db = ptr::null_mut();
        assert_eq!(flats_open(path.as_ptr(), &mut db), FLATS_OK);
        let ddl = c(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(1), f FLOAT) WITH (capacity = 10);",
        );
        assert_eq!(flats_execute(db, ddl.as_ptr(), ptr::null_mut()), FLATS_OK);

        let mut b = ptr::null_mut();
        let coll = c("t");
        assert_eq!(flats_batch_new(db, coll.as_ptr(), &mut b), FLATS_OK);
        let (id, f) = (c("id"), c("f"));
        for (key, value) in [(1, 3i64), (2, i64::MAX), (3, -7)] {
            assert_eq!(flats_batch_vector(b, [1.0f32].as_ptr(), 1), FLATS_OK);
            assert_eq!(flats_batch_int(b, id.as_ptr(), key), FLATS_OK);
            assert_eq!(
                flats_batch_int(b, f.as_ptr(), value),
                FLATS_OK,
                "{}",
                last_error()
            );
            assert_eq!(flats_batch_end_row(b), FLATS_OK);
        }
        let mut n = 0;
        assert_eq!(flats_batch_commit(b, &mut n), FLATS_OK, "{}", last_error());
        assert_eq!(n, 3);
        flats_batch_free(b);

        let sql = c("SELECT id, f FROM t;");
        let mut res = ptr::null_mut();
        assert_eq!(flats_execute(db, sql.as_ptr(), &mut res), FLATS_OK);
        let mut got = Vec::new();
        for r in 0..flats_result_row_count(res) {
            assert_eq!(flats_result_type(res, r, 1), FLATS_TYPE_FLOAT);
            got.push((flats_result_int(res, r, 0), flats_result_float(res, r, 1)));
        }
        got.sort_by_key(|x| x.0);
        assert_eq!(got, vec![(1, 3.0), (2, i64::MAX as f64), (3, -7.0)]);
        flats_result_free(res);
        assert_eq!(flats_close(db), FLATS_OK);
    }
}

#[test]
fn a_float_still_does_not_go_into_an_int_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = c(dir.path().to_str().unwrap());
    // SAFETY: as above.
    unsafe {
        let mut db = ptr::null_mut();
        assert_eq!(flats_open(path.as_ptr(), &mut db), FLATS_OK);
        let ddl = c(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(1), n INT) WITH (capacity = 10);",
        );
        assert_eq!(flats_execute(db, ddl.as_ptr(), ptr::null_mut()), FLATS_OK);
        let mut b = ptr::null_mut();
        let coll = c("t");
        assert_eq!(flats_batch_new(db, coll.as_ptr(), &mut b), FLATS_OK);
        let n = c("n");
        assert_eq!(
            flats_batch_float(b, n.as_ptr(), 1.0),
            FLATS_ERR_INVALID_ARGUMENT
        );
        flats_batch_free(b);
        assert_eq!(flats_close(db), FLATS_OK);
    }
}
