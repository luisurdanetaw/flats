//! The C ABI for Flats. `include/flats.h` is the contract; this file is its
//! only implementation, and every language binding wraps these functions.
//!
//! # Where the `unsafe` lives
//!
//! The engine quarantines `unsafe` to `simd/` and `platform/` (CLAUDE.md §5).
//! An exported C ABI cannot avoid it — every argument is a raw pointer — so
//! it lives HERE, in its own crate, rather than in the engine. Each `unsafe`
//! block names its precondition, and every precondition is one the header
//! states to the caller. Above this layer, the crate uses only `flats`'s
//! public API, exactly as any downstream Rust crate would.
//!
//! # Shape of every entry point
//!
//! Each fallible function runs its body inside [`guard`], which turns a
//! `Fail` into a status code plus the thread's last-error message, and a
//! panic into `FLATS_ERR_INTERNAL` — a Rust panic must never unwind into C.

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
// The exported types are named as C sees them (`flats_db`, …).
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;

use flats::{ColumnId, ColumnType, Db, DbOptions, QueryError, RegValue, Row, Value};

// ---------------------------------------------------------------------------
// Status codes and cell types — mirror flats.h exactly.
// ---------------------------------------------------------------------------

/// Success.
pub const FLATS_OK: i32 = 0;
/// A NULL, invalid UTF-8, an out-of-range index, or a value the schema refuses.
pub const FLATS_ERR_INVALID_ARGUMENT: i32 = 1;
/// The statement did not parse, bind or compile, or named something missing.
pub const FLATS_ERR_QUERY: i32 = 2;
/// The engine refused or failed the operation.
pub const FLATS_ERR_ENGINE: i32 = 3;
/// A bug in Flats (a caught panic).
pub const FLATS_ERR_INTERNAL: i32 = 4;

/// An `INT` cell.
pub const FLATS_TYPE_INT: i32 = 1;
/// A `FLOAT` cell.
pub const FLATS_TYPE_FLOAT: i32 = 2;
/// A `TEXT` cell.
pub const FLATS_TYPE_TEXT: i32 = 3;
/// A `VECTOR` cell.
pub const FLATS_TYPE_VECTOR: i32 = 4;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

thread_local! {
    /// The calling thread's most recent failure. Replaced, never cleared, so a
    /// pointer handed out by `flats_last_error` stays valid until the next
    /// failure on the same thread — which is what the header promises.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// A failed call: the status to return and the message to record.
struct Fail {
    code: i32,
    message: String,
}

impl Fail {
    fn invalid(message: impl Into<String>) -> Fail {
        Fail {
            code: FLATS_ERR_INVALID_ARGUMENT,
            message: message.into(),
        }
    }

    fn engine(e: flats::Error) -> Fail {
        // A name that resolves to nothing is the caller naming something that
        // does not exist — the header files that under QUERY, like a SELECT
        // of a missing collection.
        let code = match e {
            flats::Error::UnknownCollectionName { .. } => FLATS_ERR_QUERY,
            _ => FLATS_ERR_ENGINE,
        };
        Fail {
            code,
            message: e.to_string(),
        }
    }

    fn query(e: QueryError) -> Fail {
        let code = match &e {
            QueryError::Parse(_) | QueryError::Bind(_) | QueryError::Compile(_) => FLATS_ERR_QUERY,
            // The compiler emitted a malformed program: a Flats bug.
            QueryError::Validate(_) => FLATS_ERR_INTERNAL,
            _ => FLATS_ERR_ENGINE,
        };
        Fail {
            code,
            message: e.to_string(),
        }
    }
}

fn record(message: String) {
    // An interior NUL cannot be represented; keep the readable prefix.
    let message = match CString::new(message) {
        Ok(m) => m,
        Err(e) => {
            let at = e.nul_position();
            let mut bytes = e.into_vec();
            bytes.truncate(at);
            CString::new(bytes).unwrap_or_default()
        }
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message));
}

/// Run one entry point's body: `Ok` is `FLATS_OK`; a `Fail` is recorded and
/// its code returned; a panic is caught and reported as an internal error.
fn guard(body: impl FnOnce() -> Result<(), Fail>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => FLATS_OK,
        Ok(Err(fail)) => {
            record(fail.message);
            fail.code
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            record(format!("internal error (panic): {what}"));
            FLATS_ERR_INTERNAL
        }
    }
}

/// Borrow a NUL-terminated UTF-8 argument.
///
/// # Safety
/// `p` is NULL or points to a NUL-terminated string that outlives the call.
unsafe fn arg_str<'a>(p: *const c_char, what: &str) -> Result<&'a str, Fail> {
    if p.is_null() {
        return Err(Fail::invalid(format!("{what} is NULL")));
    }
    // SAFETY: non-NULL, and the caller guarantees NUL termination and
    // lifetime (header: "every const char * argument is NUL-terminated").
    let s = unsafe { CStr::from_ptr(p) };
    s.to_str()
        .map_err(|_| Fail::invalid(format!("{what} is not valid UTF-8")))
}

/// Borrow a handle argument.
///
/// # Safety
/// `p` is NULL or a live handle of type `T` this crate handed out.
unsafe fn arg_ref<'a, T>(p: *const T, what: &str) -> Result<&'a T, Fail> {
    // SAFETY: the caller guarantees `p` is NULL or live (header: handles).
    unsafe { p.as_ref() }.ok_or_else(|| Fail::invalid(format!("{what} is NULL")))
}

/// Mutable form of [`arg_ref`].
///
/// # Safety
/// `p` is NULL or a live handle of type `T`, used by one thread at a time.
unsafe fn arg_mut<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T, Fail> {
    // SAFETY: as `arg_ref`, plus the header's one-thread-at-a-time rule for
    // results and batches makes the `&mut` unique.
    unsafe { p.as_mut() }.ok_or_else(|| Fail::invalid(format!("{what} is NULL")))
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// `const char *flats_version(void)`
#[unsafe(no_mangle)]
pub extern "C" fn flats_version() -> *const c_char {
    VERSION.as_ptr().cast::<c_char>()
}

/// `const char *flats_last_error(void)`
#[unsafe(no_mangle)]
pub extern "C" fn flats_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map_or(ptr::null(), |m| m.as_ptr()))
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// An open database.
pub struct flats_db {
    db: Db,
}

/// `int flats_open(const char *path, flats_db **out)`
///
/// # Safety
/// `path` is NULL or a NUL-terminated string; `out` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_open(path: *const c_char, out: *mut *mut flats_db) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Fail::invalid("out is NULL"));
        }
        // SAFETY: forwarded from this function's own contract.
        let path = unsafe { arg_str(path, "path") }?;
        let db = Db::open(Path::new(path), &[], DbOptions::default()).map_err(Fail::engine)?;
        let handle = Box::into_raw(Box::new(flats_db { db }));
        // SAFETY: `out` is non-NULL and writable by contract.
        unsafe { *out = handle };
        Ok(())
    })
}

/// `int flats_close(flats_db *db)`
///
/// # Safety
/// `db` is NULL or a live handle from `flats_open`, not used again after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_close(db: *mut flats_db) -> i32 {
    guard(|| {
        if db.is_null() {
            return Ok(());
        }
        // SAFETY: a live handle from `flats_open` (a `Box`), released once.
        let handle = unsafe { Box::from_raw(db) };
        handle.db.close().map_err(Fail::engine)
    })
}

/// `int flats_execute(flats_db *db, const char *sql, flats_result **out)`
///
/// # Safety
/// `db` is NULL or live; `sql` is NULL or a NUL-terminated string; `out` is
/// NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_execute(
    db: *const flats_db,
    sql: *const c_char,
    out: *mut *mut flats_result,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let handle = unsafe { arg_ref(db, "db") }?;
        // SAFETY: as above.
        let sql = unsafe { arg_str(sql, "sql") }?;
        let result = collect(&handle.db, sql)?;
        if !out.is_null() {
            // SAFETY: non-NULL and writable by contract.
            unsafe { *out = Box::into_raw(Box::new(result)) };
        }
        Ok(())
    })
}

/// `int flats_backup(flats_db *db, const char *dest)`
///
/// # Safety
/// `db` is NULL or live; `dest` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_backup(db: *const flats_db, dest: *const c_char) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let handle = unsafe { arg_ref(db, "db") }?;
        // SAFETY: as above.
        let dest = unsafe { arg_str(dest, "dest") }?;
        handle.db.backup(dest).map_err(Fail::engine)
    })
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// One collected cell, in the form C reads it.
enum Cell {
    Int(i64),
    Float(f64),
    /// UTF-8 bytes plus a trailing NUL, so the pointer is a C string too.
    Text(Vec<u8>),
    Vector(Vec<f32>),
}

/// A fully collected statement result.
pub struct flats_result {
    columns: Vec<CString>,
    rows: Vec<Vec<Cell>>,
}

fn collect(db: &Db, sql: &str) -> Result<flats_result, Fail> {
    let stream = db.execute(sql).map_err(Fail::query)?;
    let columns = stream
        .labels()
        .iter()
        .map(|l| CString::new(l.as_str()).map_err(|_| Fail::invalid("column label holds a NUL")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::new();
    for row in stream {
        let row = row.map_err(Fail::query)?;
        let cells = row
            .0
            .into_iter()
            .map(|cell| match cell {
                RegValue::Int(n) => Ok(Cell::Int(n)),
                RegValue::Real(f) => Ok(Cell::Float(f)),
                RegValue::Str(s) => {
                    let mut bytes = s.into_bytes();
                    bytes.push(0);
                    Ok(Cell::Text(bytes))
                }
                RegValue::Vector(v) => Ok(Cell::Vector(v.to_vec())),
                other => Err(Fail {
                    code: FLATS_ERR_INTERNAL,
                    message: format!("a result row held a non-value register: {other:?}"),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(cells);
    }
    Ok(flats_result { columns, rows })
}

/// The cell at `(row, col)`, if `res` is live and the indices are in range.
///
/// # Safety
/// `res` is NULL or a live result.
unsafe fn cell<'a>(res: *const flats_result, row: usize, col: usize) -> Option<&'a Cell> {
    // SAFETY: NULL or live, by contract.
    unsafe { res.as_ref() }?.rows.get(row)?.get(col)
}

/// `size_t flats_result_column_count(const flats_result *res)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_column_count(res: *const flats_result) -> usize {
    // SAFETY: NULL or live, by contract.
    unsafe { res.as_ref() }.map_or(0, |r| r.columns.len())
}

/// `const char *flats_result_column_name(const flats_result *res, size_t col)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_column_name(
    res: *const flats_result,
    col: usize,
) -> *const c_char {
    // SAFETY: NULL or live, by contract.
    unsafe { res.as_ref() }
        .and_then(|r| r.columns.get(col))
        .map_or(ptr::null(), |c| c.as_ptr())
}

/// `size_t flats_result_row_count(const flats_result *res)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_row_count(res: *const flats_result) -> usize {
    // SAFETY: NULL or live, by contract.
    unsafe { res.as_ref() }.map_or(0, |r| r.rows.len())
}

/// `int flats_result_type(const flats_result *res, size_t row, size_t col)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_type(
    res: *const flats_result,
    row: usize,
    col: usize,
) -> i32 {
    // SAFETY: forwarded from this function's own contract.
    match unsafe { cell(res, row, col) } {
        Some(Cell::Int(_)) => FLATS_TYPE_INT,
        Some(Cell::Float(_)) => FLATS_TYPE_FLOAT,
        Some(Cell::Text(_)) => FLATS_TYPE_TEXT,
        Some(Cell::Vector(_)) => FLATS_TYPE_VECTOR,
        None => 0,
    }
}

/// `int64_t flats_result_int(const flats_result *res, size_t row, size_t col)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_int(res: *const flats_result, row: usize, col: usize) -> i64 {
    // SAFETY: forwarded from this function's own contract.
    match unsafe { cell(res, row, col) } {
        Some(Cell::Int(n)) => *n,
        _ => 0,
    }
}

/// `double flats_result_float(const flats_result *res, size_t row, size_t col)`
///
/// # Safety
/// `res` is NULL or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_float(
    res: *const flats_result,
    row: usize,
    col: usize,
) -> f64 {
    // SAFETY: forwarded from this function's own contract.
    match unsafe { cell(res, row, col) } {
        Some(Cell::Float(f)) => *f,
        _ => 0.0,
    }
}

/// `const char *flats_result_text(const flats_result *res, size_t row, size_t col, size_t *len)`
///
/// # Safety
/// `res` is NULL or a live result; `len` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_text(
    res: *const flats_result,
    row: usize,
    col: usize,
    len: *mut usize,
) -> *const c_char {
    // SAFETY: forwarded from this function's own contract.
    match unsafe { cell(res, row, col) } {
        Some(Cell::Text(bytes)) => {
            if !len.is_null() {
                // SAFETY: non-NULL and writable by contract. The stored bytes
                // end in the NUL this length excludes.
                unsafe { *len = bytes.len().saturating_sub(1) };
            }
            bytes.as_ptr().cast::<c_char>()
        }
        _ => ptr::null(),
    }
}

/// `const float *flats_result_vector(const flats_result *res, size_t row, size_t col, size_t *len)`
///
/// # Safety
/// `res` is NULL or a live result; `len` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_vector(
    res: *const flats_result,
    row: usize,
    col: usize,
    len: *mut usize,
) -> *const f32 {
    // SAFETY: forwarded from this function's own contract.
    match unsafe { cell(res, row, col) } {
        Some(Cell::Vector(v)) => {
            if !len.is_null() {
                // SAFETY: non-NULL and writable by contract.
                unsafe { *len = v.len() };
            }
            v.as_ptr()
        }
        _ => ptr::null(),
    }
}

/// `void flats_result_free(flats_result *res)`
///
/// # Safety
/// `res` is NULL or a live result, not used again after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_result_free(res: *mut flats_result) {
    if !res.is_null() {
        // SAFETY: a live result from `flats_execute` (a `Box`), released once.
        drop(unsafe { Box::from_raw(res) });
    }
}

// ---------------------------------------------------------------------------
// Batch insert
// ---------------------------------------------------------------------------

/// Rows being built for one `Db::insert_many`.
pub struct flats_batch {
    /// The database the batch commits to. The header requires batches to be
    /// freed before their database is closed.
    db: *const flats_db,
    collection: u32,
    dim: usize,
    /// The vector column's name, so `flats_batch_column_type` can answer it.
    vector_name: String,
    /// Scalar column name → (storage id, type).
    columns: HashMap<String, (ColumnId, ColumnType)>,
    /// Column names by storage id, for "missing column" messages.
    names: Vec<String>,
    /// The row under construction.
    vector: Option<Vec<f32>>,
    values: Vec<Option<Value>>,
    /// Finished rows.
    rows: Vec<(Vec<f32>, Row)>,
}

/// `int flats_batch_new(flats_db *db, const char *collection, flats_batch **out)`
///
/// # Safety
/// `db` is NULL or live; `collection` is NULL or a NUL-terminated string;
/// `out` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_new(
    db: *const flats_db,
    collection: *const c_char,
    out: *mut *mut flats_batch,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Fail::invalid("out is NULL"));
        }
        // SAFETY: forwarded from this function's own contract.
        let handle = unsafe { arg_ref(db, "db") }?;
        // SAFETY: as above.
        let name = unsafe { arg_str(collection, "collection") }?;
        let id = handle.db.collection_id(name).map_err(Fail::engine)?;
        let config = handle
            .db
            .collections()
            .into_iter()
            .find(|c| c.id == id)
            .ok_or_else(|| Fail::engine(flats::Error::UnknownCollection { id }))?;
        let schema = &config.schema;
        let columns = schema
            .columns
            .iter()
            .map(|c| (c.name.clone(), (c.id, c.ty)))
            .collect();
        let names = schema.columns.iter().map(|c| c.name.clone()).collect();
        let batch = flats_batch {
            db,
            collection: id,
            dim: schema.vector().dim.get(),
            vector_name: schema.vector().name.clone(),
            columns,
            names,
            vector: None,
            values: vec![None; schema.columns.len()],
            rows: Vec::new(),
        };
        // SAFETY: non-NULL and writable by contract.
        unsafe { *out = Box::into_raw(Box::new(batch)) };
        Ok(())
    })
}

impl flats_batch {
    fn set(&mut self, column: &str, value: Value) -> Result<(), Fail> {
        let (id, ty) = *self
            .columns
            .get(column)
            .ok_or_else(|| Fail::invalid(format!("no column {column:?} in this collection")))?;
        if value.column_type() != ty {
            return Err(Fail::invalid(format!(
                "column {column:?} is {ty:?}, not {:?}",
                value.column_type()
            )));
        }
        if let Value::Float(f) = value
            && f.is_nan()
        {
            return Err(Fail::invalid(format!(
                "column {column:?}: NaN is not a value"
            )));
        }
        self.values[id as usize] = Some(value);
        Ok(())
    }
}

/// `int flats_batch_vector(flats_batch *b, const float *data, size_t len)`
///
/// # Safety
/// `b` is NULL or a live batch; `data` is NULL or valid for `len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_vector(
    b: *mut flats_batch,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        if data.is_null() {
            return Err(Fail::invalid("data is NULL"));
        }
        if len != batch.dim {
            return Err(Fail::invalid(format!(
                "vector has {len} components; the collection's dimension is {}",
                batch.dim
            )));
        }
        // SAFETY: non-NULL and valid for `len` floats by contract.
        let v = unsafe { std::slice::from_raw_parts(data, len) };
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Fail::invalid("vector components must be finite"));
        }
        batch.vector = Some(v.to_vec());
        Ok(())
    })
}

/// `int flats_batch_int(flats_batch *b, const char *column, int64_t value)`
///
/// # Safety
/// `b` is NULL or a live batch; `column` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_int(
    b: *mut flats_batch,
    column: *const c_char,
    value: i64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        // SAFETY: as above.
        let column = unsafe { arg_str(column, "column") }?;
        batch.set(column, Value::Int(value))
    })
}

/// `int flats_batch_float(flats_batch *b, const char *column, double value)`
///
/// # Safety
/// `b` is NULL or a live batch; `column` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_float(
    b: *mut flats_batch,
    column: *const c_char,
    value: f64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        // SAFETY: as above.
        let column = unsafe { arg_str(column, "column") }?;
        batch.set(column, Value::Float(value))
    })
}

/// `int flats_batch_text(flats_batch *b, const char *column, const char *utf8, size_t len)`
///
/// # Safety
/// `b` is NULL or a live batch; `column` is NULL or a NUL-terminated string;
/// `utf8` is NULL or valid for `len` bytes (it need not be NUL-terminated).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_text(
    b: *mut flats_batch,
    column: *const c_char,
    utf8: *const c_char,
    len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        // SAFETY: as above.
        let column = unsafe { arg_str(column, "column") }?;
        if utf8.is_null() {
            return Err(Fail::invalid("text is NULL"));
        }
        // SAFETY: non-NULL and valid for `len` bytes by contract.
        let bytes = unsafe { std::slice::from_raw_parts(utf8.cast::<u8>(), len) };
        let text = std::str::from_utf8(bytes)
            .map_err(|_| Fail::invalid(format!("column {column:?}: text is not valid UTF-8")))?;
        batch.set(column, Value::Text(text.to_string()))
    })
}

/// `int flats_batch_column_type(const flats_batch *b, const char *column)`
///
/// # Safety
/// `b` is NULL or a live batch; `column` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_column_type(
    b: *const flats_batch,
    column: *const c_char,
) -> i32 {
    // SAFETY: NULL or live, by contract.
    let Some(batch) = (unsafe { b.as_ref() }) else {
        return 0;
    };
    // SAFETY: forwarded from this function's own contract. Not run through
    // `guard`: an answer of 0 is the documented "no such column", not an
    // error to record.
    let Ok(column) = (unsafe { arg_str(column, "column") }) else {
        return 0;
    };
    if column == batch.vector_name {
        return FLATS_TYPE_VECTOR;
    }
    match batch.columns.get(column) {
        Some((_, ColumnType::Int)) => FLATS_TYPE_INT,
        Some((_, ColumnType::Float)) => FLATS_TYPE_FLOAT,
        Some((_, ColumnType::Text)) => FLATS_TYPE_TEXT,
        None => 0,
    }
}

/// `int flats_batch_end_row(flats_batch *b)`
///
/// # Safety
/// `b` is NULL or a live batch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_end_row(b: *mut flats_batch) -> i32 {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        let missing: Vec<&str> = batch
            .values
            .iter()
            .zip(&batch.names)
            .filter(|(v, _)| v.is_none())
            .map(|(_, n)| n.as_str())
            .collect();
        if batch.vector.is_none() || !missing.is_empty() {
            let mut what: Vec<&str> = missing;
            if batch.vector.is_none() {
                what.insert(0, "the vector");
            }
            let message = format!("row is missing {}", what.join(", "));
            // The half-built row is abandoned, so the next one starts clean.
            batch.vector = None;
            batch.values.iter_mut().for_each(|v| *v = None);
            return Err(Fail::invalid(message));
        }
        let vector = batch.vector.take().unwrap_or_default();
        let row: Row = batch
            .values
            .iter_mut()
            .enumerate()
            .filter_map(|(id, v)| v.take().map(|v| (id as ColumnId, v)))
            .collect();
        batch.rows.push((vector, row));
        Ok(())
    })
}

/// `size_t flats_batch_len(const flats_batch *b)`
///
/// # Safety
/// `b` is NULL or a live batch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_len(b: *const flats_batch) -> usize {
    // SAFETY: NULL or live, by contract.
    unsafe { b.as_ref() }.map_or(0, |b| b.rows.len())
}

/// `int flats_batch_commit(flats_batch *b, size_t *inserted)`
///
/// # Safety
/// `b` is NULL or a live batch whose database is still open; `inserted` is
/// NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_commit(b: *mut flats_batch, inserted: *mut usize) -> i32 {
    if !inserted.is_null() {
        // SAFETY: non-NULL and writable by contract.
        unsafe { *inserted = 0 };
    }
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let batch = unsafe { arg_mut(b, "batch") }?;
        let rows = std::mem::take(&mut batch.rows);
        let count = rows.len();
        // SAFETY: the header requires a batch's database to outlive it.
        let handle = unsafe { arg_ref(batch.db, "the batch's database") }?;
        handle
            .db
            .insert_many(batch.collection, rows)
            .map_err(Fail::engine)?;
        if !inserted.is_null() {
            // SAFETY: non-NULL and writable by contract.
            unsafe { *inserted = count };
        }
        Ok(())
    })
}

/// `void flats_batch_free(flats_batch *b)`
///
/// # Safety
/// `b` is NULL or a live batch, not used again after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flats_batch_free(b: *mut flats_batch) {
    if !b.is_null() {
        // SAFETY: a live batch from `flats_batch_new` (a `Box`), released once.
        drop(unsafe { Box::from_raw(b) });
    }
}
