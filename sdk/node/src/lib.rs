//! Node.js addon for Flats: N-API over the C ABI.
//!
//! N-API is itself a C ABI, exported by the `node` executable, so — as the
//! engine does for `mmap` — the handful of functions this addon calls are
//! declared by hand below instead of pulling in a binding crate. They resolve
//! against the host process when Node `dlopen`s the addon.
//!
//! Every exported function is a thin translation: JavaScript values in, one
//! `flats_c` call, JavaScript values out. The semantics — validation,
//! atomicity, errors — are the C ABI's, so all three SDKs agree by
//! construction. `index.js` wraps these raw functions in the public API.
//!
//! Like the C ABI crate, this is an `unsafe` quarantine of its own; each
//! block names the N-API or C ABI precondition it relies on.

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![allow(non_camel_case_types)]

use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use flats_c::*;

// ---------------------------------------------------------------------------
// N-API, declared by hand (node_api.h, N-API version 8).
// ---------------------------------------------------------------------------

type napi_env = *mut c_void;
type napi_value = *mut c_void;
type napi_callback_info = *mut c_void;
type napi_status = i32;
type napi_callback = unsafe extern "C" fn(napi_env, napi_callback_info) -> napi_value;
type napi_finalize = unsafe extern "C" fn(napi_env, *mut c_void, *mut c_void);
type napi_deferred = *mut c_void;
type napi_async_work = *mut c_void;
type napi_ref = *mut c_void;
type napi_async_execute_callback = unsafe extern "C" fn(napi_env, *mut c_void);
type napi_async_complete_callback = unsafe extern "C" fn(napi_env, napi_status, *mut c_void);

const NAPI_OK: napi_status = 0;
const NAPI_AUTO_LENGTH: usize = usize::MAX;

// napi_valuetype
const NAPI_NUMBER: i32 = 3;
const NAPI_STRING: i32 = 4;
const NAPI_OBJECT: i32 = 6;
const NAPI_BIGINT: i32 = 9;

// napi_typedarray_type
const NAPI_FLOAT32_ARRAY: i32 = 7;
const NAPI_FLOAT64_ARRAY: i32 = 8;

unsafe extern "C" {
    fn napi_create_function(
        env: napi_env,
        name: *const c_char,
        length: usize,
        cb: napi_callback,
        data: *mut c_void,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_set_named_property(
        env: napi_env,
        object: napi_value,
        name: *const c_char,
        value: napi_value,
    ) -> napi_status;
    fn napi_get_cb_info(
        env: napi_env,
        info: napi_callback_info,
        argc: *mut usize,
        argv: *mut napi_value,
        this_arg: *mut napi_value,
        data: *mut *mut c_void,
    ) -> napi_status;
    fn napi_typeof(env: napi_env, value: napi_value, result: *mut i32) -> napi_status;
    fn napi_get_value_string_utf8(
        env: napi_env,
        value: napi_value,
        buf: *mut c_char,
        bufsize: usize,
        result: *mut usize,
    ) -> napi_status;
    fn napi_get_value_double(env: napi_env, value: napi_value, result: *mut f64) -> napi_status;
    fn napi_get_value_bigint_int64(
        env: napi_env,
        value: napi_value,
        result: *mut i64,
        lossless: *mut bool,
    ) -> napi_status;
    fn napi_is_array(env: napi_env, value: napi_value, result: *mut bool) -> napi_status;
    fn napi_is_typedarray(env: napi_env, value: napi_value, result: *mut bool) -> napi_status;
    fn napi_get_array_length(env: napi_env, value: napi_value, result: *mut u32) -> napi_status;
    fn napi_get_element(
        env: napi_env,
        object: napi_value,
        index: u32,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_get_typedarray_info(
        env: napi_env,
        typedarray: napi_value,
        kind: *mut i32,
        length: *mut usize,
        data: *mut *mut c_void,
        arraybuffer: *mut napi_value,
        byte_offset: *mut usize,
    ) -> napi_status;
    fn napi_create_string_utf8(
        env: napi_env,
        s: *const c_char,
        length: usize,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_create_double(env: napi_env, value: f64, result: *mut napi_value) -> napi_status;
    fn napi_create_int64(env: napi_env, value: i64, result: *mut napi_value) -> napi_status;
    fn napi_create_bigint_int64(env: napi_env, value: i64, result: *mut napi_value) -> napi_status;
    fn napi_create_array_with_length(
        env: napi_env,
        length: usize,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_set_element(
        env: napi_env,
        object: napi_value,
        index: u32,
        value: napi_value,
    ) -> napi_status;
    fn napi_create_external(
        env: napi_env,
        data: *mut c_void,
        finalize_cb: Option<napi_finalize>,
        finalize_hint: *mut c_void,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_get_value_external(
        env: napi_env,
        value: napi_value,
        result: *mut *mut c_void,
    ) -> napi_status;
    fn napi_get_undefined(env: napi_env, result: *mut napi_value) -> napi_status;
    fn napi_throw_error(env: napi_env, code: *const c_char, msg: *const c_char) -> napi_status;
    fn napi_create_error(
        env: napi_env,
        code: napi_value,
        msg: napi_value,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_create_type_error(
        env: napi_env,
        code: napi_value,
        msg: napi_value,
        result: *mut napi_value,
    ) -> napi_status;
    fn napi_create_promise(
        env: napi_env,
        deferred: *mut napi_deferred,
        promise: *mut napi_value,
    ) -> napi_status;
    fn napi_resolve_deferred(
        env: napi_env,
        deferred: napi_deferred,
        value: napi_value,
    ) -> napi_status;
    fn napi_reject_deferred(
        env: napi_env,
        deferred: napi_deferred,
        value: napi_value,
    ) -> napi_status;
    fn napi_create_async_work(
        env: napi_env,
        async_resource: napi_value,
        async_resource_name: napi_value,
        execute: napi_async_execute_callback,
        complete: napi_async_complete_callback,
        data: *mut c_void,
        result: *mut napi_async_work,
    ) -> napi_status;
    fn napi_queue_async_work(env: napi_env, work: napi_async_work) -> napi_status;
    fn napi_delete_async_work(env: napi_env, work: napi_async_work) -> napi_status;
    fn napi_create_reference(
        env: napi_env,
        value: napi_value,
        initial_refcount: u32,
        result: *mut napi_ref,
    ) -> napi_status;
    fn napi_delete_reference(env: napi_env, reference: napi_ref) -> napi_status;
    fn napi_throw_type_error(env: napi_env, code: *const c_char, msg: *const c_char)
    -> napi_status;
}

// ---------------------------------------------------------------------------
// Errors: thrown as JS `Error`s whose `code` is the C status, which
// `index.js` turns into the matching error class.
// ---------------------------------------------------------------------------

enum Throw {
    /// A Flats failure: the C status and its message.
    Flats(i32, String),
    /// Wrong JavaScript argument type.
    Type(String),
    /// An N-API call failed.
    Napi(&'static str),
}

type Js<T> = Result<T, Throw>;

fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).unwrap_or_default()
}

/// The C ABI's last error, after a call returned `rc`.
fn flats_fail(rc: i32) -> Throw {
    let p = flats_last_error();
    let message = if p.is_null() {
        format!("flats error {rc}")
    } else {
        // SAFETY: flats_last_error returns NULL or a NUL-terminated string,
        // valid until the next flats_* call on this thread; copied at once.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    };
    Throw::Flats(rc, message)
}

fn check(rc: i32) -> Js<()> {
    if rc == FLATS_OK {
        Ok(())
    } else {
        Err(flats_fail(rc))
    }
}

fn napi(status: napi_status, what: &'static str) -> Js<()> {
    if status == NAPI_OK {
        Ok(())
    } else {
        Err(Throw::Napi(what))
    }
}

/// Throw `t` into JavaScript.
///
/// # Safety
/// `env` is the live environment of the current callback.
unsafe fn throw(env: napi_env, t: Throw) {
    let (code, message, is_type) = match t {
        Throw::Flats(rc, m) => (rc.to_string(), m, false),
        Throw::Type(m) => ("ERR_INVALID_ARG_TYPE".to_string(), m, true),
        Throw::Napi(what) => (
            FLATS_ERR_INTERNAL.to_string(),
            format!("N-API call failed: {what}"),
            false,
        ),
    };
    let (code, message) = (cstring(&code), cstring(&message));
    // SAFETY: live env; both strings NUL-terminated and alive for the call.
    unsafe {
        if is_type {
            napi_throw_type_error(env, code.as_ptr(), message.as_ptr());
        } else {
            napi_throw_error(env, code.as_ptr(), message.as_ptr());
        }
    }
}

/// Run a callback body: its value on success; on `Err` or panic, a thrown
/// JS error and `undefined`. A panic must never unwind into Node.
///
/// # Safety
/// `env` is the live environment of the current callback.
unsafe fn run(env: napi_env, body: impl FnOnce() -> Js<napi_value>) -> napi_value {
    let outcome = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| {
        Err(Throw::Flats(
            FLATS_ERR_INTERNAL,
            "internal error (panic)".into(),
        ))
    });
    match outcome {
        Ok(v) => v,
        Err(t) => {
            // SAFETY: forwarded from this function's own contract.
            unsafe { throw(env, t) };
            ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Argument and value helpers. Each `# Safety`: `env` is the live environment
// of the current callback and `v` a value from it.
// ---------------------------------------------------------------------------

/// The callback's first `N` arguments (missing ones are `undefined`).
///
/// # Safety
/// `env` and `info` are the current callback's.
unsafe fn args<const N: usize>(env: napi_env, info: napi_callback_info) -> Js<[napi_value; N]> {
    let mut argv = [ptr::null_mut(); N];
    let mut argc = N;
    // SAFETY: live env/info; argv has room for `argc` values.
    let status = unsafe {
        napi_get_cb_info(
            env,
            info,
            &mut argc,
            argv.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    napi(status, "napi_get_cb_info")?;
    Ok(argv)
}

/// # Safety
/// See the section note.
unsafe fn type_of(env: napi_env, v: napi_value) -> Js<i32> {
    let mut t = 0;
    // SAFETY: see the section note.
    napi(unsafe { napi_typeof(env, v, &mut t) }, "napi_typeof")?;
    Ok(t)
}

/// A JS string as a NUL-terminated C string.
///
/// # Safety
/// See the section note.
unsafe fn get_cstring(env: napi_env, v: napi_value, what: &str) -> Js<CString> {
    // SAFETY: see the section note.
    if unsafe { type_of(env, v) }? != NAPI_STRING {
        return Err(Throw::Type(format!("{what} must be a string")));
    }
    let mut len = 0;
    // SAFETY: a NULL buffer asks for the length only.
    napi(
        unsafe { napi_get_value_string_utf8(env, v, ptr::null_mut(), 0, &mut len) },
        "napi_get_value_string_utf8",
    )?;
    let mut buf = vec![0u8; len + 1];
    let mut written = 0;
    // SAFETY: `buf` has room for `len` bytes plus the NUL N-API writes.
    napi(
        unsafe {
            napi_get_value_string_utf8(
                env,
                v,
                buf.as_mut_ptr().cast::<c_char>(),
                len + 1,
                &mut written,
            )
        },
        "napi_get_value_string_utf8",
    )?;
    buf.truncate(written);
    CString::new(buf)
        .map_err(|_| Throw::Flats(FLATS_ERR_INVALID_ARGUMENT, format!("{what} holds a NUL")))
}

/// # Safety
/// See the section note.
unsafe fn get_f64(env: napi_env, v: napi_value) -> Js<f64> {
    let mut x = 0.0;
    // SAFETY: see the section note.
    napi(
        unsafe { napi_get_value_double(env, v, &mut x) },
        "napi_get_value_double",
    )?;
    Ok(x)
}

/// # Safety
/// See the section note.
unsafe fn get_external<T>(env: napi_env, v: napi_value, what: &str) -> Js<*mut T> {
    let mut p = ptr::null_mut();
    // SAFETY: see the section note.
    let status = unsafe { napi_get_value_external(env, v, &mut p) };
    if status != NAPI_OK || p.is_null() {
        return Err(Throw::Type(format!("{what} is not a Flats handle")));
    }
    Ok(p as *mut T)
}

/// # Safety
/// See the section note.
unsafe fn undefined(env: napi_env) -> Js<napi_value> {
    let mut v = ptr::null_mut();
    // SAFETY: see the section note.
    napi(
        unsafe { napi_get_undefined(env, &mut v) },
        "napi_get_undefined",
    )?;
    Ok(v)
}

/// # Safety
/// `env` is live; `s` valid for `len` bytes of UTF-8.
unsafe fn string(env: napi_env, s: *const c_char, len: usize) -> Js<napi_value> {
    let mut v = ptr::null_mut();
    // SAFETY: per this function's contract.
    napi(
        unsafe { napi_create_string_utf8(env, s, len, &mut v) },
        "napi_create_string_utf8",
    )?;
    Ok(v)
}

/// # Safety
/// See the section note.
unsafe fn number(env: napi_env, x: f64) -> Js<napi_value> {
    let mut v = ptr::null_mut();
    // SAFETY: see the section note.
    napi(
        unsafe { napi_create_double(env, x, &mut v) },
        "napi_create_double",
    )?;
    Ok(v)
}

/// An INT cell: a Number when it is exactly representable, else a BigInt.
///
/// # Safety
/// See the section note.
unsafe fn integer(env: napi_env, n: i64) -> Js<napi_value> {
    const SAFE: i64 = (1 << 53) - 1;
    let mut v = ptr::null_mut();
    // SAFETY: see the section note.
    let status = unsafe {
        if (-SAFE..=SAFE).contains(&n) {
            napi_create_int64(env, n, &mut v)
        } else {
            napi_create_bigint_int64(env, n, &mut v)
        }
    };
    napi(status, "napi_create_int64")?;
    Ok(v)
}

/// # Safety
/// See the section note.
unsafe fn array(env: napi_env, len: usize) -> Js<napi_value> {
    let mut v = ptr::null_mut();
    // SAFETY: see the section note.
    napi(
        unsafe { napi_create_array_with_length(env, len, &mut v) },
        "napi_create_array",
    )?;
    Ok(v)
}

/// # Safety
/// See the section note.
unsafe fn set(env: napi_env, arr: napi_value, i: usize, v: napi_value) -> Js<()> {
    let i = u32::try_from(i).map_err(|_| Throw::Napi("array index overflows u32"))?;
    // SAFETY: see the section note.
    napi(
        unsafe { napi_set_element(env, arr, i, v) },
        "napi_set_element",
    )
}

/// A JS array or Float32Array/Float64Array as `f32`s.
///
/// # Safety
/// See the section note.
unsafe fn get_vector(env: napi_env, v: napi_value) -> Js<Vec<f32>> {
    let mut typed = false;
    // SAFETY: see the section note.
    napi(
        unsafe { napi_is_typedarray(env, v, &mut typed) },
        "napi_is_typedarray",
    )?;
    if typed {
        let (mut kind, mut len, mut data) = (0, 0, ptr::null_mut());
        // SAFETY: see the section note; N-API points `data` at the first
        // element (byte offset applied), valid for `len` elements.
        napi(
            unsafe {
                napi_get_typedarray_info(
                    env,
                    v,
                    &mut kind,
                    &mut len,
                    &mut data,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            },
            "napi_get_typedarray_info",
        )?;
        return match kind {
            // SAFETY: a Float32Array's data is `len` f32s (just checked).
            NAPI_FLOAT32_ARRAY => {
                Ok(unsafe { std::slice::from_raw_parts(data as *const f32, len) }.to_vec())
            }
            // SAFETY: a Float64Array's data is `len` f64s (just checked).
            NAPI_FLOAT64_ARRAY => {
                Ok(
                    unsafe { std::slice::from_raw_parts(data as *const f64, len) }
                        .iter()
                        .map(|&x| x as f32)
                        .collect(),
                )
            }
            _ => Err(Throw::Type(
                "a vector must be an Array, Float32Array or Float64Array".into(),
            )),
        };
    }
    let mut is_array = false;
    // SAFETY: see the section note.
    napi(
        unsafe { napi_is_array(env, v, &mut is_array) },
        "napi_is_array",
    )?;
    if !is_array {
        return Err(Throw::Type(
            "a vector must be an Array, Float32Array or Float64Array".into(),
        ));
    }
    let mut len = 0u32;
    // SAFETY: see the section note.
    napi(
        unsafe { napi_get_array_length(env, v, &mut len) },
        "napi_get_array_length",
    )?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let mut el = ptr::null_mut();
        // SAFETY: see the section note; `i` is within the array.
        napi(
            unsafe { napi_get_element(env, v, i, &mut el) },
            "napi_get_element",
        )?;
        // SAFETY: as above.
        if unsafe { type_of(env, el) }? != NAPI_NUMBER {
            return Err(Throw::Type(format!("vector element {i} is not a number")));
        }
        // SAFETY: as above.
        out.push(unsafe { get_f64(env, el) }? as f32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Handles: a database or a batch, wrapped in a JS External whose finalizer
// releases it if JavaScript never did.
// ---------------------------------------------------------------------------

struct DbHandle {
    db: *mut flats_db,
}

struct BatchHandle {
    batch: *mut flats_batch,
}

struct CursorHandle {
    cursor: *mut flats_cursor,
}

unsafe extern "C" fn finalize_cursor(_env: napi_env, data: *mut c_void, _hint: *mut c_void) {
    // SAFETY: `data` is the `Box<CursorHandle>` from `js_query`; runs once.
    let handle = unsafe { Box::from_raw(data as *mut CursorHandle) };
    if !handle.cursor.is_null() {
        // SAFETY: a live cursor that `cursorFree` never released.
        unsafe { flats_cursor_free(handle.cursor) };
    }
}

unsafe extern "C" fn finalize_db(_env: napi_env, data: *mut c_void, _hint: *mut c_void) {
    // SAFETY: `data` is the `Box<DbHandle>` this addon created in `open`;
    // N-API runs the finalizer once.
    let handle = unsafe { Box::from_raw(data as *mut DbHandle) };
    if !handle.db.is_null() {
        // SAFETY: a live handle that `close` never released.
        unsafe { flats_close(handle.db) };
    }
}

unsafe extern "C" fn finalize_batch(_env: napi_env, data: *mut c_void, _hint: *mut c_void) {
    // SAFETY: `data` is the `Box<BatchHandle>` from `batch_new`; runs once.
    let handle = unsafe { Box::from_raw(data as *mut BatchHandle) };
    if !handle.batch.is_null() {
        // SAFETY: a live batch that `batch_free` never released.
        unsafe { flats_batch_free(handle.batch) };
    }
}

/// # Safety
/// `env` is live; `data` a `Box` the matching finalizer owns.
unsafe fn external(env: napi_env, data: *mut c_void, fin: napi_finalize) -> Js<napi_value> {
    let mut v = ptr::null_mut();
    // SAFETY: per this function's contract.
    napi(
        unsafe { napi_create_external(env, data, Some(fin), ptr::null_mut(), &mut v) },
        "napi_create_external",
    )?;
    Ok(v)
}

/// The open database behind a JS handle.
///
/// # Safety
/// See the section note.
unsafe fn db_of(env: napi_env, v: napi_value) -> Js<*mut flats_db> {
    // SAFETY: see the section note; the External holds a `DbHandle`.
    let handle = unsafe { get_external::<DbHandle>(env, v, "database") }?;
    // SAFETY: a live `DbHandle` (its finalizer has not run: `v` is alive).
    let db = unsafe { (*handle).db };
    if db.is_null() {
        return Err(Throw::Flats(
            FLATS_ERR_INVALID_ARGUMENT,
            "database is closed".into(),
        ));
    }
    Ok(db)
}

/// # Safety
/// See the section note.
unsafe fn batch_of(env: napi_env, v: napi_value) -> Js<*mut flats_batch> {
    // SAFETY: see the section note; the External holds a `BatchHandle`.
    let handle = unsafe { get_external::<BatchHandle>(env, v, "batch") }?;
    // SAFETY: live, as in `db_of`.
    let batch = unsafe { (*handle).batch };
    if batch.is_null() {
        return Err(Throw::Flats(
            FLATS_ERR_INVALID_ARGUMENT,
            "batch is freed".into(),
        ));
    }
    Ok(batch)
}

// ---------------------------------------------------------------------------
// The exported functions. Each `# Safety`: called by Node with a live
// `env` and `info`.
// ---------------------------------------------------------------------------

/// `version() -> string`
unsafe extern "C" fn js_version(env: napi_env, _info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe { run(env, || string(env, flats_version(), NAPI_AUTO_LENGTH)) }
}

/// `open(path) -> handle`
unsafe extern "C" fn js_open(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [path] = args::<1>(env, info)?;
            let path = get_cstring(env, path, "path")?;
            let mut db = ptr::null_mut();
            check(flats_open(path.as_ptr(), &mut db))?;
            let handle = Box::into_raw(Box::new(DbHandle { db })) as *mut c_void;
            external(env, handle, finalize_db)
        })
    }
}

/// `close(handle)` — checkpoints; idempotent.
unsafe extern "C" fn js_close(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h] = args::<1>(env, info)?;
            let handle = get_external::<DbHandle>(env, h, "database")?;
            let db = std::mem::replace(&mut (*handle).db, ptr::null_mut());
            if !db.is_null() {
                check(flats_close(db))?;
            }
            undefined(env)
        })
    }
}

/// `execute(handle, sql) -> [columns, rows]`
unsafe extern "C" fn js_execute(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, sql, params] = args::<3>(env, info)?;
            let db = db_of(env, h)?;
            let sql = get_cstring(env, sql, "sql")?;
            // Owned values first, then the C structs that point into them;
            // both live until the call returns.
            let owned = get_params(env, params)?;
            let c_params: Vec<flats_param> = owned.iter().map(OwnedParam::as_c).collect();
            let mut res = ptr::null_mut();
            check(flats_execute_params(
                db,
                sql.as_ptr(),
                c_params.as_ptr(),
                c_params.len(),
                &mut res,
            ))?;
            let out = result_to_js(env, res);
            flats_result_free(res);
            out
        })
    }
}

/// One parameter converted from JavaScript, owning what its `flats_param`
/// points into.
enum OwnedParam {
    Int(i64),
    Float(f64),
    Text(Vec<u8>),
    Vector(Vec<f32>),
}

impl OwnedParam {
    fn as_c(&self) -> flats_param {
        match self {
            OwnedParam::Int(n) => flats_param {
                kind: FLATS_TYPE_INT,
                int_value: *n,
                ..flats_param::default()
            },
            OwnedParam::Float(x) => flats_param {
                kind: FLATS_TYPE_FLOAT,
                float_value: *x,
                ..flats_param::default()
            },
            OwnedParam::Text(b) => flats_param {
                kind: FLATS_TYPE_TEXT,
                text: b.as_ptr().cast::<c_char>(),
                text_len: b.len(),
                ..flats_param::default()
            },
            OwnedParam::Vector(v) => flats_param {
                kind: FLATS_TYPE_VECTOR,
                vector: v.as_ptr(),
                vector_len: v.len(),
                ..flats_param::default()
            },
        }
    }
}

/// The parameter array of an `execute` call: `undefined` means none.
///
/// A Number is an INT when it is a safe integer and a FLOAT otherwise —
/// exactly how the same value would read if written as a literal; a BigInt is
/// an exact INT; a string is TEXT; an Array or Float32/64Array a VECTOR.
///
/// # Safety
/// See the section note.
unsafe fn get_params(env: napi_env, v: napi_value) -> Js<Vec<OwnedParam>> {
    // SAFETY: see the section note, for every N-API call below.
    unsafe {
        let ty = type_of(env, v)?;
        if ty == 0 {
            return Ok(Vec::new()); // undefined
        }
        let mut is_array = false;
        napi(napi_is_array(env, v, &mut is_array), "napi_is_array")?;
        if !is_array {
            return Err(Throw::Type("params must be an array".into()));
        }
        let mut len = 0u32;
        napi(
            napi_get_array_length(env, v, &mut len),
            "napi_get_array_length",
        )?;
        let mut out = Vec::with_capacity(len as usize);
        for i in 0..len {
            let mut el = ptr::null_mut();
            napi(napi_get_element(env, v, i, &mut el), "napi_get_element")?;
            let bad = || {
                Throw::Flats(
                    FLATS_ERR_INVALID_ARGUMENT,
                    format!("parameter {i}: not a V-SQL value (number, bigint, string or vector)"),
                )
            };
            out.push(match type_of(env, el)? {
                NAPI_NUMBER => {
                    let x = get_f64(env, el)?;
                    if x.fract() == 0.0 && x.abs() <= 9_007_199_254_740_991.0 {
                        OwnedParam::Int(x as i64)
                    } else {
                        OwnedParam::Float(x)
                    }
                }
                NAPI_BIGINT => {
                    let (mut n, mut lossless) = (0i64, false);
                    napi(
                        napi_get_value_bigint_int64(env, el, &mut n, &mut lossless),
                        "napi_get_value_bigint_int64",
                    )?;
                    if !lossless {
                        return Err(Throw::Flats(
                            FLATS_ERR_INVALID_ARGUMENT,
                            format!("parameter {i}: BigInt does not fit in 64 bits"),
                        ));
                    }
                    OwnedParam::Int(n)
                }
                NAPI_STRING => OwnedParam::Text(get_cstring(env, el, "parameter")?.into_bytes()),
                NAPI_OBJECT => match get_vector(env, el) {
                    Ok(vector) => OwnedParam::Vector(vector),
                    Err(_) => return Err(bad()),
                },
                _ => return Err(bad()),
            });
        }
        Ok(out)
    }
}

/// # Safety
/// `env` is live; `res` a live result.
unsafe fn result_to_js(env: napi_env, res: *const flats_result) -> Js<napi_value> {
    // SAFETY: per this function's contract, for every call below; indices are
    // in range by construction, and each pointer is copied into a JS value
    // before `res` is freed by the caller.
    unsafe {
        let ncols = flats_result_column_count(res);
        let columns = array(env, ncols)?;
        for c in 0..ncols {
            set(
                env,
                columns,
                c,
                string(env, flats_result_column_name(res, c), NAPI_AUTO_LENGTH)?,
            )?;
        }
        let nrows = flats_result_row_count(res);
        let rows = array(env, nrows)?;
        for r in 0..nrows {
            let row = array(env, ncols)?;
            for c in 0..ncols {
                let mut len = 0;
                let cell = match flats_result_type(res, r, c) {
                    FLATS_TYPE_INT => integer(env, flats_result_int(res, r, c))?,
                    FLATS_TYPE_FLOAT => number(env, flats_result_float(res, r, c))?,
                    FLATS_TYPE_TEXT => string(env, flats_result_text(res, r, c, &mut len), len)?,
                    FLATS_TYPE_VECTOR => {
                        let p = flats_result_vector(res, r, c, &mut len);
                        let v = array(env, len)?;
                        for (i, x) in std::slice::from_raw_parts(p, len).iter().enumerate() {
                            set(env, v, i, number(env, f64::from(*x))?)?;
                        }
                        v
                    }
                    other => {
                        return Err(Throw::Napi(if other == 0 {
                            "cell out of range"
                        } else {
                            "unknown cell type"
                        }));
                    }
                };
                set(env, row, c, cell)?;
            }
            set(env, rows, r, row)?;
        }
        let pair = array(env, 2)?;
        set(env, pair, 0, columns)?;
        set(env, pair, 1, rows)?;
        Ok(pair)
    }
}

/// `query(handle, sql, params?) -> cursor`
unsafe extern "C" fn js_query(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, sql, params] = args::<3>(env, info)?;
            let db = db_of(env, h)?;
            let sql = get_cstring(env, sql, "sql")?;
            let owned = get_params(env, params)?;
            let c_params: Vec<flats_param> = owned.iter().map(OwnedParam::as_c).collect();
            let mut cursor = ptr::null_mut();
            check(flats_query(
                db,
                sql.as_ptr(),
                c_params.as_ptr(),
                c_params.len(),
                &mut cursor,
            ))?;
            let handle = Box::into_raw(Box::new(CursorHandle { cursor })) as *mut c_void;
            external(env, handle, finalize_cursor)
        })
    }
}

/// `cursorNext(cursor, maxRows) -> [columns, rows]` — no rows when exhausted.
unsafe extern "C" fn js_cursor_next(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [c, max] = args::<2>(env, info)?;
            let handle = get_external::<CursorHandle>(env, c, "cursor")?;
            let cursor = (*handle).cursor;
            if cursor.is_null() {
                return Err(Throw::Flats(
                    FLATS_ERR_INVALID_ARGUMENT,
                    "cursor is closed".into(),
                ));
            }
            let max = get_f64(env, max)?;
            if max.fract() != 0.0 || max < 1.0 {
                return Err(Throw::Flats(
                    FLATS_ERR_INVALID_ARGUMENT,
                    "maxRows must be a positive integer".into(),
                ));
            }
            let mut res = ptr::null_mut();
            check(flats_cursor_next(cursor, max as usize, &mut res))?;
            let out = result_to_js(env, res);
            flats_result_free(res);
            out
        })
    }
}

/// `cursorFree(cursor)` — idempotent.
unsafe extern "C" fn js_cursor_free(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [c] = args::<1>(env, info)?;
            let handle = get_external::<CursorHandle>(env, c, "cursor")?;
            let cursor = std::mem::replace(&mut (*handle).cursor, ptr::null_mut());
            flats_cursor_free(cursor);
            undefined(env)
        })
    }
}

/// `backup(handle, dest)`
unsafe extern "C" fn js_backup(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, dest] = args::<2>(env, info)?;
            let db = db_of(env, h)?;
            let dest = get_cstring(env, dest, "dest")?;
            check(flats_backup(db, dest.as_ptr()))?;
            undefined(env)
        })
    }
}

/// `batchNew(handle, collection) -> batch`
unsafe extern "C" fn js_batch_new(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, name] = args::<2>(env, info)?;
            let db = db_of(env, h)?;
            let name = get_cstring(env, name, "collection")?;
            let mut batch = ptr::null_mut();
            check(flats_batch_new(db, name.as_ptr(), &mut batch))?;
            let handle = Box::into_raw(Box::new(BatchHandle { batch })) as *mut c_void;
            external(env, handle, finalize_batch)
        })
    }
}

/// `batchSet(batch, column, value)` — the value's JS type, and for numbers the
/// column's declared type, pick the C setter.
unsafe extern "C" fn js_batch_set(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b, column, value] = args::<3>(env, info)?;
            let batch = batch_of(env, b)?;
            let name = get_cstring(env, column, "column")?;
            let column_type = flats_batch_column_type(batch, name.as_ptr());
            let shown = name.to_string_lossy().into_owned();
            match type_of(env, value)? {
                _ if column_type == FLATS_TYPE_VECTOR => {
                    let v = get_vector(env, value)?;
                    check(flats_batch_vector(batch, v.as_ptr(), v.len()))?;
                }
                NAPI_STRING => {
                    let s = get_cstring(env, value, "value")?;
                    let bytes = s.as_bytes();
                    check(flats_batch_text(
                        batch,
                        name.as_ptr(),
                        bytes.as_ptr().cast::<c_char>(),
                        bytes.len(),
                    ))?;
                }
                NAPI_BIGINT => {
                    let (mut n, mut lossless) = (0i64, false);
                    napi(
                        napi_get_value_bigint_int64(env, value, &mut n, &mut lossless),
                        "napi_get_value_bigint_int64",
                    )?;
                    if !lossless {
                        return Err(Throw::Flats(
                            FLATS_ERR_INVALID_ARGUMENT,
                            format!("column {shown:?}: BigInt does not fit in 64 bits"),
                        ));
                    }
                    check(flats_batch_int(batch, name.as_ptr(), n))?;
                }
                NAPI_NUMBER => {
                    let x = get_f64(env, value)?;
                    if column_type == FLATS_TYPE_INT {
                        // Exactly an integer, and exactly representable.
                        if x.fract() != 0.0 || !x.is_finite() || x.abs() > 9_007_199_254_740_991.0 {
                            return Err(Throw::Flats(
                                FLATS_ERR_INVALID_ARGUMENT,
                                format!(
                                    "column {shown:?} is INT; {x} is not an integer (use a BigInt beyond 2^53)"
                                ),
                            ));
                        }
                        check(flats_batch_int(batch, name.as_ptr(), x as i64))?;
                    } else {
                        // FLOAT, or an unknown column — the C ABI names the
                        // problem in either case.
                        check(flats_batch_float(batch, name.as_ptr(), x))?;
                    }
                }
                NAPI_OBJECT => {
                    // An array for a column that is not the vector column.
                    return Err(Throw::Flats(
                        FLATS_ERR_INVALID_ARGUMENT,
                        format!("column {shown:?} does not take a vector"),
                    ));
                }
                // Booleans, null, undefined, symbols, functions.
                _ => {
                    return Err(Throw::Flats(
                        FLATS_ERR_INVALID_ARGUMENT,
                        format!(
                            "column {shown:?}: not a V-SQL value (INT, FLOAT, TEXT or a vector)"
                        ),
                    ));
                }
            }
            undefined(env)
        })
    }
}

/// `batchEndRow(batch)`
unsafe extern "C" fn js_batch_end_row(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b] = args::<1>(env, info)?;
            check(flats_batch_end_row(batch_of(env, b)?))?;
            undefined(env)
        })
    }
}

/// `batchLen(batch) -> number`
unsafe extern "C" fn js_batch_len(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b] = args::<1>(env, info)?;
            number(env, flats_batch_len(batch_of(env, b)?) as f64)
        })
    }
}

/// `batchCommit(batch) -> number`
unsafe extern "C" fn js_batch_commit(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b] = args::<1>(env, info)?;
            let mut n = 0;
            check(flats_batch_commit(batch_of(env, b)?, &mut n))?;
            number(env, n as f64)
        })
    }
}

/// `batchFree(batch)` — idempotent.
unsafe extern "C" fn js_batch_free(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b] = args::<1>(env, info)?;
            let handle = get_external::<BatchHandle>(env, b, "batch")?;
            let batch = std::mem::replace(&mut (*handle).batch, ptr::null_mut());
            flats_batch_free(batch);
            undefined(env)
        })
    }
}

// ---------------------------------------------------------------------------
// Async: the same calls, run on libuv's thread pool, settling a Promise.
//
// A job's `run` closure executes on a pool thread and touches NO N-API — it
// only calls the C ABI, which is thread-safe for a flats_db and used one
// thread at a time for a cursor or batch (index.js serializes those). The
// `complete` callback runs back on the JS thread, turns the outcome into
// JavaScript and settles the Promise. Every JS handle the job uses is held by
// a reference until then, so the garbage collector cannot finalize — and
// free — a database or cursor that a pool thread is still using.
// ---------------------------------------------------------------------------

/// A raw handle moved to a pool thread.
struct SendPtr<T>(*mut T);
// SAFETY: the C ABI permits a flats_db on any thread, and a cursor, batch or
// result on one thread at a time; the job owns its use of the handle until
// `complete` runs.
unsafe impl<T> Send for SendPtr<T> {}

/// What a pool-thread run produced, still in C terms.
enum Done {
    Unit,
    Count(usize),
    Result(SendPtr<flats_result>),
    Db(SendPtr<flats_db>),
    Cursor(SendPtr<flats_cursor>),
}

type Run = Box<dyn FnOnce() -> Js<Done> + Send>;

struct Job {
    run: Option<Run>,
    outcome: Option<Js<Done>>,
    deferred: napi_deferred,
    work: napi_async_work,
    keep: Vec<napi_ref>,
}

unsafe extern "C" fn job_execute(_env: napi_env, data: *mut c_void) {
    // SAFETY: `data` is the `Job` `spawn` boxed; only this callback touches it
    // until `job_complete`.
    let job = unsafe { &mut *(data as *mut Job) };
    if let Some(run) = job.run.take() {
        job.outcome = Some(catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|_| {
            Err(Throw::Flats(
                FLATS_ERR_INTERNAL,
                "internal error (panic)".into(),
            ))
        }));
    }
}

unsafe extern "C" fn job_complete(env: napi_env, _status: napi_status, data: *mut c_void) {
    // SAFETY: `data` is the boxed `Job`; `complete` runs once, on the JS thread.
    let job = unsafe { Box::from_raw(data as *mut Job) };
    // SAFETY: live env on the JS thread for every call below.
    unsafe {
        let outcome = job
            .outcome
            .unwrap_or(Err(Throw::Napi("async work was cancelled")));
        let settled = outcome.and_then(|done| done_to_js(env, done));
        match settled {
            Ok(v) => {
                napi_resolve_deferred(env, job.deferred, v);
            }
            Err(t) => {
                let err = error_value(env, t);
                napi_reject_deferred(env, job.deferred, err);
            }
        }
        for r in job.keep {
            napi_delete_reference(env, r);
        }
        napi_delete_async_work(env, job.work);
    }
}

/// # Safety
/// Live env, on the JS thread.
unsafe fn done_to_js(env: napi_env, done: Done) -> Js<napi_value> {
    // SAFETY: per this function's contract.
    unsafe {
        match done {
            Done::Unit => undefined(env),
            Done::Count(n) => number(env, n as f64),
            Done::Result(res) => {
                let v = result_to_js(env, res.0);
                flats_result_free(res.0);
                v
            }
            Done::Db(db) => {
                let handle = Box::into_raw(Box::new(DbHandle { db: db.0 })) as *mut c_void;
                external(env, handle, finalize_db)
            }
            Done::Cursor(cursor) => {
                let handle =
                    Box::into_raw(Box::new(CursorHandle { cursor: cursor.0 })) as *mut c_void;
                external(env, handle, finalize_cursor)
            }
        }
    }
}

/// A rejection value: an `Error` (or `TypeError`) with `code`, like `throw`.
///
/// # Safety
/// Live env, on the JS thread.
unsafe fn error_value(env: napi_env, t: Throw) -> napi_value {
    let (code, message, is_type) = match t {
        Throw::Flats(rc, m) => (rc.to_string(), m, false),
        Throw::Type(m) => ("ERR_INVALID_ARG_TYPE".to_string(), m, true),
        Throw::Napi(what) => (
            FLATS_ERR_INTERNAL.to_string(),
            format!("N-API call failed: {what}"),
            false,
        ),
    };
    let (code, message) = (cstring(&code), cstring(&message));
    // SAFETY: per this function's contract.
    unsafe {
        let (mut c, mut m, mut e) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        napi_create_string_utf8(env, code.as_ptr(), NAPI_AUTO_LENGTH, &mut c);
        napi_create_string_utf8(env, message.as_ptr(), NAPI_AUTO_LENGTH, &mut m);
        if is_type {
            napi_create_type_error(env, c, m, &mut e);
        } else {
            napi_create_error(env, c, m, &mut e);
        }
        e
    }
}

/// Queue `run` on the thread pool, holding `keep` alive until it settles, and
/// return its Promise.
///
/// # Safety
/// Live env, on the JS thread; `keep` are values of this callback.
unsafe fn spawn(env: napi_env, keep: &[napi_value], run: Run) -> Js<napi_value> {
    // SAFETY: per this function's contract, for every call below.
    unsafe {
        let (mut deferred, mut promise) = (ptr::null_mut(), ptr::null_mut());
        napi(
            napi_create_promise(env, &mut deferred, &mut promise),
            "napi_create_promise",
        )?;
        let mut refs = Vec::with_capacity(keep.len());
        for v in keep {
            let mut r = ptr::null_mut();
            napi(
                napi_create_reference(env, *v, 1, &mut r),
                "napi_create_reference",
            )?;
            refs.push(r);
        }
        let job = Box::into_raw(Box::new(Job {
            run: Some(run),
            outcome: None,
            deferred,
            work: ptr::null_mut(),
            keep: refs,
        }));
        let name = string(env, c"flats".as_ptr(), NAPI_AUTO_LENGTH)?;
        let mut work = ptr::null_mut();
        let created = napi_create_async_work(
            env,
            ptr::null_mut(),
            name,
            job_execute,
            job_complete,
            job as *mut c_void,
            &mut work,
        );
        if created != NAPI_OK {
            let job = Box::from_raw(job);
            for r in job.keep {
                napi_delete_reference(env, r);
            }
            return Err(Throw::Napi("napi_create_async_work"));
        }
        (*job).work = work;
        if napi_queue_async_work(env, work) != NAPI_OK {
            let job = Box::from_raw(job);
            for r in job.keep {
                napi_delete_reference(env, r);
            }
            napi_delete_async_work(env, work);
            return Err(Throw::Napi("napi_queue_async_work"));
        }
        Ok(promise)
    }
}

/// `openAsync(path) -> Promise<handle>`
unsafe extern "C" fn js_open_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [path] = args::<1>(env, info)?;
            let path = get_cstring(env, path, "path")?;
            spawn(
                env,
                &[],
                Box::new(move || {
                    let mut db = ptr::null_mut();
                    check(flats_open(path.as_ptr(), &mut db))?;
                    Ok(Done::Db(SendPtr(db)))
                }),
            )
        })
    }
}

/// `closeAsync(handle) -> Promise<void>` — idempotent.
unsafe extern "C" fn js_close_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h] = args::<1>(env, info)?;
            let handle = get_external::<DbHandle>(env, h, "database")?;
            // Taken on the JS thread, so no later call can see it.
            let db = SendPtr(std::mem::replace(&mut (*handle).db, ptr::null_mut()));
            spawn(
                env,
                &[h],
                Box::new(move || {
                    let db = db;
                    if !db.0.is_null() {
                        check(flats_close(db.0))?;
                    }
                    Ok(Done::Unit)
                }),
            )
        })
    }
}

/// `executeAsync(handle, sql, params?) -> Promise<[columns, rows]>`
unsafe extern "C" fn js_execute_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, sql, params] = args::<3>(env, info)?;
            let db = SendPtr(db_of(env, h)?);
            let sql = get_cstring(env, sql, "sql")?;
            let owned = get_params(env, params)?;
            spawn(
                env,
                &[h],
                Box::new(move || {
                    let db = db;
                    let c_params: Vec<flats_param> = owned.iter().map(OwnedParam::as_c).collect();
                    let mut res = ptr::null_mut();
                    check(flats_execute_params(
                        db.0,
                        sql.as_ptr(),
                        c_params.as_ptr(),
                        c_params.len(),
                        &mut res,
                    ))?;
                    Ok(Done::Result(SendPtr(res)))
                }),
            )
        })
    }
}

/// `queryAsync(handle, sql, params?) -> Promise<cursor>`
unsafe extern "C" fn js_query_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, sql, params] = args::<3>(env, info)?;
            let db = SendPtr(db_of(env, h)?);
            let sql = get_cstring(env, sql, "sql")?;
            let owned = get_params(env, params)?;
            spawn(
                env,
                &[h],
                Box::new(move || {
                    let db = db;
                    let c_params: Vec<flats_param> = owned.iter().map(OwnedParam::as_c).collect();
                    let mut cursor = ptr::null_mut();
                    check(flats_query(
                        db.0,
                        sql.as_ptr(),
                        c_params.as_ptr(),
                        c_params.len(),
                        &mut cursor,
                    ))?;
                    Ok(Done::Cursor(SendPtr(cursor)))
                }),
            )
        })
    }
}

/// `cursorNextAsync(cursor, maxRows) -> Promise<[columns, rows]>`
unsafe extern "C" fn js_cursor_next_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [c, max] = args::<2>(env, info)?;
            let handle = get_external::<CursorHandle>(env, c, "cursor")?;
            let cursor = (*handle).cursor;
            if cursor.is_null() {
                return Err(Throw::Flats(
                    FLATS_ERR_INVALID_ARGUMENT,
                    "cursor is closed".into(),
                ));
            }
            let max = get_f64(env, max)?;
            if max.fract() != 0.0 || max < 1.0 {
                return Err(Throw::Flats(
                    FLATS_ERR_INVALID_ARGUMENT,
                    "maxRows must be a positive integer".into(),
                ));
            }
            let cursor = SendPtr(cursor);
            spawn(
                env,
                &[c],
                Box::new(move || {
                    let cursor = cursor;
                    let mut res = ptr::null_mut();
                    check(flats_cursor_next(cursor.0, max as usize, &mut res))?;
                    Ok(Done::Result(SendPtr(res)))
                }),
            )
        })
    }
}

/// `batchCommitAsync(batch) -> Promise<number>`
unsafe extern "C" fn js_batch_commit_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [b] = args::<1>(env, info)?;
            let batch = SendPtr(batch_of(env, b)?);
            spawn(
                env,
                &[b],
                Box::new(move || {
                    let batch = batch;
                    let mut n = 0;
                    check(flats_batch_commit(batch.0, &mut n))?;
                    Ok(Done::Count(n))
                }),
            )
        })
    }
}

/// `backupAsync(handle, dest) -> Promise<void>`
unsafe extern "C" fn js_backup_async(env: napi_env, info: napi_callback_info) -> napi_value {
    // SAFETY: see the section note.
    unsafe {
        run(env, || {
            let [h, dest] = args::<2>(env, info)?;
            let db = SendPtr(db_of(env, h)?);
            let dest = get_cstring(env, dest, "dest")?;
            spawn(
                env,
                &[h],
                Box::new(move || {
                    let db = db;
                    check(flats_backup(db.0, dest.as_ptr()))?;
                    Ok(Done::Unit)
                }),
            )
        })
    }
}

/// Node's entry point for an N-API addon.
///
/// # Safety
/// Called by Node with a live `env` and the module's `exports` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn napi_register_module_v1(env: napi_env, exports: napi_value) -> napi_value {
    let functions: [(&str, napi_callback); 21] = [
        ("openAsync", js_open_async),
        ("closeAsync", js_close_async),
        ("executeAsync", js_execute_async),
        ("queryAsync", js_query_async),
        ("cursorNextAsync", js_cursor_next_async),
        ("batchCommitAsync", js_batch_commit_async),
        ("backupAsync", js_backup_async),
        ("query", js_query),
        ("cursorNext", js_cursor_next),
        ("cursorFree", js_cursor_free),
        ("version", js_version),
        ("open", js_open),
        ("close", js_close),
        ("execute", js_execute),
        ("backup", js_backup),
        ("batchNew", js_batch_new),
        ("batchSet", js_batch_set),
        ("batchEndRow", js_batch_end_row),
        ("batchLen", js_batch_len),
        ("batchCommit", js_batch_commit),
        ("batchFree", js_batch_free),
    ];
    // SAFETY: live env and exports, per this function's contract.
    unsafe {
        run(env, || {
            for (name, cb) in functions {
                let cname = cstring(name);
                let mut f = ptr::null_mut();
                napi(
                    napi_create_function(
                        env,
                        cname.as_ptr(),
                        NAPI_AUTO_LENGTH,
                        cb,
                        ptr::null_mut(),
                        &mut f,
                    ),
                    "napi_create_function",
                )?;
                napi(
                    napi_set_named_property(env, exports, cname.as_ptr(), f),
                    "napi_set_named_property",
                )?;
            }
            Ok(exports)
        })
    }
}
