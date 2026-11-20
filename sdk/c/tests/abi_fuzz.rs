//! The C ABI under random call sequences: random functions, random (but
//! in-contract) arguments, handles created and freed in random order —
//! including closing the database while batches and cursors are alive.
//!
//! "In contract" means what `flats.h` promises to survive: NULL where NULL is
//! documented as refused, any bytes as a C string (invalid UTF-8 included),
//! out-of-range row/column indices, any lengths that match their buffers. Out
//! of contract (a dangling pointer, a length longer than its buffer) is the
//! caller's undefined behaviour and is not exercised.
//!
//! The property: no crash, every status is a documented code, every failure
//! leaves a message, every accessor answers. Run under AddressSanitizer in CI
//! (the `sanitizers` job), where a memory error is a failure too.
//! `FLATS_FUZZ_ITERS` scales the number of sequences.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use flats_c::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

const STATUSES: [i32; 6] = [
    FLATS_OK,
    FLATS_ERR_INVALID_ARGUMENT,
    FLATS_ERR_QUERY,
    FLATS_ERR_ENGINE,
    FLATS_ERR_INTERNAL,
    FLATS_ERR_INTERRUPTED,
];

fn check(rc: i32, what: &str) {
    assert!(STATUSES.contains(&rc), "{what}: undocumented status {rc}");
    assert_ne!(rc, FLATS_ERR_INTERNAL, "{what}: an internal error is a bug");
    if rc != FLATS_OK {
        let msg = flats_last_error();
        assert!(!msg.is_null(), "{what}: failed with no message");
        // SAFETY: as above.
        let _ = unsafe { CStr::from_ptr(msg) }.to_bytes().len();
    }
}

const SQL: &[&str] = &[
    "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), s TEXT, f FLOAT) WITH (capacity = 64);",
    "INSERT INTO t (id, vector, s, f) VALUES (1, [1, 0], 'a', 0.5);",
    "INSERT INTO t (id, vector, s, f) VALUES (2, [0, 1], 'b', 1), (3, [1, 1], 'c', 2);",
    "UPSERT INTO t (id, vector, s, f) VALUES (1, [2, 2], 'z', 9.5);",
    "SELECT * FROM t;",
    "SELECT id, vector FROM t WHERE f > 0.7 LIMIT 2 OFFSET 1;",
    "SEARCH TOP 3 NEAREST TO [1, 0] FROM t RETURNING id, score, vector;",
    "UPDATE t SET s = 'q' WHERE id = 2;",
    "DELETE FROM t WHERE id = 3;",
    "DROP COLLECTION t;",
    "SELECT nope FROM t;",
    "garbage ;; ' ",
    "SEARCH TOP 0 NEAREST TO [1] FROM t;",
    "SELECT * FROM t WHERE id = ?;",
    "SEARCH TOP ? NEAREST TO ? FROM t WHERE s = ?;",
];

/// A C string from bytes, possibly invalid UTF-8 (never containing NUL).
fn cbytes(rng: &mut Rng, base: &str) -> CString {
    let mut b = base.as_bytes().to_vec();
    if rng.below(8) == 0 {
        let at = rng.below(b.len() as u64 + 1) as usize;
        b.insert(at, *rng.pick(&[0xFF, 0xC3, 0x80, 0xE2]));
    }
    CString::new(b).unwrap()
}

/// Owned storage for one random parameter array (the C structs point into it).
struct Params {
    texts: Vec<Vec<u8>>,
    vectors: Vec<Vec<f32>>,
    raw: Vec<flats_param>,
}

fn params(rng: &mut Rng) -> Params {
    let n = rng.below(4) as usize;
    let mut p = Params {
        texts: Vec::new(),
        vectors: Vec::new(),
        raw: Vec::new(),
    };
    for _ in 0..n {
        match rng.below(5) {
            0 => p
                .texts
                .push(rng.pick(&[&b"a"[..], b"", b"\xff\xfe", b"it's"]).to_vec()),
            1 => p
                .vectors
                .push((0..rng.below(4)).map(|i| i as f32).collect()),
            _ => {}
        }
    }
    for i in 0..n {
        // SAFETY: an all-zero flats_param is a valid value of the C struct.
        let mut c: flats_param = unsafe { std::mem::zeroed() };
        match rng.below(6) {
            0 => {
                c.kind = FLATS_TYPE_INT;
                c.int_value = rng.next() as i64;
            }
            1 => {
                c.kind = FLATS_TYPE_FLOAT;
                c.float_value = *rng.pick(&[0.5, f64::NAN, f64::INFINITY, -0.0]);
            }
            2 if !p.texts.is_empty() => {
                let t = &p.texts[i % p.texts.len()];
                c.kind = FLATS_TYPE_TEXT;
                c.text = t.as_ptr() as *const c_char;
                c.text_len = t.len();
            }
            3 if !p.vectors.is_empty() => {
                let v = &p.vectors[i % p.vectors.len()];
                c.kind = FLATS_TYPE_VECTOR;
                c.vector = v.as_ptr();
                c.vector_len = v.len();
            }
            4 => {
                c.kind = FLATS_TYPE_TEXT; // NULL text with zero length: refused, not read
                c.text = ptr::null();
                c.text_len = 0;
            }
            _ => c.kind = 99, // an unknown type tag: refused
        }
        p.raw.push(c);
    }
    p
}

/// Read every cell of a result through every accessor, plus out-of-range ones.
///
/// # Safety
/// `res` is NULL or a live result.
unsafe fn read_all(res: *const flats_result, rng: &mut Rng) {
    // SAFETY: forwarded.
    unsafe {
        let cols = flats_result_column_count(res);
        let rows = flats_result_row_count(res);
        for c in 0..cols + 2 {
            let name = flats_result_column_name(res, c);
            if c < cols {
                assert!(!name.is_null());
            }
        }
        for r in 0..rows + 1 {
            for c in 0..cols + 1 {
                let _ = flats_result_type(res, r, c);
                let _ = flats_result_int(res, r, c);
                let _ = flats_result_float(res, r, c);
                let mut len = 0;
                let t = flats_result_text(res, r, c, &mut len);
                if !t.is_null() {
                    assert_eq!(CStr::from_ptr(t).to_bytes().len(), len);
                }
                let v = flats_result_vector(res, r, c, &mut len);
                if !v.is_null() {
                    let s = std::slice::from_raw_parts(v, len);
                    let _ = s.iter().sum::<f32>();
                }
            }
        }
        let _ = flats_result_text(
            res,
            rng.next() as usize,
            rng.next() as usize,
            ptr::null_mut(),
        );
    }
}

fn sequence(seed: u64) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("db").to_str().unwrap()).unwrap();
    let mut db: *mut flats_db = ptr::null_mut();
    let mut batches: Vec<*mut flats_batch> = Vec::new();
    let mut cursors: Vec<*mut flats_cursor> = Vec::new();
    let mut results: Vec<*mut flats_result> = Vec::new();

    // SAFETY: every pointer passed is NULL (refused by contract), a live
    // handle from this sequence, or valid storage owned right here.
    unsafe {
        for step in 0..120 {
            let what = format!("seed {seed} step {step}");
            match rng.below(16) {
                0 => {
                    if db.is_null() {
                        let opener = if rng.below(5) == 0 {
                            flats_open_salvage
                        } else {
                            flats_open
                        };
                        check(opener(path.as_ptr(), &mut db), &what);
                    } else if rng.below(4) == 0 {
                        check(flats_close(db), &what); // batches/cursors may outlive it
                        db = ptr::null_mut();
                    }
                }
                1..=4 => {
                    let sql = {
                        let base = rng.clone_pick(SQL);
                        cbytes(&mut rng, base)
                    };
                    let mut res = ptr::null_mut();
                    let out = if rng.below(4) == 0 {
                        ptr::null_mut()
                    } else {
                        &mut res
                    };
                    check(flats_execute(db, sql.as_ptr(), out), &what);
                    if !res.is_null() {
                        results.push(res);
                    }
                }
                5 => {
                    let sql = {
                        let base = rng.clone_pick(SQL);
                        cbytes(&mut rng, base)
                    };
                    let p = params(&mut rng);
                    let mut res = ptr::null_mut();
                    check(
                        flats_execute_params(
                            db,
                            sql.as_ptr(),
                            p.raw.as_ptr(),
                            p.raw.len(),
                            &mut res,
                        ),
                        &what,
                    );
                    if !res.is_null() {
                        results.push(res);
                    }
                }
                6 => {
                    let sql = {
                        let base = rng.clone_pick(SQL);
                        cbytes(&mut rng, base)
                    };
                    let p = params(&mut rng);
                    let mut cur = ptr::null_mut();
                    check(
                        flats_query(db, sql.as_ptr(), p.raw.as_ptr(), p.raw.len(), &mut cur),
                        &what,
                    );
                    if !cur.is_null() {
                        cursors.push(cur);
                    }
                }
                7 if !cursors.is_empty() => {
                    let cur = cursors[rng.below(cursors.len() as u64) as usize];
                    let mut res = ptr::null_mut();
                    check(
                        flats_cursor_next(cur, rng.below(4) as usize, &mut res),
                        &what,
                    );
                    if !res.is_null() {
                        results.push(res);
                    }
                }
                8 => {
                    let name = {
                        let base = rng.clone_pick(&["t", "nope", ""]);
                        cbytes(&mut rng, base)
                    };
                    let mut b = ptr::null_mut();
                    check(flats_batch_new(db, name.as_ptr(), &mut b), &what);
                    if !b.is_null() {
                        batches.push(b);
                    }
                }
                9 if !batches.is_empty() => {
                    let b = batches[rng.below(batches.len() as u64) as usize];
                    let col = {
                        let base = rng.clone_pick(&["id", "s", "f", "vector", "nope"]);
                        cbytes(&mut rng, base)
                    };
                    let v = [1.0f32, 2.0, 3.0];
                    let rc = match rng.below(7) {
                        0 => flats_batch_vector(b, v.as_ptr(), rng.below(4) as usize),
                        1 => flats_batch_int(b, col.as_ptr(), rng.next() as i64),
                        2 => flats_batch_float(b, col.as_ptr(), *rng.pick(&[1.5, f64::NAN, 2.0])),
                        3 => flats_batch_text(
                            b,
                            col.as_ptr(),
                            b"x\xff".as_ptr() as *const c_char,
                            rng.below(3) as usize,
                        ),
                        4 => flats_batch_end_row(b),
                        5 => {
                            let mut n = 0;
                            if rng.below(2) == 0 {
                                flats_batch_commit(b, &mut n)
                            } else {
                                flats_batch_upsert(b, &mut n)
                            }
                        }
                        _ => {
                            let t = flats_batch_column_type(b, col.as_ptr());
                            assert!((0..=4).contains(&t), "{what}: column type {t}");
                            FLATS_OK
                        }
                    };
                    check(rc, &what);
                    let _ = flats_batch_len(b);
                }
                10 if !results.is_empty() => {
                    let res = results[rng.below(results.len() as u64) as usize];
                    read_all(res, &mut rng);
                }
                11 => {
                    let mut res = ptr::null_mut();
                    let rc = if rng.below(2) == 0 {
                        flats_collections(db, &mut res)
                    } else {
                        let name = {
                            let base = rng.clone_pick(&["t", "nope"]);
                            cbytes(&mut rng, base)
                        };
                        flats_describe(db, name.as_ptr(), &mut res)
                    };
                    check(rc, &what);
                    if !res.is_null() {
                        results.push(res);
                    }
                }
                12 => {
                    check(flats_interrupt(db), &what);
                    check(
                        flats_set_statement_timeout_ms(db, *rng.pick(&[-1, 0, 1, 1000])),
                        &what,
                    );
                }
                13 if !results.is_empty() => {
                    let i = rng.below(results.len() as u64) as usize;
                    flats_result_free(results.swap_remove(i));
                }
                14 if !batches.is_empty() => {
                    let i = rng.below(batches.len() as u64) as usize;
                    flats_batch_free(batches.swap_remove(i));
                }
                15 if !cursors.is_empty() => {
                    let i = rng.below(cursors.len() as u64) as usize;
                    flats_cursor_free(cursors.swap_remove(i));
                }
                _ => {}
            }
        }
        // Everything still alive, in a random order, the database last or not.
        for r in results.drain(..) {
            read_all(r, &mut rng);
            flats_result_free(r);
        }
        if rng.below(2) == 0 && !db.is_null() {
            check(flats_close(db), "final close");
            db = ptr::null_mut();
        }
        for c in cursors.drain(..) {
            let mut res = ptr::null_mut();
            check(flats_cursor_next(c, 10, &mut res), "late cursor_next");
            flats_result_free(res);
            flats_cursor_free(c);
        }
        for b in batches.drain(..) {
            let mut n = 0;
            check(flats_batch_commit(b, &mut n), "late commit");
            flats_batch_free(b);
        }
        if !db.is_null() {
            check(flats_close(db), "close");
        }
        // NULLs everywhere a handle goes: refused or a no-op, never a crash.
        flats_result_free(ptr::null_mut());
        flats_batch_free(ptr::null_mut());
        flats_cursor_free(ptr::null_mut());
        check(flats_close(ptr::null_mut()), "close NULL");
    }
}

trait PickOwned {
    fn clone_pick<'a>(&mut self, xs: &'a [&'a str]) -> &'a str;
}
impl PickOwned for Rng {
    fn clone_pick<'a>(&mut self, xs: &'a [&'a str]) -> &'a str {
        xs[self.below(xs.len() as u64) as usize]
    }
}

#[test]
fn random_call_sequences_never_crash() {
    let iters: u64 = std::env::var("FLATS_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    for seed in 0..iters {
        sequence(seed);
    }
}
