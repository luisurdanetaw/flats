//! Dependency-free fuzzing of every input Flats parses.
//!
//! Two attack surfaces, each fed hostile bytes:
//!
//! * **Statement text** — random token soup, random bytes, and mutations of
//!   valid statements, parsed AND executed against a live database.
//! * **Files on disk** — every file a database directory holds (WAL, catalog,
//!   both metadata snapshots, the vector index), mutated by bit flips,
//!   overwrites, truncation and extension, then opened and queried.
//!
//! The property is the same everywhere: an error is fine, a PANIC or a crash
//! is a bug, and a database that a bad statement could POISON is a bug. Each
//! case is reproducible from its seed, which every failure message names.
//!
//! Iterations scale with `FLATS_FUZZ_ITERS` (default keeps `cargo test`
//! quick); run e.g. `FLATS_FUZZ_ITERS=20000 cargo test --release --test fuzz`
//! for a long session.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use flats::{Db, DbOptions, Param};

fn iters(default: u64) -> u64 {
    std::env::var("FLATS_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// xorshift64*: deterministic, so a failing seed replays exactly.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: std::time::Duration::from_secs(3600),
        ..DbOptions::default()
    }
}

const SCHEMA: &str = "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(4), s TEXT, f FLOAT) \
                      WITH (capacity = 1000);";

/// A database with a bit of everything in it: inserts, an update, a delete.
fn populate(db: &Db) {
    db.execute(SCHEMA).unwrap();
    for i in 0..30i64 {
        db.execute_with(
            "INSERT INTO t (id, vector, s, f) VALUES (?, ?, ?, ?);",
            &[
                Param::Int(i),
                Param::Vector(vec![i as f32, 1.0, -0.5, 1e-7]),
                Param::Text(format!("row {i} 'quoted'")),
                Param::Float(i as f64 / 3.0),
            ],
        )
        .unwrap();
    }
    db.execute("UPDATE t SET s = 'changed' WHERE id = 3;")
        .unwrap();
    db.execute("DELETE FROM t WHERE id > 25;").unwrap();
}

/// Run a handful of reads and a write; errors are fine, panics are not.
fn exercise(db: &Db) {
    for sql in [
        "SELECT * FROM t;",
        "SELECT id, vector FROM t WHERE f > 1.5 OR s = 'changed';",
        "SEARCH TOP 5 NEAREST TO [1, 1, 0, 0] FROM t RETURNING id, score;",
    ] {
        if let Ok(stream) = db.execute(sql) {
            for row in stream {
                let _ = row;
            }
        }
    }
    let _ = db.execute("INSERT INTO t (id, vector, s, f) VALUES (999, [0, 0, 0, 1], 'late', 0.0);");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// Damage `bytes` one of several ways.
fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) -> &'static str {
    match rng.below(6) {
        0 if !bytes.is_empty() => {
            for _ in 0..=rng.below(8) {
                let i = rng.below(bytes.len() as u64) as usize;
                bytes[i] ^= 1 << rng.below(8);
            }
            "bit flips"
        }
        1 if !bytes.is_empty() => {
            let at = rng.below(bytes.len() as u64) as usize;
            let len = (rng.below(64) as usize).min(bytes.len() - at);
            for b in &mut bytes[at..at + len] {
                *b = rng.next() as u8;
            }
            "overwrite"
        }
        2 => {
            let keep = rng.below(bytes.len() as u64 + 1) as usize;
            bytes.truncate(keep);
            "truncate"
        }
        3 => {
            for _ in 0..rng.below(256) {
                bytes.push(rng.next() as u8);
            }
            "extend"
        }
        4 if bytes.len() >= 8 => {
            // Plausible-looking damage: a length or count field set huge.
            let at = rng.below(bytes.len() as u64 - 7) as usize;
            bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes()[..8]);
            "huge field"
        }
        _ => {
            bytes.iter_mut().for_each(|b| *b = 0);
            "zeroed"
        }
    }
}

/// Fuzz one file of a database directory.
fn fuzz_file(relative: &str, checkpointed: bool, seed_base: u64, n: u64) {
    let pristine = tempfile::tempdir().unwrap();
    {
        let db = Db::open(pristine.path(), &[], opts()).unwrap();
        populate(&db);
        if checkpointed {
            db.close().unwrap();
        }
        // else: dropped without close — the WAL keeps every record.
    }
    let original = std::fs::read(pristine.path().join(relative)).unwrap();
    for i in 0..n {
        let seed = seed_base + i;
        let mut rng = Rng::new(seed);
        let work = tempfile::tempdir().unwrap();
        let dir = work.path().join("db");
        copy_dir(pristine.path(), &dir);
        let mut bytes = original.clone();
        let how = mutate(&mut rng, &mut bytes);
        std::fs::write(dir.join(relative), &bytes).unwrap();

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if let Ok(db) = Db::open(&dir, &[], opts()) {
                exercise(&db);
            }
        }));
        assert!(
            outcome.is_ok(),
            "PANIC opening a database with {relative} damaged by {how} (seed {seed})"
        );
    }
}

#[test]
fn a_damaged_wal_never_panics() {
    fuzz_file("wal.log", false, 1_000_000, iters(150));
}

#[test]
fn a_damaged_catalog_never_panics() {
    fuzz_file("catalog.snap", true, 2_000_000, iters(150));
}

#[test]
fn a_damaged_metadata_snapshot_never_panics() {
    fuzz_file("collection-0/metadata.snap", true, 3_000_000, iters(150));
}

#[test]
fn a_damaged_tuple_snapshot_never_panics() {
    fuzz_file("collection-0/tuples.snap", true, 4_000_000, iters(150));
}

#[test]
fn a_damaged_vector_index_never_panics() {
    fuzz_file("collection-0/vectors.idx", true, 5_000_000, iters(150));
}

// -- statement text --------------------------------------------------------------

const VOCAB: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "INTO",
    "VALUES",
    "CREATE",
    "COLLECTION",
    "WITH",
    "SEARCH",
    "TOP",
    "NEAREST",
    "TO",
    "RETURNING",
    "AND",
    "OR",
    "NOT",
    "IN",
    "DELETE",
    "UPDATE",
    "SET",
    "DROP",
    "LIMIT",
    "OFFSET",
    "PRIMARY",
    "KEY",
    "VECTOR",
    "INT",
    "TEXT",
    "FLOAT",
    "t",
    "id",
    "vector",
    "s",
    "f",
    "score",
    "capacity",
    "(",
    ")",
    "[",
    "]",
    ",",
    ";",
    "*",
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "-",
    "?",
    "0",
    "1",
    "-1",
    "4",
    "1e-5",
    "1e999",
    "3.5e38",
    "0.5",
    "9223372036854775807",
    "99999999999999999999",
    "'x'",
    "'it''s'",
    "'",
    "--",
    "\0",
    "é",
];

fn token_soup(rng: &mut Rng) -> String {
    let n = 1 + rng.below(40);
    (0..n)
        .map(|_| *rng.pick(VOCAB))
        .collect::<Vec<_>>()
        .join(" ")
}

const VALID: &[&str] = &[
    "SELECT * FROM t WHERE id < 5 AND (f > 0.5 OR s = 'x') LIMIT 3 OFFSET 1;",
    "SEARCH TOP 5 NEAREST TO [1, 0, 0, 0] FROM t WHERE s != 'y' RETURNING id, score, vector;",
    "INSERT INTO t (id, vector, s, f) VALUES (100, [1, 2, 3, 4], 'a', 1.5), (101, [0, 0, 0, 1], 'b', -2e-3);",
    "UPDATE t SET s = 'z', f = 2.0 WHERE id = 1;",
    "UPDATE t SET vector = [4, 3, 2, 1] WHERE id = 2;",
    "DELETE FROM t WHERE f >= 100;",
    "CREATE COLLECTION u (k TEXT PRIMARY KEY, vector VECTOR(2), n INT) WITH (capacity = 10);",
];

fn mutate_text(rng: &mut Rng, src: &str) -> String {
    let mut s: Vec<u8> = src.as_bytes().to_vec();
    for _ in 0..=rng.below(4) {
        match rng.below(4) {
            0 if !s.is_empty() => {
                s.remove(rng.below(s.len() as u64) as usize);
            }
            1 => {
                let at = rng.below(s.len() as u64 + 1) as usize;
                let tok = rng.pick(VOCAB).as_bytes().to_vec();
                s.splice(at..at, tok);
            }
            2 if !s.is_empty() => {
                let at = rng.below(s.len() as u64) as usize;
                s[at] = rng.next() as u8;
            }
            _ => {
                let at = rng.below(s.len() as u64 + 1) as usize;
                s.truncate(at);
            }
        }
    }
    String::from_utf8_lossy(&s).into_owned()
}

fn random_params(rng: &mut Rng) -> Vec<Param> {
    (0..rng.below(4))
        .map(|_| match rng.below(5) {
            0 => Param::Int(rng.next() as i64),
            1 => Param::Float(f64::from_bits(rng.next())),
            2 => Param::Text("p'; DROP COLLECTION t; --".into()),
            3 => Param::Vector(
                (0..rng.below(6))
                    .map(|_| f32::from_bits(rng.next() as u32))
                    .collect(),
            ),
            _ => Param::Vector(vec![1.0, 0.0, 0.0, 0.0]),
        })
        .collect()
}

#[test]
fn hostile_statements_never_panic_or_poison() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &[], opts()).unwrap();
    populate(&db);
    let n = iters(3000);
    for i in 0..n {
        let seed = 10_000_000 + i;
        let mut rng = Rng::new(seed);
        let sql = match rng.below(3) {
            0 => token_soup(&mut rng),
            1 => {
                let base = *rng.pick(VALID);
                mutate_text(&mut rng, base)
            }
            _ => (0..rng.below(200))
                .map(|_| (rng.below(128) as u8) as char)
                .collect(),
        };
        let params = random_params(&mut rng);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if let Ok(stream) = db.execute_with(&sql, &params) {
                for row in stream.take(50) {
                    let _ = row;
                }
            }
        }));
        assert!(
            outcome.is_ok(),
            "PANIC on statement {sql:?} with {params:?} (seed {seed})"
        );
        assert!(
            !db.is_poisoned(),
            "statement {sql:?} (seed {seed}) poisoned the database"
        );
        // Keep the collection around for the statements that need it.
        if db.collection_id("t").is_err() {
            populate(&db);
        }
    }
    drop(db);
    let db = Db::open(dir.path(), &[], opts()).expect("the database reopens after the fuzzing");
    assert!(!db.is_poisoned());
}

// -- forged checksums -------------------------------------------------------------
//
// Random damage is almost always caught by a checksum, which exercises little
// past it. An attacker who crafts a file computes valid checksums, so these
// cases damage the CONTENT and then fix the CRC: every decoder behind a
// checksum sees hostile, well-framed input.

/// Mutate a snapshot's body, then re-seal it with a valid trailing CRC32.
fn forge_snapshot(rng: &mut Rng, bytes: &mut Vec<u8>) -> &'static str {
    if bytes.len() < 8 {
        return "too short to forge";
    }
    bytes.truncate(bytes.len() - 4);
    let how = mutate(rng, bytes);
    let crc = crc32fast::hash(bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    how
}

/// Mutate the payload of one WAL frame, then fix that frame's CRC (which
/// covers the LSN and the payload). Frames: [len u32][crc u32][lsn u64][payload].
fn forge_wal(rng: &mut Rng, bytes: &mut Vec<u8>) -> &'static str {
    let mut frames = Vec::new();
    let mut at = 8; // file header: magic + version
    while at + 16 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        if at + 16 + len > bytes.len() {
            break;
        }
        frames.push((at, len));
        at += 16 + len;
    }
    if frames.is_empty() {
        return "no frames";
    }
    let (start, len) = *rng.pick(&frames);
    let mut payload = bytes[start + 16..start + 16 + len].to_vec();
    let how = mutate(rng, &mut payload);
    let lsn = bytes[start + 8..start + 16].to_vec();
    let mut covered = lsn.clone();
    covered.extend_from_slice(&payload);
    let crc = crc32fast::hash(&covered);
    let mut frame = Vec::new();
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(&lsn);
    frame.extend_from_slice(&payload);
    bytes.splice(start..start + 16 + len, frame);
    how
}

fn fuzz_forged(relative: &str, checkpointed: bool, seed_base: u64, n: u64, wal: bool) {
    let pristine = tempfile::tempdir().unwrap();
    {
        let db = Db::open(pristine.path(), &[], opts()).unwrap();
        populate(&db);
        if checkpointed {
            db.close().unwrap();
        }
    }
    let original = std::fs::read(pristine.path().join(relative)).unwrap();
    let mut opened = std::collections::BTreeMap::new();
    let mut refused: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for i in 0..n {
        let seed = seed_base + i;
        let mut rng = Rng::new(seed);
        let work = tempfile::tempdir().unwrap();
        let dir = work.path().join("db");
        copy_dir(pristine.path(), &dir);
        let mut bytes = original.clone();
        let how = if wal {
            forge_wal(&mut rng, &mut bytes)
        } else {
            forge_snapshot(&mut rng, &mut bytes)
        };
        std::fs::write(dir.join(relative), &bytes).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| match Db::open(&dir, &[], opts()) {
            Ok(db) => {
                exercise(&db);
                *opened.entry("opened").or_insert(0) += 1;
            }
            Err(e) => {
                let kind = format!("{e:?}");
                let kind = kind.split(['(', ' ', '{']).next().unwrap_or("").to_string();
                *refused.entry(kind).or_insert(0) += 1;
            }
        }));
        assert!(
            outcome.is_ok(),
            "PANIC opening a database with {relative} forged ({how}, checksum fixed; seed {seed})"
        );
    }
    // What the forgeries reached: shown with --nocapture, so a harness whose
    // checksums stopped matching (and so tests nothing) is visible.
    eprintln!("{relative}: {opened:?} refused: {refused:?}");
}

#[test]
fn a_forged_catalog_never_panics() {
    fuzz_forged("catalog.snap", true, 6_000_000, iters(300), false);
}

#[test]
fn a_forged_metadata_snapshot_never_panics() {
    fuzz_forged(
        "collection-0/metadata.snap",
        true,
        7_000_000,
        iters(300),
        false,
    );
}

#[test]
fn a_forged_tuple_snapshot_never_panics() {
    fuzz_forged(
        "collection-0/tuples.snap",
        true,
        8_000_000,
        iters(300),
        false,
    );
}

#[test]
fn a_forged_wal_frame_never_panics() {
    fuzz_forged("wal.log", false, 9_000_000, iters(300), true);
}
