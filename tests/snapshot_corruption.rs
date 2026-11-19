//! What opening a database does when a collection's store file is damaged or
//! missing.
//!
//! The rule (the same one SQLite, Postgres, RocksDB and InnoDB follow): open
//! never trades data away silently.
//!
//! * If the write-ahead log provably still holds everything the store ever
//!   contained — nothing was checkpointed since the collection was created —
//!   the store is rebuilt from the log and the database opens with every row.
//! * Otherwise open REFUSES with `Error::CorruptSnapshot`, naming the
//!   collection and the file, and leaves every file on disk untouched.
//! * `Db::open_salvage` is the explicit, opt-in way past a refusal: the
//!   damaged collection opens with only the rows the log can prove (those
//!   written since the last checkpoint), and the database is usable again.

use std::path::Path;
use std::time::Duration;

use flats::{CompactionOptions, Db, DbOptions, Error, Param, RegValue};

fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
        compaction: CompactionOptions {
            enabled: false,
            ..Default::default()
        },
    }
}

fn salvage(dir: &Path) -> flats::Result<Db> {
    Db::open_salvage(dir, &[], opts())
}

fn open(dir: &Path) -> Db {
    Db::open(dir, &[], opts()).expect("open")
}

const SCHEMA: &str =
    "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), s TEXT) WITH (capacity = 100);";

fn insert(db: &Db, id: i64) {
    db.execute_with(
        "INSERT INTO t (id, vector, s) VALUES (?, ?, ?);",
        &[
            Param::Int(id),
            Param::Vector(vec![id as f32, 1.0]),
            Param::Text(format!("row {id}")),
        ],
    )
    .expect("insert");
}

/// Every live id, through both stores (`s` lives in the tuple store, the
/// filter goes through the metadata index).
fn ids(db: &Db) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .execute("SELECT id, s FROM t WHERE id >= 0;")
        .expect("select")
        .map(|row| match row.expect("row").0.as_slice() {
            [RegValue::Int(id), RegValue::Str(s)] => {
                assert_eq!(*s, format!("row {id}"));
                *id
            }
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

fn search_ids(db: &Db) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .execute("SEARCH TOP 100 NEAREST TO [1, 0] FROM t RETURNING id;")
        .expect("search")
        .map(|row| match row.expect("row").0.as_slice() {
            [RegValue::Int(id)] => *id,
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

fn store(dir: &Path, file: &str) -> std::path::PathBuf {
    dir.join("collection-0").join(file)
}

fn damage(path: &Path) {
    let mut bytes = std::fs::read(path).expect("read store file");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x5A;
    std::fs::write(path, bytes).expect("write store file");
}

/// Every file under `dir`, with its bytes: proof a refused open touched nothing.
fn tree(dir: &Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n != "LOCK") {
                out.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

fn refusal(dir: &Path) -> String {
    match Db::open(dir, &[], opts()) {
        Err(Error::CorruptSnapshot(why)) => why,
        Err(other) => panic!("expected CorruptSnapshot, got {other:?}"),
        Ok(db) => panic!("opened with rows {:?} instead of refusing", ids(&db)),
    }
}

/// Rows 0..10 checkpointed (the WAL is truncated past them), then rows 10..13
/// only in the WAL.
fn checkpointed_then_tail(dir: &Path) {
    let db = open(dir);
    db.execute(SCHEMA).unwrap();
    for id in 0..10 {
        insert(&db, id);
    }
    db.checkpoint().unwrap();
    for id in 10..13 {
        insert(&db, id);
    }
    drop(db); // a crash: no final checkpoint
}

// ---------------------------------------------------------------------------
// Refuse when the log can't rebuild the store
// ---------------------------------------------------------------------------

#[test]
fn a_damaged_store_after_a_clean_close_refuses_to_open() {
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            db.execute(SCHEMA).unwrap();
            for id in 0..10 {
                insert(&db, id);
            }
            db.close().unwrap();
        }
        damage(&store(dir.path(), victim));

        let why = refusal(dir.path());
        assert!(
            why.contains(victim),
            "{victim}: message names the file: {why}"
        );
        assert!(
            why.contains("'t'"),
            "{victim}: message names the collection: {why}"
        );
        assert!(
            why.contains("salvage"),
            "{victim}: message names the way out: {why}"
        );
    }
}

#[test]
fn a_damaged_store_after_a_mid_session_checkpoint_refuses_to_open() {
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        checkpointed_then_tail(dir.path());
        damage(&store(dir.path(), victim));
        let why = refusal(dir.path());
        assert!(why.contains(victim), "{why}");
    }
}

#[test]
fn a_missing_store_file_after_a_checkpoint_refuses_to_open() {
    for victim in ["metadata.snap", "tuples.snap", "vectors.idx"] {
        let dir = tempfile::tempdir().unwrap();
        checkpointed_then_tail(dir.path());
        std::fs::remove_file(store(dir.path(), victim)).unwrap();
        let why = refusal(dir.path());
        assert!(why.contains(victim), "{victim}: {why}");
        assert!(why.contains("missing"), "{victim}: {why}");
    }
}

#[test]
fn a_refused_open_leaves_every_file_untouched() {
    let dir = tempfile::tempdir().unwrap();
    checkpointed_then_tail(dir.path());
    damage(&store(dir.path(), "tuples.snap"));
    let before = tree(dir.path());

    refusal(dir.path());
    refusal(dir.path()); // and refuses the same way every time
    assert_eq!(tree(dir.path()), before);
}

#[test]
fn a_damaged_store_in_a_compacted_collection_refuses_even_with_a_full_log() {
    // A compaction renumbers rows. Records from before it name the OLD
    // numbering, so even a log that still holds them can't rebuild the new
    // generation from empty.
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            db.execute(SCHEMA).unwrap();
            for id in 0..10 {
                insert(&db, id);
            }
            db.execute("DELETE FROM t WHERE id < 5;").unwrap();
            assert!(db.compact_now(0).unwrap(), "something was dead");
            drop(db); // no checkpoint after the flip
        }
        damage(&store(dir.path(), victim));
        let why = refusal(dir.path());
        assert!(why.contains(victim), "{why}");
    }
}

#[test]
fn a_damaged_store_in_one_collection_names_that_collection() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(SCHEMA).unwrap();
        db.execute(
            "CREATE COLLECTION other (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);",
        )
        .unwrap();
        insert(&db, 1);
        db.execute("INSERT INTO other (id, vector) VALUES (1, [0, 1]);")
            .unwrap();
        db.close().unwrap();
    }
    damage(&dir.path().join("collection-1").join("metadata.snap"));
    let why = refusal(dir.path());
    assert!(why.contains("'other'"), "{why}");
    assert!(!why.contains("'t'"), "{why}");
}

#[test]
fn a_damaged_store_in_a_collection_that_was_never_written_is_rebuilt() {
    // An empty collection's stores stay at watermark 0, which proves the log
    // was never cut past anything it held: rebuilding it loses nothing.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(SCHEMA).unwrap();
        db.execute(
            "CREATE COLLECTION other (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);",
        )
        .unwrap();
        insert(&db, 1);
        db.close().unwrap();
    }
    damage(&dir.path().join("collection-1").join("metadata.snap"));
    let db = open(dir.path());
    assert_eq!(ids(&db), vec![1]);
    assert_eq!(db.execute("SELECT id FROM other;").unwrap().count(), 0);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Rebuild when the log provably can
// ---------------------------------------------------------------------------

#[test]
fn a_damaged_store_before_any_checkpoint_is_rebuilt_from_the_log() {
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            db.execute(SCHEMA).unwrap();
            for id in 0..10 {
                insert(&db, id);
            }
            db.execute("DELETE FROM t WHERE id = 4;").unwrap();
            db.execute("UPDATE t SET s = 'row 7' WHERE id = 7;")
                .unwrap();
            drop(db); // crash before any checkpoint: the WAL holds everything
        }
        damage(&store(dir.path(), victim));

        let expect: Vec<i64> = (0..10).filter(|&i| i != 4).collect();
        let db = open(dir.path());
        assert_eq!(ids(&db), expect, "{victim}");
        assert_eq!(search_ids(&db), expect, "{victim}");
        db.close().unwrap();

        // The rebuilt store was checkpointed: the next open is normal.
        let db = open(dir.path());
        assert_eq!(ids(&db), expect, "{victim}");
        db.close().unwrap();
    }
}

#[test]
fn a_missing_store_before_any_checkpoint_is_rebuilt_from_the_log() {
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            db.execute(SCHEMA).unwrap();
            for id in 0..5 {
                insert(&db, id);
            }
            drop(db);
        }
        std::fs::remove_file(store(dir.path(), victim)).unwrap();
        let db = open(dir.path());
        assert_eq!(ids(&db), vec![0, 1, 2, 3, 4], "{victim}");
        db.close().unwrap();
    }
}

#[test]
fn a_healthy_database_opens_normally_after_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    checkpointed_then_tail(dir.path());
    let db = open(dir.path());
    assert_eq!(ids(&db), (0..13).collect::<Vec<_>>());
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Salvage: the explicit way past a refusal
// ---------------------------------------------------------------------------

#[test]
fn salvage_keeps_what_the_log_proves_and_drops_the_rest() {
    for victim in ["metadata.snap", "tuples.snap"] {
        let dir = tempfile::tempdir().unwrap();
        checkpointed_then_tail(dir.path());
        damage(&store(dir.path(), victim));

        let db = salvage(dir.path()).expect("salvage opens");
        // Rows 0..10 were only in the damaged snapshot; 10..13 are in the log.
        assert_eq!(ids(&db), vec![10, 11, 12], "{victim}");
        assert_eq!(search_ids(&db), vec![10, 11, 12], "{victim}");
        // And the collection is fully usable again.
        insert(&db, 50);
        db.execute("DELETE FROM t WHERE id = 11;").unwrap();
        assert_eq!(ids(&db), vec![10, 12, 50], "{victim}");
        db.close().unwrap();

        // After a salvage, a normal open just works and sees the same rows.
        let db = open(dir.path());
        assert_eq!(ids(&db), vec![10, 12, 50], "{victim}");
        db.close().unwrap();
    }
}

#[test]
fn writes_after_a_salvage_survive_a_crash() {
    // The log's sequence numbers must keep climbing past the healthy stores'
    // checkpoints after a salvage. If they restarted low, the next recovery
    // would skip these writes as "already checkpointed".
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(SCHEMA).unwrap();
        for id in 0..10 {
            insert(&db, id);
        }
        db.close().unwrap(); // the log is now empty
    }
    damage(&store(dir.path(), "metadata.snap"));
    {
        let db = salvage(dir.path()).unwrap();
        assert_eq!(ids(&db), Vec::<i64>::new());
        for id in 20..23 {
            insert(&db, id);
        }
        drop(db); // crash
    }
    let db = open(dir.path());
    assert_eq!(ids(&db), vec![20, 21, 22]);
    assert_eq!(search_ids(&db), vec![20, 21, 22]);
    db.close().unwrap();
}

#[test]
fn salvage_of_a_compacted_collection_never_misplaces_rows() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(SCHEMA).unwrap();
        for id in 0..10 {
            insert(&db, id);
        }
        db.execute("DELETE FROM t WHERE id < 5;").unwrap();
        assert!(db.compact_now(0).unwrap());
        insert(&db, 30);
        drop(db);
    }
    damage(&store(dir.path(), "tuples.snap"));
    let db = salvage(dir.path()).unwrap();
    // Whatever survives must be a real row with its own text (ids() checks
    // that `s` matches `id`) — never a pre-compaction record replayed onto
    // renumbered rows.
    let got = ids(&db);
    assert!(
        got.iter().all(|id| (5..10).contains(id) || *id == 30),
        "{got:?}"
    );
    assert!(
        got.contains(&30),
        "the post-flip insert is in the log: {got:?}"
    );
    db.close().unwrap();
}

#[test]
fn salvage_on_a_healthy_database_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    checkpointed_then_tail(dir.path());
    let db = salvage(dir.path()).unwrap();
    assert_eq!(ids(&db), (0..13).collect::<Vec<_>>());
    db.close().unwrap();
}

#[test]
fn salvage_does_not_paper_over_a_damaged_catalog() {
    // The catalog is the list of collections. No log can rebuild it, and
    // salvage is about one collection's rows, not about guessing which
    // collections exist.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(SCHEMA).unwrap();
        db.close().unwrap();
    }
    damage(&dir.path().join("catalog.snap"));
    assert!(matches!(
        salvage(dir.path()),
        Err(Error::CorruptSnapshot(_))
    ));
}

// ---------------------------------------------------------------------------
// The property, fuzzed: never fewer rows without an error
// ---------------------------------------------------------------------------

/// xorshift64*, as in `tests/fuzz.rs`: a failing seed replays exactly.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

#[test]
fn a_damaged_store_never_opens_with_rows_missing() {
    let iters: u64 = std::env::var("FLATS_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let golden = tempfile::tempdir().unwrap();
    checkpointed_then_tail(golden.path());
    let expect: Vec<i64> = (0..13).collect();

    for seed in 0..iters {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let victim = ["metadata.snap", "tuples.snap"][rng.below(2) as usize];
        let dir = tempfile::tempdir().unwrap();
        for f in ["catalog.snap", "wal.log"] {
            std::fs::copy(golden.path().join(f), dir.path().join(f)).unwrap();
        }
        std::fs::create_dir(dir.path().join("collection-0")).unwrap();
        for f in ["metadata.snap", "tuples.snap", "vectors.idx"] {
            std::fs::copy(store(golden.path(), f), store(dir.path(), f)).unwrap();
        }

        let path = store(dir.path(), victim);
        let mut bytes = std::fs::read(&path).unwrap();
        match rng.below(3) {
            0 => {
                for _ in 0..=rng.below(8) {
                    let i = rng.below(bytes.len() as u64) as usize;
                    bytes[i] ^= 1 << rng.below(8);
                }
            }
            1 => bytes.truncate(rng.below(bytes.len() as u64) as usize),
            _ => {
                let i = rng.below(bytes.len() as u64) as usize;
                let n = (1 + rng.below(64) as usize).min(bytes.len() - i);
                for b in &mut bytes[i..i + n] {
                    *b = rng.next() as u8;
                }
            }
        }
        std::fs::write(&path, &bytes).unwrap();

        match Db::open(dir.path(), &[], opts()) {
            Ok(db) => {
                // A CRC-forged change can decode; a damaged file that opens
                // must still hold every row, or it should have refused.
                assert_eq!(
                    ids(&db),
                    expect,
                    "seed {seed}: {victim} opened with rows missing"
                );
                db.close().unwrap();
            }
            Err(Error::CorruptSnapshot(why)) => {
                assert!(why.contains(victim), "seed {seed}: {why}");
            }
            Err(other) => panic!("seed {seed}: {victim}: unexpected error {other:?}"),
        }
    }
}
