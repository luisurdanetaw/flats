//! The write-ahead log sheds what every store has checkpointed — including
//! when some collections are never, or rarely, written.
//!
//! Truncation stops at the lowest checkpoint watermark of any store. A store
//! that only advanced its watermark when its own collection was written would
//! sit at 0 forever in an idle collection and hold the whole log back: the
//! file grows without bound and every open replays all of it.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use flats::{CompactionOptions, Db, DbOptions, Param, RegValue};

fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
        compaction: CompactionOptions {
            enabled: false,
            ..Default::default()
        },
    }
}

fn open(dir: &Path) -> Db {
    Db::open(dir, &[], opts()).expect("open")
}

fn create(db: &Db, name: &str) {
    db.execute(&format!(
        "CREATE COLLECTION {name} (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 1000);"
    ))
    .expect("create");
}

fn insert(db: &Db, name: &str, id: i64) {
    db.execute_with(
        &format!("INSERT INTO {name} (id, vector) VALUES (?, ?);"),
        &[Param::Int(id), Param::Vector(vec![id as f32, 1.0])],
    )
    .expect("insert");
}

fn ids(db: &Db, name: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .execute(&format!("SELECT id FROM {name};"))
        .expect("select")
        .map(|row| match row.expect("row").0.as_slice() {
            [RegValue::Int(id)] => *id,
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

fn wal_len(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("wal.log")).map_or(0, |m| m.len())
}

/// The log of a database whose every record has been checkpointed: just the
/// file header.
fn empty_wal_len() -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, "a");
    insert(&db, "a", 1);
    db.close().unwrap();
    wal_len(dir.path())
}

#[test]
fn an_idle_collection_does_not_hold_the_log_back() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, "busy");
    create(&db, "idle");
    for id in 0..200 {
        insert(&db, "busy", id);
    }
    db.checkpoint().unwrap();
    assert_eq!(
        wal_len(dir.path()),
        empty_wal_len(),
        "checkpoint sheds the log"
    );
    db.close().unwrap();
    assert_eq!(wal_len(dir.path()), empty_wal_len(), "close sheds the log");
}

#[test]
fn a_collection_written_once_does_not_hold_the_log_back() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, "busy");
    create(&db, "rare");
    insert(&db, "rare", 1);
    for id in 0..200 {
        insert(&db, "busy", id);
    }
    db.close().unwrap();
    assert_eq!(wal_len(dir.path()), empty_wal_len());
}

#[test]
fn the_log_stays_bounded_across_sessions() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        create(&db, "busy");
        create(&db, "idle");
        db.close().unwrap();
    }
    for session in 0..5 {
        let db = open(dir.path());
        for i in 0..50 {
            insert(&db, "busy", session * 50 + i);
        }
        db.close().unwrap();
        assert_eq!(wal_len(dir.path()), empty_wal_len(), "session {session}");
    }
    let db = open(dir.path());
    assert_eq!(ids(&db, "busy"), (0..250).collect::<Vec<_>>());
    assert_eq!(ids(&db, "idle"), Vec::<i64>::new());
    db.close().unwrap();
}

#[test]
fn an_idle_collection_written_after_a_checkpoint_survives_a_crash() {
    // Its watermark moved past records it never saw; its first real write
    // must land above that, and replay must not skip it.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        create(&db, "busy");
        create(&db, "idle");
        for id in 0..20 {
            insert(&db, "busy", id);
        }
        db.close().unwrap();
    }
    {
        let db = open(dir.path());
        insert(&db, "idle", 7);
        insert(&db, "busy", 100);
        drop(db); // crash
    }
    let db = open(dir.path());
    assert_eq!(ids(&db, "idle"), vec![7]);
    assert_eq!(ids(&db, "busy").len(), 21);
    db.close().unwrap();
}

/// xorshift64*: a failing seed replays exactly.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn uneven_writes_checkpoints_and_crashes_never_lose_a_row() {
    // Three collections written at very different rates, with checkpoints,
    // clean closes, crashes, deletes and a late-created collection mixed in at
    // random. After every reopen the database must equal the model exactly.
    let seeds: u64 = std::env::var("FLATS_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    for seed in 0..seeds {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let dir = tempfile::tempdir().unwrap();
        let mut model: BTreeMap<&str, Vec<i64>> = BTreeMap::new();
        let mut db = open(dir.path());
        for name in ["hot", "warm", "cold"] {
            create(&db, name);
            model.insert(name, Vec::new());
        }
        let mut next_id = 0i64;
        for step in 0..300 {
            match rng.below(100) {
                0..=59 => {
                    // Weighted: hot ≫ warm ≫ cold.
                    let name = match rng.below(20) {
                        0 => "cold",
                        1..=4 => "warm",
                        _ => "hot",
                    };
                    if let Some(rows) = model.get_mut(name) {
                        insert(&db, name, next_id);
                        rows.push(next_id);
                        next_id += 1;
                    }
                }
                60..=69 => {
                    let names: Vec<&str> = model.keys().copied().collect();
                    let name = names[rng.below(names.len() as u64) as usize];
                    let rows = model.get_mut(name).unwrap();
                    if !rows.is_empty() {
                        let victim = rows.remove(rng.below(rows.len() as u64) as usize);
                        db.execute_with(
                            &format!("DELETE FROM {name} WHERE id = ?;"),
                            &[Param::Int(victim)],
                        )
                        .unwrap();
                    }
                }
                70..=79 => db.checkpoint().unwrap(),
                80..=84 => {
                    db.close().unwrap();
                    db = open(dir.path());
                }
                85..=92 => {
                    drop(db); // crash
                    db = open(dir.path());
                }
                93 if !model.contains_key("late") => {
                    create(&db, "late");
                    model.insert("late", Vec::new());
                }
                _ => {
                    let name = if model.contains_key("late") {
                        "late"
                    } else {
                        "cold"
                    };
                    insert(&db, name, next_id);
                    model.get_mut(name).unwrap().push(next_id);
                    next_id += 1;
                }
            }
            if step % 25 == 0 {
                for (name, rows) in &model {
                    let mut want = rows.clone();
                    want.sort_unstable();
                    assert_eq!(ids(&db, name), want, "seed {seed} step {step} {name}");
                }
            }
        }
        drop(db);
        let db = open(dir.path());
        for (name, rows) in &model {
            let mut want = rows.clone();
            want.sort_unstable();
            assert_eq!(ids(&db, name), want, "seed {seed} final {name}");
        }
        db.close().unwrap();
        assert_eq!(
            wal_len(dir.path()),
            empty_wal_len(),
            "seed {seed}: close sheds the log"
        );
    }
}
