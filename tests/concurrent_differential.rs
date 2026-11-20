//! Differential testing under concurrency: many writer and reader threads
//! at once, with the background checkpointer and compactor at full tilt.
//!
//! * Each WRITER owns the ids `≡ w (mod WRITERS)`, so its own model is the
//!   ground truth for its rows whatever the interleaving. At the end — and
//!   after a crash — the table must be exactly the union of the writers'
//!   models.
//! * Every row checks itself: its vector is `[n, id]` and its `n` column is
//!   `n`. A READER that ever sees a row whose vector and values disagree has
//!   seen two versions of one row mixed (a tuple from one generation, a vector
//!   from another, say — exactly what a compaction flip racing a read could
//!   produce). Readers also check that no id appears twice in one answer,
//!   that SEARCH never returns more than k rows, and that its scores are in
//!   order and match the vector it returned.
//!
//! `FLATS_CONC_ROUNDS` sets how many rounds run (default 3).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use flats::{CompactionOptions, Db, DbOptions, Param, RegValue};

const WRITERS: i64 = 4;
const READERS: usize = 3;
const OPS_PER_WRITER: usize = 400;

fn busy() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_millis(3),
        compaction: CompactionOptions {
            enabled: true,
            waste_ratio: 0.0,
            waste_min_dead: 1,
            headroom_ratio: 0.0,
            headroom_min_dead_ratio: 0.0,
        },
    }
}

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

/// One writer: random inserts, updates (value, or vector and value together)
/// and deletes on its own ids. Returns its model: id → n.
fn writer(db: &Db, w: i64, seed: u64) -> BTreeMap<i64, i64> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut model = BTreeMap::new();
    for _ in 0..OPS_PER_WRITER {
        let id = w + WRITERS * rng.below(40) as i64;
        let n = rng.below(1000) as i64;
        let exists = model.contains_key(&id);
        match rng.below(10) {
            0..=3 if !exists => {
                db.execute_with(
                    "INSERT INTO t (id, vector, n, owner) VALUES (?, ?, ?, ?);",
                    &[
                        Param::Int(id),
                        Param::Vector(vec![n as f32, id as f32]),
                        Param::Int(n),
                        Param::Int(w),
                    ],
                )
                .unwrap_or_else(|e| panic!("writer {w}: insert {id}: {e}"));
                model.insert(id, n);
            }
            4..=6 if exists => {
                // The vector and `n` change together, in one statement.
                db.execute_with(
                    "UPDATE t SET vector = ?, n = ? WHERE id = ?;",
                    &[
                        Param::Vector(vec![n as f32, id as f32]),
                        Param::Int(n),
                        Param::Int(id),
                    ],
                )
                .unwrap_or_else(|e| panic!("writer {w}: update {id}: {e}"));
                model.insert(id, n);
            }
            7..=8 if exists => {
                db.execute_with("DELETE FROM t WHERE id = ?;", &[Param::Int(id)])
                    .unwrap_or_else(|e| panic!("writer {w}: delete {id}: {e}"));
                model.remove(&id);
            }
            _ => {
                // A multi-row delete of this writer's rows with small n.
                let cut = rng.below(100) as i64;
                db.execute_with(
                    "DELETE FROM t WHERE owner = ? AND n < ?;",
                    &[Param::Int(w), Param::Int(cut)],
                )
                .unwrap_or_else(|e| panic!("writer {w}: delete n < {cut}: {e}"));
                model.retain(|_, v| *v >= cut);
            }
        }
    }
    model
}

/// One row as read: (id, n, vector).
fn row(cells: &[RegValue]) -> (i64, i64, Vec<f32>) {
    match cells {
        [RegValue::Int(id), RegValue::Int(n), RegValue::Vector(v)] => (*id, *n, v.to_vec()),
        other => panic!("unexpected row {other:?}"),
    }
}

fn consistent(what: &str, (id, n, v): &(i64, i64, Vec<f32>)) {
    assert_eq!(
        v.as_slice(),
        [*n as f32, *id as f32],
        "{what}: row {id} has n = {n} but vector {v:?}: two versions of one row were mixed"
    );
}

/// One reader: SELECTs and SEARCHes until told to stop, checking every
/// answer. Returns how many answers it checked.
fn reader(db: &Db, seed: u64, stop: &AtomicBool) -> u64 {
    let mut rng = Rng(seed | 1);
    let mut checked = 0;
    while !stop.load(Ordering::Relaxed) {
        if rng.below(2) == 0 {
            let owner = rng.below(WRITERS as u64) as i64;
            let rows: Vec<_> = db
                .execute_with(
                    "SELECT id, n, vector FROM t WHERE owner = ?;",
                    &[Param::Int(owner)],
                )
                .expect("select")
                .map(|r| row(&r.expect("row").0))
                .collect();
            let mut seen = std::collections::BTreeSet::new();
            for r in &rows {
                consistent("SELECT", r);
                assert_eq!(r.0 % WRITERS, owner, "row {} under the wrong owner", r.0);
                assert!(seen.insert(r.0), "SELECT returned row {} twice", r.0);
            }
        } else {
            let k = 1 + rng.below(20);
            let q = [rng.below(9) as f32 - 4.0, rng.below(9) as f32 - 4.0];
            let rows: Vec<(i64, i64, Vec<f32>, f64)> = db
                .execute_with(
                    "SEARCH TOP ? NEAREST TO ? FROM t RETURNING id, n, vector, score;",
                    &[Param::Int(k as i64), Param::Vector(q.to_vec())],
                )
                .expect("search")
                .map(|r| match r.expect("row").0.as_slice() {
                    [
                        RegValue::Int(id),
                        RegValue::Int(n),
                        RegValue::Vector(v),
                        RegValue::Real(s),
                    ] => (*id, *n, v.to_vec(), *s),
                    other => panic!("unexpected row {other:?}"),
                })
                .collect();
            assert!(
                rows.len() as u64 <= k,
                "SEARCH TOP {k} returned {}",
                rows.len()
            );
            let mut seen = std::collections::BTreeSet::new();
            for (id, n, v, s) in &rows {
                consistent("SEARCH", &(*id, *n, v.clone()));
                assert!(seen.insert(*id), "SEARCH returned row {id} twice");
                let want = f64::from(q[0]) * f64::from(v[0]) + f64::from(q[1]) * f64::from(v[1]);
                assert!(
                    (s - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "row {id} scored {s} but its vector {v:?} scores {want}"
                );
            }
            for pair in rows.windows(2) {
                assert!(pair[0].3 >= pair[1].3 - 1e-3, "SEARCH out of order");
            }
        }
        checked += 1;
    }
    checked
}

fn table(db: &Db) -> BTreeMap<i64, i64> {
    db.execute("SELECT id, n, vector FROM t;")
        .expect("select")
        .map(|r| {
            let r = row(&r.expect("row").0);
            consistent("final", &r);
            (r.0, r.1)
        })
        .collect()
}

fn round(seed: u64) {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Db::open(dir.path(), &[], busy()).unwrap());
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), n INT, owner INT) WITH (capacity = 100000);")
        .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let checked = Arc::new(AtomicU64::new(0));

    let readers: Vec<_> = (0..READERS)
        .map(|r| {
            let (db, stop, checked) = (db.clone(), stop.clone(), checked.clone());
            std::thread::spawn(move || {
                checked.fetch_add(reader(&db, seed * 31 + r as u64, &stop), Ordering::Relaxed);
            })
        })
        .collect();
    let writers: Vec<_> = (0..WRITERS)
        .map(|w| {
            let db = db.clone();
            std::thread::spawn(move || writer(&db, w, seed * 7 + w as u64))
        })
        .collect();
    let mut want = BTreeMap::new();
    for w in writers {
        want.extend(w.join().expect("a writer panicked"));
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().expect("a reader panicked");
    }
    assert!(
        checked.load(Ordering::Relaxed) > 0,
        "the readers checked nothing"
    );
    if std::env::var_os("FLATS_SIM_STATS").is_some() {
        eprintln!(
            "seed {seed}: {} reads checked; {} rows left",
            checked.load(Ordering::Relaxed),
            want.len()
        );
    }

    assert_eq!(
        table(&db),
        want,
        "seed {seed}: the table is not the union of the writers' rows"
    );
    // And after a crash.
    drop(Arc::try_unwrap(db).ok().expect("all threads joined"));
    let db = Db::open(dir.path(), &[], busy()).unwrap();
    assert_eq!(table(&db), want, "seed {seed}: after a crash");
    db.close().unwrap();
}

#[test]
fn concurrent_writers_and_readers_match_the_model() {
    let base = std::env::var("FLATS_SIM_SEED_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0u64);
    let rounds = std::env::var("FLATS_CONC_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3u64);
    for seed in base..base + rounds {
        round(seed);
    }
}
