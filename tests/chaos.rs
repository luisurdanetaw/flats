//! Chaos / property harness.
//!
//! Drives a `Db` through a long random sequence of inserts, deletes,
//! checkpoints, and full reopens, checking after every step (and especially
//! after every reopen) that the engine agrees with an in-memory reference model.
//!
//! Why reopen is the interesting bit: every `insert`/`delete` blocks until the
//! WAL has fsynced it, so every acked op is durable. A reopen therefore MUST
//! recover to exactly the model — no losses, no duplicates, no phantom (zero
//! filled) ordinals, and tombstones still hidden. Interleaving random
//! checkpoints (which advance the index watermark and truncate the WAL) means
//! reopens land in every reconciliation state: pure WAL replay, pure index, and
//! everything between.
//!
//! Deterministic: a fixed-seed xorshift RNG, so any failure reproduces. Set
//! `CHAOS_ITERS` to run more than the default 10_000 iterations.

use std::collections::{BTreeMap, BTreeSet};

use flats::{CollectionConfig, Db, DbOptions};

/// xorshift64* — tiny, deterministic, good enough to drive the op stream.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// A small, exactly-representable f32 so model and engine scores compare
    /// bit-for-bit (integers in [-8, 7] / 4 are exact in f32).
    fn coord(&mut self) -> f32 {
        (self.below(16) as f32 - 8.0) / 4.0
    }
}

/// Reference model: the ground truth the engine must match.
struct Model {
    /// Live (non-tombstoned) ordinal -> vector.
    live: BTreeMap<u64, Vec<f32>>,
    /// Next ordinal the allocator will hand out == the high-water mark.
    next_ordinal: u64,
    capacity: u64,
    /// Rolling tail of recent ops for failure diagnostics.
    history: std::collections::VecDeque<String>,
}

impl Model {
    fn new(capacity: u64) -> Self {
        Model {
            live: BTreeMap::new(),
            next_ordinal: 0,
            capacity,
            history: std::collections::VecDeque::new(),
        }
    }
    fn log(&mut self, s: String) {
        self.history.push_back(s);
        if self.history.len() > 25 {
            self.history.pop_front();
        }
    }
    fn at_capacity(&self) -> bool {
        self.next_ordinal >= self.capacity
    }
}

const DIM: usize = 8;
const CAPACITY: u64 = 4096;

fn random_vector(rng: &mut Rng) -> Vec<f32> {
    (0..DIM).map(|_| rng.coord()).collect()
}

fn cfgs() -> Vec<CollectionConfig> {
    vec![CollectionConfig {
        id: 0,
        dim: std::num::NonZeroUsize::new(DIM).unwrap(),
        capacity: CAPACITY as usize,
    }]
}

fn opts() -> DbOptions {
    // Drive checkpoints explicitly from the op stream; no background timer.
    DbOptions {
        checkpoint_interval: std::time::Duration::from_secs(3600),
    }
}

/// The core invariant check: searching with k >= high-water returns *exactly*
/// the model's live set, with scores that match the model's vectors. This
/// catches losses, duplicates, phantom zero-slots, and leaked tombstones.
fn verify(db: &Db, model: &Model, query: &[f32]) {
    let k = (model.next_ordinal.max(1)) as usize;
    let hits = db.search(0, query, k).expect("search");

    let got: BTreeSet<u64> = hits.iter().map(|h| h.id.0 as u64).collect();
    let want: BTreeSet<u64> = model.live.keys().copied().collect();
    if got != want {
        let extra: Vec<u64> = got.difference(&want).copied().collect();
        let missing: Vec<u64> = want.difference(&got).copied().collect();
        eprintln!("HISTORY (tail): {:?}", model.history);
        panic!(
            "search id set diverged: high_water={} k={} hits={} extra={:?} missing={:?}",
            model.next_ordinal, k, hits.len(), extra, missing
        );
    }

    for h in &hits {
        let v = &model.live[&(h.id.0 as u64)];
        assert_eq!(
            h.score,
            flats::dot(query, v),
            "score mismatch for ordinal {}",
            h.id.0
        );
    }
}

fn run(seed: u64, iters: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let cfgs = cfgs();
    let mut rng = Rng::new(seed);

    let mut model = Model::new(CAPACITY);
    let mut db: Option<Db> = Some(Db::open(&path, &cfgs, opts()).unwrap());
    let query = random_vector(&mut rng);

    for _ in 0..iters {
        match rng.below(100) {
            // ~50% insert
            0..=49 => {
                let v = random_vector(&mut rng);
                let res = db.as_ref().unwrap().insert(0, &v);
                if model.at_capacity() {
                    assert!(
                        matches!(res, Err(flats::Error::CapacityExceeded { .. })),
                        "expected CapacityExceeded at high-water {}",
                        model.next_ordinal
                    );
                } else {
                    let ord = res.expect("insert below capacity").0 as u64;
                    assert_eq!(ord, model.next_ordinal, "engine/model ordinal drift");
                    model.live.insert(ord, v);
                    model.next_ordinal += 1;
                    model.log(format!("insert ord={ord}"));
                }
            }
            // ~20% delete a random live ordinal
            50..=69 => {
                let n = model.live.len() as u64;
                if n > 0 {
                    let pick = rng.below(n) as usize;
                    let ord = *model.live.keys().nth(pick).unwrap();
                    db.as_ref().unwrap().delete(0, ord).expect("delete");
                    model.live.remove(&ord);
                    model.log(format!("delete ord={ord}"));
                }
            }
            // ~15% verify against the model
            70..=84 => verify(db.as_ref().unwrap(), &model, &query),
            // ~10% checkpoint (advance index watermark + truncate WAL)
            85..=94 => {
                db.as_ref().unwrap().checkpoint().expect("checkpoint");
                model.log("checkpoint".into());
            }
            // ~5% crash & recover, then verify the reconciled state
            _ => {
                drop(db.take()); // fully drop old instance: joins WAL thread, frees files
                db = Some(Db::open(&path, &cfgs, opts()).unwrap());
                model.log("REOPEN".into());
                verify(db.as_ref().unwrap(), &model, &query);
            }
        }
    }

    // One last reopen to prove the durable state matches end to end.
    drop(db.take());
    let reopened = Db::open(&path, &cfgs, opts()).unwrap();
    verify(&reopened, &model, &query);
    reopened.close().unwrap();
}

#[test]
fn chaos_reconciliation() {
    let iters: u64 = std::env::var("CHAOS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    // A couple of independent seeds for breadth while keeping each run reproducible.
    run(0x9E37_79B9_7F4A_7C15, iters);
    run(0xD1B5_4A32_D192_ED03, iters / 2);
}
