//! Chaos / property harness.
//!
//! Drives a `Db` through a long random sequence of inserts (with metadata),
//! deletes, checkpoints, and full reopens, checking after every step (and
//! especially after every reopen) that the engine agrees with an in-memory
//! reference model — across ALL THREE subsystems: flat vector index,
//! metadata index, and tuple store.
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

use flats::index::index::Ordinal;
use flats::metadata::tuples::RowGet;
use flats::{ColumnSpec, ColumnType, CollectionConfig, Db, DbOptions, RangeOp, Row, Schema, Value};

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

const TEXTS: [&str; 4] = ["red", "green", "blue", "teal"];

/// One row of ground truth: the vector plus the metadata values.
#[derive(Clone)]
struct ModelRow {
    vector: Vec<f32>,
    a: i64,
    b: f64,
    c: &'static str,
}

impl ModelRow {
    fn row(&self) -> Row {
        vec![
            (0, Value::Int(self.a)),
            (1, Value::Float(self.b)),
            (2, Value::Text(self.c.into())),
        ]
    }
}

/// Reference model: the ground truth the engine must match.
struct Model {
    /// Live (non-tombstoned) ordinal -> row.
    live: BTreeMap<u64, ModelRow>,
    /// Ordinals that were inserted then deleted (tuple store must say Deleted).
    deleted: BTreeSet<u64>,
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
            deleted: BTreeSet::new(),
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

fn random_row(rng: &mut Rng, vector: Vec<f32>) -> ModelRow {
    ModelRow {
        vector,
        a: rng.below(8) as i64,
        b: rng.coord() as f64, // exactly representable; compares bit-for-bit
        c: TEXTS[rng.below(TEXTS.len() as u64) as usize],
    }
}

fn schema() -> Schema {
    Schema::from_columns(vec![
        ColumnSpec::Vector {
            name: "vector".into(),
            dim: std::num::NonZeroUsize::new(DIM).unwrap(),
        },
        ColumnSpec::Scalar {
            name: "a".into(),
            ty: ColumnType::Int,
        },
        ColumnSpec::Scalar {
            name: "b".into(),
            ty: ColumnType::Float,
        },
        ColumnSpec::Scalar {
            name: "c".into(),
            ty: ColumnType::Text,
        },
    ])
    .unwrap()
}

fn cfgs() -> Vec<CollectionConfig> {
    vec![CollectionConfig {
        id: 0,
        name: "chaos".into(),
        capacity: CAPACITY as usize,
        schema: schema(),
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
        let v = &model.live[&(h.id.0 as u64)].vector;
        assert_eq!(
            h.score,
            flats::dot(query, v),
            "score mismatch for ordinal {}",
            h.id.0
        );
    }
}

/// The metadata half of the invariant: the metadata index's bitmaps and the
/// tuple store's values must match the model exactly — including tombstone
/// masking and Deleted markers.
fn verify_metadata(db: &Db, model: &Model, rng: &mut Rng) {
    let meta = db.metadata_reader(0).expect("metadata reader");
    let tuples = db.tuple_reader(0).expect("tuple reader");

    assert_eq!(meta.live_count(), model.live.len() as u64, "live_count");
    let live: BTreeSet<u64> = meta.live().iter().map(u64::from).collect();
    let want: BTreeSet<u64> = model.live.keys().copied().collect();
    assert_eq!(live, want, "live bitmap diverged");

    // lookup_eq on a random INT value and a random TEXT value.
    let a = rng.below(8) as i64;
    let got: BTreeSet<u64> = meta
        .lookup_eq(0, &Value::Int(a))
        .expect("lookup_eq int")
        .iter()
        .map(u64::from)
        .collect();
    let want: BTreeSet<u64> = model
        .live
        .iter()
        .filter(|(_, r)| r.a == a)
        .map(|(&o, _)| o)
        .collect();
    assert_eq!(got, want, "lookup_eq a={a}");

    let c = TEXTS[rng.below(TEXTS.len() as u64) as usize];
    let got: BTreeSet<u64> = meta
        .lookup_eq(2, &Value::Text(c.into()))
        .expect("lookup_eq text")
        .iter()
        .map(u64::from)
        .collect();
    let want: BTreeSet<u64> = model
        .live
        .iter()
        .filter(|(_, r)| r.c == c)
        .map(|(&o, _)| o)
        .collect();
    assert_eq!(got, want, "lookup_eq c={c}");

    // lookup_range with a random op on the INT column.
    let bound = rng.below(9) as i64 - 1; // sometimes outside the value range
    let op = [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge][rng.below(4) as usize];
    let got: BTreeSet<u64> = meta
        .lookup_range(0, op, &Value::Int(bound))
        .expect("lookup_range")
        .iter()
        .map(u64::from)
        .collect();
    let keep = |a: i64| match op {
        RangeOp::Lt => a < bound,
        RangeOp::Le => a <= bound,
        RangeOp::Gt => a > bound,
        RangeOp::Ge => a >= bound,
    };
    let want: BTreeSet<u64> = model
        .live
        .iter()
        .filter(|(_, r)| keep(r.a))
        .map(|(&o, _)| o)
        .collect();
    assert_eq!(got, want, "lookup_range a {op:?} {bound}");

    // Tuple store: a random live ordinal round-trips its full row…
    if !model.live.is_empty() {
        let pick = rng.below(model.live.len() as u64) as usize;
        let (&ord, row) = model.live.iter().nth(pick).unwrap();
        assert_eq!(
            tuples.get(Ordinal(ord as u32), &[0, 1, 2]).expect("get live"),
            RowGet::Live(vec![
                Value::Int(row.a),
                Value::Float(row.b),
                Value::Text(row.c.into()),
            ]),
            "tuple values for ordinal {ord}"
        );
    }
    // …and a random deleted ordinal reports the deleted-marker.
    if !model.deleted.is_empty() {
        let pick = rng.below(model.deleted.len() as u64) as usize;
        let &ord = model.deleted.iter().nth(pick).unwrap();
        assert_eq!(
            tuples.get(Ordinal(ord as u32), &[0]).expect("get deleted"),
            RowGet::Deleted,
            "deleted marker for ordinal {ord}"
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
            // ~50% insert (vector + metadata row, one durable record)
            0..=49 => {
                let vector = random_vector(&mut rng);
                let row = random_row(&mut rng, vector);
                let res = db.as_ref().unwrap().insert(0, &row.vector, row.row());
                if model.at_capacity() {
                    assert!(
                        matches!(res, Err(flats::Error::CapacityExceeded { .. })),
                        "expected CapacityExceeded at high-water {}",
                        model.next_ordinal
                    );
                } else {
                    let ord = res.expect("insert below capacity").0 as u64;
                    assert_eq!(ord, model.next_ordinal, "engine/model ordinal drift");
                    model.live.insert(ord, row);
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
                    model.deleted.insert(ord);
                    model.log(format!("delete ord={ord}"));
                }
            }
            // ~8% verify vectors against the model
            70..=77 => verify(db.as_ref().unwrap(), &model, &query),
            // ~7% verify metadata + tuples against the model
            78..=84 => verify_metadata(db.as_ref().unwrap(), &model, &mut rng),
            // ~10% checkpoint (advance index watermark + truncate WAL)
            85..=94 => {
                db.as_ref().unwrap().checkpoint().expect("checkpoint");
                model.log("checkpoint".into());
            }
            // ~5% crash & recover, then verify the reconciled state everywhere
            _ => {
                drop(db.take()); // fully drop old instance: joins WAL thread, frees files
                db = Some(Db::open(&path, &cfgs, opts()).unwrap());
                model.log("REOPEN".into());
                verify(db.as_ref().unwrap(), &model, &query);
                verify_metadata(db.as_ref().unwrap(), &model, &mut rng);
            }
        }
    }

    // One last reopen to prove the durable state matches end to end.
    drop(db.take());
    let reopened = Db::open(&path, &cfgs, opts()).unwrap();
    verify(&reopened, &model, &query);
    verify_metadata(&reopened, &model, &mut rng);
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
