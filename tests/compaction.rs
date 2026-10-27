//! Chaos harness for compaction.
//!
//! `chaos.rs` checks the engine against a model keyed by ORDINAL, which is
//! exactly what compaction renumbers, so it runs with compaction off. This
//! harness keys its model by PRIMARY KEY instead and drives the engine through
//! the SQL front door: inserts, deletes, copy-on-write updates, duplicate keys,
//! compactions, checkpoints, clean reopens and crashes (drop without close),
//! all interleaved. After every step the whole collection must read back as
//! the model, and the ordinal high-water mark must be exactly what the model
//! says was allocated since the last compaction — which is how renumbering is
//! checked to be dense, and capacity to be really reclaimed.
//!
//! A second, never-written collection keeps the global truncation point at
//! zero, so every crash replays the full log — records from before each flip
//! included. Those name old ordinals and must not land on the new generation.
//!
//! Deterministic: a fixed-seed xorshift RNG. Set `COMPACTION_ITERS` to run
//! more than the default.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use flats::vm::exec::RegValue;
use flats::{CompactionOptions, Db, DbOptions};

/// xorshift64* — same generator as `chaos.rs`.
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
    /// Exactly representable, so vectors compare bit-for-bit.
    fn coord(&mut self) -> f32 {
        (self.below(16) as f32 - 8.0) / 4.0
    }
}

const CAPACITY: u64 = 48;

/// Ground truth: key → (n, vector), plus the ordinals the current generation
/// has handed out.
struct Model {
    rows: BTreeMap<i64, (i64, Vec<f32>)>,
    allocated: u64,
    next_key: i64,
}

fn opts() -> DbOptions {
    // Every compaction and checkpoint is driven from the op stream.
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

fn rows(db: &Db) -> BTreeMap<i64, (i64, Vec<f32>)> {
    db.execute("SELECT id, n, vector FROM docs;")
        .expect("select")
        .map(|row| match row.expect("row").0.as_slice() {
            [RegValue::Int(id), RegValue::Int(n), RegValue::Vector(v)] => (*id, (*n, v.to_vec())),
            other => panic!("unexpected row {other:?}"),
        })
        .collect()
}

fn vector_sql(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", parts.join(", "))
}

fn verify(db: &Db, model: &Model, step: u64, op: &str) {
    assert_eq!(rows(db), model.rows, "rows diverged at step {step} after {op}");
    assert_eq!(
        db.reader(0).expect("docs").len() as u64,
        model.allocated,
        "high-water diverged at step {step} after {op}"
    );
}

fn run(seed: u64, iters: u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rng = Rng::new(seed);
    let mut db = Some(open(dir.path()));
    {
        let db = db.as_ref().unwrap();
        db.execute(&format!(
            "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(2), n INT) \
             WITH (capacity = {CAPACITY});"
        ))
        .expect("create docs");
        db.execute("CREATE COLLECTION idle (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 4);")
            .expect("create idle");
    }
    let mut model = Model {
        rows: BTreeMap::new(),
        allocated: 0,
        next_key: 0,
    };

    for step in 0..iters {
        let roll = rng.below(100);
        let d = db.as_ref().unwrap();
        let op: &str;
        if roll < 40 {
            op = "insert";
            let key = model.next_key;
            model.next_key += 1;
            let v = vec![rng.coord(), rng.coord()];
            let n = rng.below(1000) as i64;
            let res = d.execute(&format!(
                "INSERT INTO docs (id, vector, n) VALUES ({key}, {}, {n});",
                vector_sql(&v)
            ));
            if model.allocated < CAPACITY {
                res.expect("insert below capacity");
                model.rows.insert(key, (n, v));
                model.allocated += 1;
            } else {
                assert!(res.is_err(), "insert past capacity succeeded at step {step}");
            }
        } else if roll < 43 {
            op = "duplicate insert";
            if let Some(&key) = pick(&mut rng, &model) {
                let res = d.execute(&format!(
                    "INSERT INTO docs (id, vector, n) VALUES ({key}, [0.0, 0.0], 1);"
                ));
                assert!(res.is_err(), "duplicate key {key} accepted at step {step}");
            }
        } else if roll < 63 {
            op = "delete";
            if let Some(&key) = pick(&mut rng, &model) {
                d.execute(&format!("DELETE FROM docs WHERE id = {key};"))
                    .expect("delete");
                model.rows.remove(&key);
            }
        } else if roll < 78 {
            op = "update";
            if let Some(&key) = pick(&mut rng, &model) {
                let n = rng.below(1000) as i64;
                let res = d.execute(&format!("UPDATE docs SET n = {n} WHERE id = {key};"));
                if model.allocated < CAPACITY {
                    res.expect("update below capacity");
                    model.rows.get_mut(&key).unwrap().0 = n;
                    model.allocated += 1;
                } else {
                    assert!(res.is_err(), "update past capacity succeeded at step {step}");
                }
            }
        } else if roll < 86 {
            op = "compact";
            let did = d.compact_now(0).expect("compact");
            let live = model.rows.len() as u64;
            assert_eq!(did, model.allocated > live, "compact verdict at step {step}");
            model.allocated = live;
        } else if roll < 91 {
            op = "checkpoint";
            d.checkpoint().expect("checkpoint");
        } else if roll < 95 {
            op = "reopen";
            db.take().unwrap().close().expect("close");
            db = Some(open(dir.path()));
        } else {
            op = "crash";
            drop(db.take());
            db = Some(open(dir.path()));
        }
        verify(db.as_ref().unwrap(), &model, step, op);
    }
    db.take().unwrap().close().expect("final close");
}

fn pick<'a>(rng: &mut Rng, model: &'a Model) -> Option<&'a i64> {
    if model.rows.is_empty() {
        return None;
    }
    let at = rng.below(model.rows.len() as u64) as usize;
    model.rows.keys().nth(at)
}

#[test]
fn compaction_chaos() {
    let iters: u64 = std::env::var("COMPACTION_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500);
    run(0x9E37_79B9_7F4A_7C15, iters);
    run(0xD1B5_4A32_D192_ED03, iters / 2);
}
