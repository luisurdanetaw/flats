//! Crash and fail at every step of every durability path.
//!
//! Each fault point (`src/failpoint.rs`) makes the engine fail with an I/O
//! error at one precise step: before a WAL write, mid-write, before and after
//! the fsync, between the stores of a checkpoint, before each snapshot
//! rename, between a compaction's two renames. Two ways to use one:
//!
//! * **crash there**: fail, then drop the database without closing and
//!   reopen. The files are exactly what a crash at that step leaves. Every
//!   acknowledged write must survive; the op in flight may or may not have.
//! * **fail there and carry on**: the op reports the error, and an error
//!   must mean it did not happen — not now, and not after a restart either.
//!
//! Run with `cargo test --release --features failpoints --test failpoints`.
//! `FLATS_FP_SEEDS` sets the sweep's size.

mod sim;

use std::sync::Mutex;

use flats::failpoint;
use flats::{Db, Param, RegValue};
use sim::{Model, Op, Rng, Sim, env_u64, gen_workload, opts, verify_db};

/// Fault points are process-wide: one test at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    failpoint::disarm();
    g
}

fn ids(db: &Db, coll: &str) -> Vec<i64> {
    let mut v: Vec<i64> = db
        .execute(&format!("SELECT id FROM {coll};"))
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [RegValue::Int(i)] => *i,
            other => panic!("{other:?}"),
        })
        .collect();
    v.sort_unstable();
    v
}

fn insert(db: &Db, id: i64) -> Result<(), flats::QueryError> {
    db.execute_with(
        "INSERT INTO t (id, vector, s) VALUES (?, ?, ?);",
        &[
            Param::Int(id),
            Param::Vector(vec![id as f32, 1.0]),
            Param::Text(format!("r{id}")),
        ],
    )
    .map(|_| ())
}

fn fresh(dir: &std::path::Path) -> Db {
    let db = Db::open(dir, &[], opts()).unwrap();
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), s TEXT) WITH (capacity = 1000);")
        .unwrap();
    db
}

// ---------------------------------------------------------------------------
// Regressions, each found by the crash tests
// ---------------------------------------------------------------------------

/// Found by `tests/crash_kill.rs`. A checkpoint writes metadata.snap, then
/// tuples.snap. Crash between them and the metadata index is one statement
/// ahead of the tuple store; recovery replays that statement to catch the
/// tuple store up — and the apply-time primary-key check, consulting the
/// metadata index, saw the key held (by the statement's own row) and refused
/// it as a duplicate. The tuple store never got the row: the database opened
/// with rows that had no values.
#[test]
fn a_crash_between_the_metadata_snapshots_loses_nothing() {
    let _g = serial();
    for statement in [
        "INSERT INTO t (id, vector, s) VALUES (10, [1, 0], 'a');",
        "INSERT INTO t (id, vector, s) VALUES (10, [1, 0], 'a'), (11, [0, 1], 'b');",
        "UPDATE t SET s = 'changed' WHERE id = 1;",
        "UPDATE t SET vector = [5, 5] WHERE id = 1;",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let before;
        {
            let db = fresh(dir.path());
            for id in 0..3 {
                insert(&db, id).unwrap();
            }
            db.checkpoint().unwrap();
            db.execute(statement).unwrap();
            before = db
                .execute("SELECT id, s, vector FROM t;")
                .unwrap()
                .map(|r| format!("{:?}", r.unwrap().0))
                .collect::<std::collections::BTreeSet<_>>();
            failpoint::arm("checkpoint.before_tuples", 0);
            assert!(db.checkpoint().is_err(), "the failpoint fired");
            drop(db); // crash: metadata.snap is ahead of tuples.snap
        }
        let db = Db::open(dir.path(), &[], opts()).unwrap();
        let after = db
            .execute("SELECT id, s, vector FROM t;")
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .map(|r| format!("{:?}", r.unwrap().0))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(after, before, "{statement}");
        db.close().unwrap();
    }
}

/// An fsync that fails leaves the batch's bytes in the file. They used to
/// stay there: the caller was told the write failed, and the next restart
/// replayed it anyway.
#[test]
fn a_write_whose_fsync_failed_never_comes_back() {
    let _g = serial();
    for site in ["wal.before_fsync", "wal.after_fsync", "wal.before_write"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = fresh(dir.path());
            insert(&db, 1).unwrap();
            failpoint::arm(site, 0);
            assert!(
                insert(&db, 2).is_err(),
                "{site}: the write reports the failure"
            );
            insert(&db, 3).unwrap_or_else(|e| panic!("{site}: the log carries on: {e}"));
            assert_eq!(ids(&db, "t"), vec![1, 3], "{site}");
            drop(db);
        }
        let db = Db::open(dir.path(), &[], opts()).unwrap();
        assert_eq!(ids(&db, "t"), vec![1, 3], "{site}: after a restart too");
        db.close().unwrap();
    }
}

/// A short write (a full disk, say) put half a frame in the MIDDLE of the
/// log once later batches appended after it. Recovery stops at the first bad
/// frame, so every acknowledged write after the failed one vanished on the
/// next restart.
#[test]
fn a_torn_write_does_not_hide_later_writes() {
    let _g = serial();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = fresh(dir.path());
        insert(&db, 1).unwrap();
        failpoint::arm("wal.torn_write", 0);
        assert!(insert(&db, 2).is_err());
        for id in 3..10 {
            insert(&db, id).unwrap();
        }
        drop(db);
    }
    let db = Db::open(dir.path(), &[], opts()).unwrap();
    assert_eq!(ids(&db, "t"), vec![1, 3, 4, 5, 6, 7, 8, 9]);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The sweep: random workloads, a random fault point armed at a random hit
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Fail there, then crash and reopen.
    Crash,
    /// Fail there and carry on; restart later.
    CarryOn,
}

fn model_without(before: &Model) -> Model {
    before.clone()
}

/// Run `ops`, re-arming a random fault point after each one fires. Returns
/// how many faults fired.
fn sweep_one(seed: u64, mode: Mode) -> u64 {
    let ops: Vec<Op> = gen_workload(seed, 220);
    let mut rng = Rng::new(seed ^ 0xFA17);
    // Quiet background: a fault must fire inside the op it's attributed to.
    let mut sim = Sim::new_with(seed, opts());
    let arm = |rng: &mut Rng| {
        let site = *rng.pick(failpoint::SITES);
        failpoint::arm(site, rng.below(6));
        site
    };
    let mut site = arm(&mut rng);
    let mut fired = 0;
    for (i, op) in ops.iter().enumerate() {
        let before = sim.model.clone();
        let fired_before = failpoint::fired();
        let result = sim.step(op);
        if failpoint::fired() == fired_before {
            if let Err(e) = result {
                panic!("seed {seed} {mode:?}: op {i} with {site} armed (not fired): {e}");
            }
            continue;
        }
        fired += 1;
        let ctx =
            format!("seed {seed} {mode:?}: {site} fired during op {i} ({op:?}) -> {result:?}");
        match mode {
            Mode::Crash => {
                sim.crash_and_reopen()
                    .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                // The op in flight may or may not have reached its commit
                // point; nothing else may differ.
                let after = sim.model.clone();
                if verify_db(sim.db(), &after).is_ok() {
                    sim.model = after;
                } else if let Err(e) = verify_db(sim.db(), &before) {
                    panic!(
                        "{ctx}: after the crash the database matches neither side of the op: {e}"
                    );
                } else {
                    sim.model = before;
                }
            }
            Mode::CarryOn => {
                // An op that reported failure did not happen. (An engine op
                // — checkpoint, compaction, reopen — changes no rows either
                // way.) A failed close or compaction may have taken the
                // handle down with it; reopen in that case.
                //
                // The one exception is a failure AFTER the commit point (the
                // record is in the log, applying it failed): the log is then
                // poisoned, the error says the record is durable, and the
                // next open applies it. Hold the engine to that message.
                let after = sim.model.clone();
                let committed = matches!(&result, Err(e) if e.contains("unapplied"));
                sim.model = if committed {
                    after
                } else {
                    model_without(&before)
                };
                if op.is_write() {
                    assert!(result.is_err(), "{ctx}: the failure was not reported");
                }
                if committed {
                    sim.crash_and_reopen()
                        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                }
                let live =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.verify_all()));
                if !matches!(live, Ok(Ok(()))) {
                    sim.crash_and_reopen()
                        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                }
                if let Err(e) = sim.verify_all() {
                    panic!("{ctx}: carrying on, the database disagrees with the model: {e}");
                }
                // And a restart agrees too.
                sim.crash_and_reopen()
                    .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                if let Err(e) = sim.verify_all() {
                    panic!(
                        "{ctx}: after a restart, a write that reported failure is back (or another is gone): {e}"
                    );
                }
            }
        }
        if fired < 4 {
            site = arm(&mut rng);
        }
    }
    failpoint::disarm();
    if let Err(e) = sim.verify_all() {
        panic!("seed {seed} {mode:?}: at the end: {e}");
    }
    fired
}

#[test]
fn every_fault_point_crashes_and_fails_safely() {
    let _g = serial();
    let base = env_u64("FLATS_SIM_SEED_BASE", 0);
    let n = env_u64("FLATS_FP_SEEDS", 24);
    let mut fired = 0;
    for seed in base..base + n {
        fired += sweep_one(seed, Mode::Crash);
        fired += sweep_one(seed, Mode::CarryOn);
    }
    assert!(
        fired > n,
        "the sweep must actually inject faults (fired {fired})"
    );
    // Every site is reachable by the workload — a point nothing reaches
    // tests nothing.
    let unreached: Vec<&str> = failpoint::SITES
        .iter()
        .copied()
        .filter(|s| failpoint::hits(s) == 0)
        .collect();
    assert!(unreached.is_empty(), "never reached: {unreached:?}");
}

/// Found by the sweep below, shrunk to three statements. An UPDATE copies a
/// row to a new ordinal under the same key. If a crash mid-checkpoint left the
/// metadata index (where uniqueness is judged) holding the copy while the
/// tuple store was behind, replaying the original INSERT found its key "held"
/// by the copy and refused it, and the tuple store never got the other row of
/// that INSERT. Now the metadata index is checkpointed last, so it is never
/// the one ahead; this crashes at every step between the stores.
#[test]
fn a_crash_mid_checkpoint_after_an_update_loses_nothing() {
    let _g = serial();
    for site in [
        "checkpoint.before_flat_header",
        "checkpoint.before_tuples",
        "checkpoint.before_metadata",
        "checkpoint.before_catalog",
    ] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[], opts()).unwrap();
            db.execute("CREATE COLLECTION c (b FLOAT, w TEXT PRIMARY KEY, emb VECTOR(2)) WITH (capacity = 100);")
                .unwrap();
            db.execute("INSERT INTO c (emb, b, w) VALUES ([2, -2], -4.0, ''), ([1, -2], 12, 'x');")
                .unwrap();
            db.execute("UPDATE c SET emb = [-1, 2], b = -1e300 WHERE w = '';")
                .unwrap();
            failpoint::arm(site, 0);
            assert!(db.close().is_err(), "{site} fired");
        }
        let db = Db::open(dir.path(), &[], opts()).unwrap();
        let rows: Vec<String> = db
            .execute("SELECT w, b, emb FROM c;")
            .unwrap()
            .map(|r| format!("{:?}", r.unwrap_or_else(|e| panic!("{site}: {e}")).0))
            .collect();
        let mut rows = rows;
        rows.sort();
        assert_eq!(
            rows,
            vec![
                r#"[Str(""), Real(-1e300), Vector([-1.0, 2.0])]"#.to_string(),
                r#"[Str("x"), Real(12.0), Vector([1.0, -2.0])]"#.to_string(),
            ],
            "{site}"
        );
        db.close().unwrap();
    }
}

/// Found untested by `scripts/coverage.sh`: a read that opens while a
/// compaction flips generations. Opening resolves several handles (the live
/// set, the metadata reader, the vectors) one at a time; a flip between two of
/// them would pair one generation's ordinals with the other's rows, and the
/// read must notice (the epoch check) and start again. Real timing almost
/// never hits that window, so pause a reader inside it, flip, and let go.
#[test]
fn a_read_that_opens_across_a_compaction_flip_starts_again() {
    let _g = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = std::sync::Arc::new(fresh(dir.path()));
    for id in 0..30 {
        insert(&db, id).unwrap();
    }
    // Dead rows, so the flip renumbers every live one.
    db.execute("DELETE FROM t WHERE id < 20;").unwrap();
    let coll = db.collections()[0].id;

    for query in [
        "SELECT id, s, vector FROM t WHERE id >= 0;",
        "SELECT id, s, vector FROM t;",
        "SEARCH TOP 5 NEAREST TO [1, 0] FROM t WHERE id >= 22 RETURNING id, s, vector;",
    ] {
        // Refill the dead space so the next round's flip renumbers again.
        for id in 100..110 {
            insert(&db, id).unwrap();
        }
        db.execute("DELETE FROM t WHERE id >= 100;").unwrap();

        let hits_before = failpoint::hits("vm.open_cursors.between_ops");
        failpoint::arm_pause("vm.open_cursors.between_ops", 0);
        let reader = {
            let db = db.clone();
            std::thread::spawn(move || -> Vec<(i64, String, Vec<f32>)> {
                db.execute(query)
                    .unwrap()
                    .map(|r| match r.unwrap().0.as_slice() {
                        [RegValue::Int(i), RegValue::Str(s), RegValue::Vector(v)] => {
                            (*i, s.clone(), v.to_vec())
                        }
                        other => panic!("{other:?}"),
                    })
                    .collect()
            })
        };
        let parked = failpoint::wait_parked(std::time::Duration::from_secs(10));
        if parked {
            assert!(db.compact_now(coll).unwrap(), "{query}: the flip happened");
        }
        failpoint::release();
        let rows = reader.join().expect("the reader");
        for (id, s, v) in &rows {
            assert_eq!(
                s,
                &format!("r{id}"),
                "{query}: row {id} got another row's value"
            );
            assert_eq!(
                v.as_slice(),
                [*id as f32, 1.0],
                "{query}: row {id} got another row's vector"
            );
        }
        let mut ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
        ids.sort_unstable();
        let want: Vec<i64> = if query.starts_with("SEARCH") {
            (25..30).collect()
        } else {
            (20..30).collect()
        };
        assert_eq!(ids, want, "{query}");
        if parked {
            // The read went round again after the flip.
            assert!(
                failpoint::hits("vm.open_cursors.between_ops") > hits_before + 1,
                "{query}: the read did not start again"
            );
        }
    }
}

/// Found by `tests/crash_kill.rs` (seed 500052): a kill during CREATE
/// COLLECTION, after the vector index file was sized but before its header
/// was written, left a zero-filled `vectors.idx`. Replaying the CREATE found
/// the file, tried to open it, and the database refused to open at all ("bad
/// magic number: 0"). A half-made collection must be absorbed by replay.
#[test]
fn a_crash_while_creating_a_collection_leaves_the_database_openable() {
    let _g = serial();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Db::open(dir.path(), &[], opts()).unwrap();
        failpoint::arm("index.create.before_header", 0);
        let r = db.execute(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), s TEXT) WITH (capacity = 1000);",
        );
        assert!(r.is_err(), "the failpoint fired");
        drop(db); // crash
    }
    failpoint::disarm();
    let db = Db::open(dir.path(), &[], opts()).expect("the database opens");
    // The CREATE was committed before it failed to materialize, so replay
    // finishes it: the collection exists, empty, and works.
    insert(&db, 1).expect("the collection works");
    assert_eq!(ids(&db, "t"), vec![1]);
    db.close().unwrap();
}

/// A collection's files are created BEFORE the catalog names it (so a listed
/// collection with no files can be recognized as lost). A crash between the
/// two must leave a database that opens: the orphaned files are passed over,
/// or reused when the collection is registered again.
#[test]
fn a_crash_between_creating_a_collection_and_registering_it_is_harmless() {
    use flats::{CollectionConfig, ColumnSpec, ColumnType, Schema};
    let _g = serial();
    let schema = Schema::from_columns(vec![
        ColumnSpec::Key {
            name: "id".into(),
            ty: ColumnType::Int,
        },
        ColumnSpec::Vector {
            name: "vector".into(),
            dim: std::num::NonZeroUsize::new(2).unwrap(),
        },
    ])
    .unwrap();
    let cfg = CollectionConfig {
        id: 0,
        name: "pre".into(),
        capacity: 10,
        schema,
    };
    for reregister in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        failpoint::arm("catalog.before_write", 0);
        assert!(
            Db::open(dir.path(), std::slice::from_ref(&cfg), opts()).is_err(),
            "the failpoint fired"
        );
        failpoint::disarm();
        assert!(
            dir.path().join("collection-0").exists(),
            "the files came first"
        );
        let with: &[CollectionConfig] = if reregister {
            std::slice::from_ref(&cfg)
        } else {
            &[]
        };
        let db = Db::open(dir.path(), with, opts()).expect("the database opens");
        assert_eq!(db.collections().len(), usize::from(reregister));
        if reregister {
            db.execute("INSERT INTO pre (id, vector) VALUES (1, [1, 0]);")
                .unwrap();
        }
        db.close().unwrap();
    }
}
