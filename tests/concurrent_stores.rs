//! Cross-store visibility: what a concurrent reader can observe BETWEEN the
//! metadata index and the tuple store.
//!
//! ============================================================================
//! WORLD VERDICT: **WORLD B — `Missing` on a live ordinal is IMPOSSIBLE.**
//!
//! It was not always. This file's original run observed the meta->tuple window
//! **~20-28k times in ~6 seconds** across two seeds, and that measurement is
//! why the fan-out order in `IndexApplier::apply` is what it is. Liveness is
//! now published LAST on the way in: `flat.write_at` and `tuple.write_row`
//! both complete before `meta.insert_row` sets the live bit, so an ordinal
//! cannot be enumerable before the stores hold it. `live_then_get_under_
//! concurrent_writes` asserts the window is gone; it reports 0 sightings where
//! it used to report tens of thousands.
//!
//! POLICY THE CURSOR MUST IMPLEMENT:
//!
//!   * `RowGet::Missing` for an ordinal enumerated from a liveness snapshot is
//!     a BUG — in the applier's ordering, not in the reader. It is no longer a
//!     transient to skip past. Treat it loudly.
//!   * `RowGet::Deleted` for an enumerated ordinal remains legal: the row can
//!     be deleted after the snapshot is taken, and today's delete destroys the
//!     values. Skip it.
//!   * A row whose VALUES are wrong is loud in every world, and is asserted
//!     throughout here.
//!
//! WHAT IS STILL OPEN: the DELETE side. `flat.delete` tombstones before
//! `tuple.delete_row` clears the row, so a snapshot taken before a delete can
//! see the two stores disagree (~50 sightings per 3s run, measured). That skew
//! is inherent to a destructive delete and closes only when retiring a row
//! stops destroying its values.
//! ============================================================================
//!
//! # Why this file exists
//!
//! The cursor abstraction's core loop is `for o in live() { tuples.get(o) }`.
//! The engine's apply fan-out (`src/engine/mod.rs`, `Apply for IndexApplier`)
//! writes flat → meta → tuple across THREE INDEPENDENT MUTEXES with no
//! cross-store atomicity, so between `meta.insert_row()` releasing its lock and
//! `tuple.write_row()` taking its, a concurrent reader can observe ordinal `N`
//! in `live()` while `tuples.get(N)` returns [`RowGet::Missing`].
//!
//! `tuples.rs` documents `Missing` as "replay hasn't caught up or there's a
//! consistency bug — the executor may want to treat it loudly". The cursor IS
//! that executor, so it has to pick: skip, retry, or shout. That was a guess
//! until this file; nothing else covers the path (`swmr.rs` is FlatIndex-only,
//! `chaos.rs` is single-threaded, and the `Missing` unit tests in `tuples.rs`
//! only read never-written ordinals).
//!
//! # What these tests can and cannot prove
//!
//! Like `swmr.rs`, this is black-box stress: it demonstrates the race exists
//! and that every sighting resolves. It cannot prove a bound on the window.
//! The retry budget below is deliberately enormous relative to the measured
//! resolve time so that a failure means "the row is genuinely lost", never
//! "the machine was busy".

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flats::index::index::Ordinal;
use flats::metadata::index as meta;
use flats::metadata::tuples::{self, RowGet};
use flats::{CollectionConfig, ColumnSpec, ColumnType, Db, DbOptions, Row, Schema, Value};
use static_assertions::assert_impl_all;

// The reader handles must be shareable for any of this to be possible — the
// same type-level posture `swmr.rs` pins for the flat index.
assert_impl_all!(meta::Reader: Clone, Send, Sync);
assert_impl_all!(tuples::Reader: Clone, Send, Sync);

const DIM: usize = 8;
const CAPACITY: usize = 200_000;
/// The scalar columns, in storage `ColumnId` order: a INT, b FLOAT, c TEXT.
const COLS: [u32; 3] = [0, 1, 2];
const TEXTS: [&str; 4] = ["red", "green", "blue", "teal"];
/// Rows pre-loaded before the readers start, so they never spin on an empty
/// bitmap waiting for the first insert.
const SEED_ROWS: u64 = 64;
/// How long the writer hammers the engine before a run winds down.
const WRITE_FOR: Duration = Duration::from_secs(3);
/// Cap on captured `Missing` observations — enough to characterize the race
/// without an unbounded log.
const MAX_OBSERVATIONS: usize = 8;
/// Which world this build implements.
///
/// **A** — liveness has three independent owners (the flat index's tombstone
/// bitset, the metadata index's `live` bitmap, the tuple store's
/// `Slot::Tombstone`), updated at three different moments during apply. A
/// reader can therefore observe an ordinal in `live()` before the tuple store
/// holds it.
///
/// **B** — liveness is unified behind one versioned snapshot published after
/// every store already holds the row, so the window is closed structurally.
///
/// The snapshot-isolation work flips this constant to [`World::B`]. It is the
/// single line that changes: the decider reads the verdict from here.
#[allow(dead_code)] // `B` is unconstructed until liveness unification lands.
enum World {
    A,
    B,
}

const EXPECTED_WORLD: World = World::B;

/// xorshift64* — same generator `chaos.rs` uses, so a failure reproduces from
/// the seed alone.
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
}

// ---------------------------------------------------------------------------
// Ground truth as a pure function of the ordinal
//
// There is no shared model to lock: every row's contents are derived from its
// ordinal, so a reader thread can verify any row it observes with zero
// coordination. A torn write, or a row filed under the wrong ordinal, breaks
// the relationship and trips the assert — the same trick `swmr.rs` plays with
// `pattern`/`expected_score`.
// ---------------------------------------------------------------------------

fn vector_for(o: u64) -> Vec<f32> {
    vec![((o % 7) + 1) as f32; DIM]
}

/// The row's values in `COLS` order. Every number is exactly representable, so
/// comparisons are bit-for-bit.
fn values_for(o: u64) -> Vec<Value> {
    vec![
        Value::Int((o % 8) as i64),
        Value::Float((o % 16) as f64 / 4.0 - 2.0),
        Value::Text(TEXTS[(o % 4) as usize].into()),
    ]
}

fn row_for(o: u64) -> Row {
    values_for(o)
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect()
}

/// Verify a row the reader observed. Holds in EVERY world: whatever the stores'
/// mutual visibility, a value they hand back must be the value that ordinal was
/// written with.
fn verify_values(o: u64, got: &[Value]) {
    assert_eq!(
        got,
        values_for(o).as_slice(),
        "tuple store returned the wrong values for ordinal {o} — torn or misfiled row"
    );
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
        name: "races".into(),
        capacity: CAPACITY,
        schema: schema(),
    }]
}

/// No background flusher: a checkpoint mid-run would add lock traffic that
/// muddies the tally without making the race any more likely.
fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
    }
}

fn reader_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(4)
}

// ---------------------------------------------------------------------------
// Shared harness
// ---------------------------------------------------------------------------

/// Pre-load `SEED_ROWS` rows and assert the allocator hands out dense ordinals
/// from 0 — the assumption the cursor's enumeration rests on.
fn seed_rows(db: &Db) {
    for o in 0..SEED_ROWS {
        let ord = db
            .insert(0, &vector_for(o), row_for(o))
            .expect("seed insert");
        assert_eq!(ord.0 as u64, o, "ordinals are dense from 0");
    }
}

/// Randomized insert/delete mix through the **`Db` API** (not the stores
/// directly) for `WRITE_FOR`. Runs on the caller's thread; keeps its live set
/// thread-local, so readers need no lock to verify anything.
///
/// Returns `(live ordinals, next ordinal)`.
fn drive_writer(db: &Db, seed: u64, acked: &AtomicU64) -> (Vec<u64>, u64) {
    let mut rng = Rng::new(seed);
    let mut live: Vec<u64> = (0..SEED_ROWS).collect();
    let mut next = SEED_ROWS;
    let deadline = Instant::now() + WRITE_FOR;

    while Instant::now() < deadline {
        // ~55/45 insert/delete keeps the live set small, which keeps `live()`
        // cheap to clone and the probe loop tight.
        if live.is_empty() || rng.below(100) < 55 {
            let ord = db
                .insert(0, &vector_for(next), row_for(next))
                .expect("insert");
            assert_eq!(
                ord.0 as u64, next,
                "engine ordinal drifted from dense order"
            );
            live.push(next);
            next += 1;
        } else {
            let idx = rng.below(live.len() as u64) as usize;
            let victim = live.swap_remove(idx);
            db.delete(0, victim).expect("delete");
        }
        acked.fetch_add(1, Ordering::Relaxed);
    }
    (live, next)
}

/// The quiescent cross-check, run after the writer stops and the readers join.
///
/// Every ack'd op was applied BEFORE its ack (`wal.rs`, `commit_batch`), the
/// writer is the only writer and has all its acks, and no reader is running. So
/// the stores are settled — and this is the one point where "nothing outside
/// `live()` is live" is even well-defined. During the run an insert can always
/// land between a snapshot and a get, which is exactly why the concurrent phase
/// counts transients instead of asserting on them.
fn assert_settled(meta: &meta::Reader, tuples: &tuples::Reader, live: &[u64], next: u64) {
    let want_live: BTreeSet<u64> = live.iter().copied().collect();
    let got_live: BTreeSet<u64> = meta.live().iter().map(u64::from).collect();
    assert_eq!(got_live, want_live, "live bitmap diverged from the writer");

    for o in 0..next {
        match tuples.get(Ordinal(o as u32), &COLS).expect("get") {
            RowGet::Live(values) => {
                verify_values(o, &values);
                assert!(
                    want_live.contains(&o),
                    "ordinal {o} is live but absent from live()"
                );
            }
            RowGet::Deleted => {
                assert!(
                    !want_live.contains(&o),
                    "ordinal {o} is in live() but tombstoned"
                );
            }
            RowGet::Missing => panic!("ordinal {o} still Missing after the run quiesced"),
        }
    }
}

// ---------------------------------------------------------------------------
// Observation tally
// ---------------------------------------------------------------------------

/// What the reader threads saw. The `missing_*` counters are the instrument the
/// decider asserts against; which of them, and in which direction, is dictated
/// by [`EXPECTED_WORLD`].
/// `Missing`-on-live sightings from one run, kept split by reader role.
struct Sightings {
    /// Sightings from the full-scan role — the cursor's exact future loop.
    scan: u64,
    /// Sightings from the frontier-probe role.
    probe: u64,
}

#[derive(Default)]
struct Tally {
    /// `live()` snapshots taken.
    snapshots: AtomicU64,
    /// `tuples.get` calls issued for an ordinal that was in the snapshot.
    gets: AtomicU64,
    /// …of those, how many returned a live row (values verified).
    live_hits: AtomicU64,
    /// …how many returned the deleted-marker. LEGAL: the row can be deleted
    /// between the snapshot and the get.
    deleted_on_live: AtomicU64,
    /// …how many returned `Missing` from the full-scan role.
    missing_scan: AtomicU64,
    /// …how many returned `Missing` from the frontier-probe role.
    missing_probe: AtomicU64,
    /// Context for the first `MAX_OBSERVATIONS` missing sightings.
    observations: Mutex<Vec<String>>,
}

impl Tally {
    fn record(&self, line: String) {
        let mut obs = self.observations.lock().unwrap_or_else(|e| e.into_inner());
        if obs.len() < MAX_OBSERVATIONS {
            obs.push(line);
        }
    }

    /// Print the tally and return the `Missing`-on-live counts, split by role.
    ///
    /// Split on purpose: the two roles trip the window at wildly different rates
    /// (see the header), so collapsing them into one number would hand the
    /// decider a statistic it cannot safely assert on.
    fn report(&self, label: &str) -> Sightings {
        let missing_scan = self.missing_scan.load(Ordering::Relaxed);
        let missing_probe = self.missing_probe.load(Ordering::Relaxed);
        eprintln!(
            "\n[{label}]\n  snapshots        {}\n  gets on live     {}\n  live rows        {}\n  \
             deleted-on-live  {}\n  MISSING (scan)   {}\n  MISSING (probe)  {}",
            self.snapshots.load(Ordering::Relaxed),
            self.gets.load(Ordering::Relaxed),
            self.live_hits.load(Ordering::Relaxed),
            self.deleted_on_live.load(Ordering::Relaxed),
            missing_scan,
            missing_probe,
        );
        for line in self
            .observations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            eprintln!("  {line}");
        }
        Sightings {
            scan: missing_scan,
            probe: missing_probe,
        }
    }
}

// ---------------------------------------------------------------------------
// 1. THE DECIDER
// ---------------------------------------------------------------------------

/// One writer thread through the `Db` API, N reader threads racing `live()`
/// against `tuples.get()`, then a quiescent full cross-check.
///
/// Two reader roles, because they probe different parts of the window:
///
///   * **SCANNER** — the cursor's exact future loop: snapshot `live()`, then
///     `get` EVERY ordinal in it, ascending.
///   * **PROBER** — same snapshot, but it immediately gets the snapshot's
///     MAXIMUM ordinal and nothing else. That ordinal is the one the apply
///     thread just published into the metadata index, so this aims straight at
///     the meta→tuple window. If the race is observable at all, this role sees
///     it — and it is the reason a clean run would have meant something.
///
/// Keeping BOTH roles matters: the prober answers "is the window real?", the
/// scanner answers "does the cursor's own access pattern reach it?" — and it
/// does, rarely (see the header).
///
/// Returns the `Missing`-on-live counts split by role. This function asserts
/// the invariants that hold in BOTH worlds — that the run did real work, and
/// that every value it did see was correct — and leaves the world verdict to
/// the caller, which asserts it against [`EXPECTED_WORLD`].
fn run_decider(seed: u64) -> Sightings {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &cfgs(), opts()).unwrap();

    // Reader handles are taken once and shared; they observe the applier's
    // writes through the same Arc'd inner state, so they stay live for the run.
    let meta = db.metadata_reader(0).expect("metadata reader");
    let tuples = db.tuple_reader(0).expect("tuple reader");
    seed_rows(&db);

    let stop = Arc::new(AtomicBool::new(false));
    let tally = Arc::new(Tally::default());
    // Acked writer ops, published purely as diagnostic context for a sighting.
    let acked = Arc::new(AtomicU64::new(SEED_ROWS));

    let mut handles = Vec::new();
    for role in 0..reader_threads() {
        let scanner = role == 0;
        let meta = meta.clone();
        let tuples = tuples.clone();
        let stop = stop.clone();
        let tally = tally.clone();
        let acked = acked.clone();
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let live = meta.live();
                tally.snapshots.fetch_add(1, Ordering::Relaxed);
                let snap_len = live.len();
                let snap_max = match live.max() {
                    Some(m) => m as u64,
                    None => continue,
                };

                let targets: Vec<u32> = if scanner {
                    live.iter().collect()
                } else {
                    vec![snap_max as u32]
                };

                for o in targets {
                    let o64 = o as u64;
                    let got = tuples.get(Ordinal(o), &COLS).expect("get");
                    tally.gets.fetch_add(1, Ordering::Relaxed);
                    match got {
                        RowGet::Live(values) => {
                            verify_values(o64, &values);
                            tally.live_hits.fetch_add(1, Ordering::Relaxed);
                        }
                        // Legal in every world: the writer may have deleted
                        // this ordinal after the snapshot was taken.
                        RowGet::Deleted => {
                            tally.deleted_on_live.fetch_add(1, Ordering::Relaxed);
                        }
                        // THE observation this whole file exists to count.
                        RowGet::Missing => {
                            let (counter, role) = if scanner {
                                (&tally.missing_scan, "SCANNER")
                            } else {
                                (&tally.missing_probe, "PROBER")
                            };
                            counter.fetch_add(1, Ordering::Relaxed);
                            tally.record(format!(
                                "{role}: ordinal {o64} in live() (len={snap_len}, max={snap_max}) \
                                 but tuple store says Missing; writer had acked {} ops",
                                acked.load(Ordering::Relaxed)
                            ));
                        }
                    }
                }
            }
        }));
    }

    let (live, next) = drive_writer(&db, seed, &acked);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("reader thread panicked");
    }

    assert_settled(&meta, &tuples, &live, next);

    // A run that never raced would report zero missings for the wrong reason.
    // Pin that it did real work before its silence means anything.
    assert!(
        tally.snapshots.load(Ordering::Relaxed) > 1_000,
        "readers barely ran; the result would not be meaningful"
    );
    assert!(
        tally.live_hits.load(Ordering::Relaxed) > 1_000,
        "readers saw almost no rows"
    );
    assert!(next > SEED_ROWS, "writer made no progress");

    eprintln!(
        "\nwriter: {} acked ops, {next} ordinals allocated, {} live at rest",
        acked.load(Ordering::Relaxed),
        live.len()
    );
    let sightings = tally.report(&format!("seed {seed:#x}"));
    db.close().unwrap();
    sightings
}

/// THE DECIDER. Runs the race hard and **asserts** whether `Missing` on a live
/// ordinal is observable, against whatever [`EXPECTED_WORLD`] says this build
/// owes. Today: **WORLD A**, ~20-28k sightings per run. See the file header.
///
/// The assertion is deliberately asymmetric between the two worlds, because the
/// evidence is:
///
///   * **World A checks the PROBER only.** The scanner trips the window about
///     once in three million gets (header), so a run in which it saw nothing
///     proves nothing — asserting on the combined total would be asserting on a
///     coin flip. The prober sees thousands per run; that is the instrument.
///   * **World B checks BOTH roles for zero.** Once liveness is unified behind
///     a single versioned snapshot published after every store already holds
///     the row, the window is closed *structurally* rather than statistically.
///     One sighting from either role is then a real regression, and a rare
///     signal is exactly the one worth keeping.
///
/// The `run_decider` guards (`snapshots > 1_000`, `live_hits > 1_000`,
/// `next > SEED_ROWS`) are what make the World B direction meaningful rather
/// than vacuous: they prove the readers and writer actually ran before silence
/// is allowed to count as evidence.
#[test]
fn live_then_get_under_concurrent_writes() {
    // A genuinely serial host cannot be relied on to interleave two threads
    // across two independent mutexes. Report and bail rather than fail for a
    // reason that has nothing to do with the code under test — the same posture
    // `RETRY_BUDGET` takes toward a loaded box. (Note this cannot reuse
    // `reader_threads()`, which is floored at 4 regardless of the hardware.)
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if parallelism < 2 {
        eprintln!(
            "SKIP: available_parallelism() == {parallelism}; the cross-store \
             window is not demonstrable on a serial host"
        );
        return;
    }

    let a = run_decider(0x9E37_79B9_7F4A_7C15);
    let b = run_decider(0xD1B5_4A32_D192_ED03);
    let scan = a.scan + b.scan;
    let probe = a.probe + b.probe;
    let total = scan + probe;

    eprintln!(
        "\n================ VERDICT ================\n\
         Missing-on-live observations: {total} (scan {scan}, probe {probe})\n\
         expected: {}\n\
         =========================================\n",
        match EXPECTED_WORLD {
            World::A => "WORLD A — the window IS observable; Missing is a legal transient.",
            World::B => "WORLD B — liveness is unified; the window is closed.",
        }
    );

    match EXPECTED_WORLD {
        World::A => assert!(
            probe > 0,
            "EXPECTED_WORLD is A, so liveness still has three independent owners \
             and the meta->tuple window must be observable — but the prober saw \
             0 Missing-on-live across both seeds (scanner saw {scan}). Either the \
             window closed without EXPECTED_WORLD being updated, or the readers \
             never reached the write frontier."
        ),
        World::B => assert_eq!(
            total, 0,
            "EXPECTED_WORLD is B, so no reader may observe an ordinal in live() \
             before the tuple store holds it — but there were {scan} scanner and \
             {probe} prober sightings. See the captured observations above for \
             the offending ordinals."
        ),
    }
}

// NOTE: bare SEARCH's liveness is NOT tested here, deliberately.
//
// A concurrent version of it was written and thrown away: with the applier
// publishing liveness last, the interval in which a vector is searchable but
// not yet live is a few microseconds, and a single `search` call takes longer
// than that. The test scored **578,524 searches / 2,314,096 hits / 0
// violations** against the KNOWN-BROKEN implementation — it could not tell the
// two apart, and an always-green test is worse than no test.
//
// The property is asserted deterministically instead, in `index.rs`, by handing
// `search_filtered` a snapshot that excludes a row the flat index still holds.

// ---------------------------------------------------------------------------
// 3. SNAPSHOT SAFETY — no snapshot admits a row the stores don't hold
// ---------------------------------------------------------------------------

/// The snapshot-based analogue of the PROBER, and the property the whole
/// liveness-unification exists to buy:
///
/// **Every ordinal in a resolved `LiveSet` is fully written in every store.**
///
/// A `LiveSet` is immutable, so unlike a raw `live()` + `get()` sighting this
/// cannot resolve by retrying — a row admitted early is wrong for as long as
/// the snapshot lives. That makes the ordering rule strict: liveness must be
/// published only after flat AND tuple already hold the row.
///
/// Probes the snapshot's MAXIMUM ordinal, for the same reason the prober does:
/// it is the one the applier just published, so it aims straight at the window.
///
/// SCOPE: this is the INSERT-side property — a row must not become enumerable
/// before it is written. The delete side is deliberately not asserted here.
/// `flat.delete` tombstones before `tuple.delete_row` clears the row, so a
/// snapshot taken before a delete can briefly see the flat index and the tuple
/// store disagree about it (measured: ~50 sightings per 3s run). That skew is
/// inherent to a DESTRUCTIVE delete and closes only when retiring a row stops
/// destroying its values; agreement between bare SEARCH and a snapshot is its
/// own separate property.
#[test]
fn snapshot_never_contains_an_unwritten_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &cfgs(), opts()).unwrap();
    let tuples = db.tuple_reader(0).expect("tuple reader");
    seed_rows(&db);

    let stop = Arc::new(AtomicBool::new(false));
    let violations = Arc::new(AtomicU64::new(0));
    let resolves = Arc::new(AtomicU64::new(0));
    let checks = Arc::new(AtomicU64::new(0));
    let notes = Arc::new(Mutex::new(Vec::<String>::new()));
    let acked = Arc::new(AtomicU64::new(SEED_ROWS));

    let next = std::thread::scope(|scope| {
    let mut handles = Vec::new();
    for _ in 0..reader_threads() {
        let db = &db;
        let tuples = tuples.clone();
        let stop = stop.clone();
        let violations = violations.clone();
        let resolves = resolves.clone();
        let checks = checks.clone();
        let notes = notes.clone();
        handles.push(scope.spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let set = db.live_snapshot(0).expect("collection 0");
                resolves.fetch_add(1, Ordering::Relaxed);
                let Some(max) = set.iter().max() else {
                    continue;
                };
                checks.fetch_add(1, Ordering::Relaxed);

                match tuples.get(Ordinal(max), &COLS).expect("get") {
                    RowGet::Live(values) => verify_values(max as u64, &values),
                    // Legal: the row was deleted after this snapshot was taken.
                    RowGet::Deleted => {}
                    RowGet::Missing => {
                        violations.fetch_add(1, Ordering::Relaxed);
                        let mut n = notes.lock().unwrap_or_else(|e| e.into_inner());
                        if n.len() < MAX_OBSERVATIONS {
                            n.push(format!(
                                "ordinal {max} is in a v{} snapshot (len={}) but the tuple \
                                 store says Missing",
                                set.version(),
                                set.len()
                            ));
                        }
                    }
                }
            }
        }));
    }

    let (_live, next) = drive_writer(&db, 0x51A2_7E31_0C4D_9B77, &acked);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("reader thread panicked");
    }
    next
    });

    // The same meaningfulness guards the decider uses: silence is only evidence
    // if the readers and the writer actually ran.
    assert!(
        resolves.load(Ordering::Relaxed) > 1_000,
        "readers barely ran; the result would not be meaningful"
    );
    assert!(checks.load(Ordering::Relaxed) > 1_000, "readers saw no rows");
    assert!(next > SEED_ROWS, "writer made no progress");

    let violations = violations.load(Ordering::Relaxed);
    for note in notes.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        eprintln!("  {note}");
    }
    assert_eq!(
        violations, 0,
        "{violations} snapshot(s) admitted a row the stores did not hold — \
         liveness was published before the data"
    );

    db.close().unwrap();
}
