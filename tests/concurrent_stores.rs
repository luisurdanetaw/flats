//! Cross-store visibility: what a concurrent reader can observe BETWEEN the
//! metadata index and the tuple store.
//!
//! ============================================================================
//! WORLD VERDICT: **WORLD A — `Missing` on a live ordinal is a LEGAL TRANSIENT.**
//!
//! The race is not theoretical and it is not rare. `live_then_get_under_
//! concurrent_writes` observes it **~20-28k times in ~6 seconds** across two
//! seeds on an 8-thread dev box. `missing_is_transient_not_corruption` then
//! proves every sighting resolves: worst observed transient **83 retry
//! attempts / ~920µs**, typically far less.
//!
//! POLICY THE CURSOR MUST IMPLEMENT:
//!
//!   * `RowGet::Missing` for an ordinal enumerated from `live()` is **NOT an
//!     error and NOT corruption**. The cursor must **SKIP the row** and keep
//!     going. It must never propagate an error, panic, or abort the scan.
//!   * Skipping is *semantically* correct, not just pragmatic: apply runs
//!     BEFORE the ack (`wal.rs`, `commit_batch` — "COMMIT POINT crossed. Now
//!     apply (post-fsync) then ack"), so an ordinal that is in `live()` while
//!     the tuple store still says `Missing` belongs to an insert **whose ack
//!     has not yet been sent to its caller**. No query is obliged to observe a
//!     write that hasn't returned to the writer yet.
//!   * A bounded retry is also sound — `missing_is_transient_not_corruption`
//!     proves every sighting resolves — but it buys nothing except latency: it
//!     would pull in a write that was still in flight when the scan started.
//!     **Prefer skip.**
//!   * `RowGet::Deleted` for an enumerated ordinal is likewise legal (deleted
//!     after the snapshot) — skip that too.
//!   * What the cursor may still treat as loud: a row whose VALUES are wrong.
//!     That invariant holds in every world and is asserted throughout here.
//!
//! Note for whoever writes the enumeration loop: the two reader roles hit this
//! at wildly different rates, and the difference is a trap. The frontier PROBER
//! sees thousands per run; the SCANNER — the cursor's exact future loop — saw
//! **1 sighting in ~3.1M gets** on one run and 0 on the next. An ascending walk
//! usually reaches the frontier long after the window shut, so the plain loop
//! *rarely* notices. **Rarely is not never**: the scan role demonstrably trips
//! it. A cursor that treats `Missing` as an error would be a once-in-millions
//! production panic — the worst possible failure shape. Handle it.
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
/// How long a `Missing` ordinal may stay missing before we call it a lost row.
/// Measured resolve time is microseconds; this is ~6 orders of magnitude of
/// headroom so a loaded CI box can never fail this for the wrong reason.
const RETRY_BUDGET: Duration = Duration::from_secs(5);

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

/// What the reader threads saw. Counters only — the decider does NOT assert on
/// `missing_*`; it reports them, and the reported result picks the policy.
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

    /// Print the tally and return the total `Missing`-on-live count.
    fn report(&self, label: &str) -> u64 {
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
        missing_scan + missing_probe
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
/// Returns the total `Missing`-on-live count. The caller reports; nothing here
/// asserts a world.
fn run_decider(seed: u64) -> u64 {
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
    let missing = tally.report(&format!("seed {seed:#x}"));
    db.close().unwrap();
    missing
}

/// THE DECIDER. Runs the race hard and reports whether `Missing` on a live
/// ordinal is observable. Asserts only the invariants that hold in BOTH worlds
/// — the tally, not this test's pass/fail, picked the cursor's policy.
///
/// Result: **WORLD A**, ~20-28k sightings per run. See the file header.
#[test]
fn live_then_get_under_concurrent_writes() {
    let a = run_decider(0x9E37_79B9_7F4A_7C15);
    let b = run_decider(0xD1B5_4A32_D192_ED03);
    let total = a + b;

    eprintln!(
        "\n================ VERDICT ================\n\
         Missing-on-live observations: {total}\n\
         {}\n\
         =========================================\n",
        if total == 0 {
            "WORLD B on this host: the meta->tuple window was not observed."
        } else {
            "WORLD A: the window IS observable; Missing is a legal transient."
        }
    );
}

// ---------------------------------------------------------------------------
// 2. WORLD A LOCK-IN — the contract the cursor relies on
// ---------------------------------------------------------------------------

/// World A's regression guard: a `Missing` observed on a live ordinal is a row
/// **mid-apply**, never a lost one. Every sighting must become gettable within
/// `RETRY_BUDGET`.
///
/// This is the assertion that licenses the cursor to skip `Missing` instead of
/// erroring. If a future change to the apply fan-out ever leaves a row
/// permanently missing from the tuple store while it is live in the metadata
/// index, this test fails — loudly, with the ordinal.
///
/// Every reader is a PROBER here: the scan role provably never trips the window
/// (see the header), so it would only dilute the sample.
#[test]
fn missing_is_transient_not_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &cfgs(), opts()).unwrap();
    let meta = db.metadata_reader(0).expect("metadata reader");
    let tuples = db.tuple_reader(0).expect("tuple reader");
    seed_rows(&db);

    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(AtomicU64::new(0));
    let acked = Arc::new(AtomicU64::new(SEED_ROWS));
    // How hard the retries had to work — reported so the cursor's author knows
    // the real magnitude of the window rather than guessing at it.
    let max_attempts = Arc::new(AtomicU64::new(0));
    let max_nanos = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for _ in 0..reader_threads() {
        let meta = meta.clone();
        let tuples = tuples.clone();
        let stop = stop.clone();
        let observed = observed.clone();
        let max_attempts = max_attempts.clone();
        let max_nanos = max_nanos.clone();
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let live = meta.live();
                let Some(frontier) = live.max() else { continue };
                if !matches!(
                    tuples.get(Ordinal(frontier), &COLS).expect("get"),
                    RowGet::Missing
                ) {
                    continue;
                }
                observed.fetch_add(1, Ordering::Relaxed);

                // Caught one mid-apply. Retry until it resolves — to a live row
                // (the insert's `write_row` landed) or to the deleted-marker
                // (the writer deleted it afterwards; applies are sequential on
                // the WAL thread, so that too proves `write_row` ran).
                let start = Instant::now();
                let mut attempts = 0u64;
                let resolved = loop {
                    attempts += 1;
                    match tuples.get(Ordinal(frontier), &COLS).expect("get") {
                        RowGet::Live(values) => {
                            verify_values(frontier as u64, &values);
                            break true;
                        }
                        RowGet::Deleted => break true,
                        RowGet::Missing => {}
                    }
                    if start.elapsed() > RETRY_BUDGET {
                        break false;
                    }
                    std::thread::yield_now();
                };
                let elapsed = start.elapsed();
                assert!(
                    resolved,
                    "ordinal {frontier} was in live() but stayed Missing for {elapsed:?} across \
                     {attempts} attempts — that is a LOST ROW, not a mid-apply transient. The \
                     cursor's skip-on-Missing policy is no longer safe; see this file's header."
                );
                max_attempts.fetch_max(attempts, Ordering::Relaxed);
                max_nanos.fetch_max(elapsed.as_nanos() as u64, Ordering::Relaxed);
            }
        }));
    }

    let (live, next) = drive_writer(&db, 0x5DEE_CE66_D1B5_4A32, &acked);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("reader thread panicked");
    }

    assert_settled(&meta, &tuples, &live, next);

    let seen = observed.load(Ordering::Relaxed);
    eprintln!(
        "\n[transience] {seen} Missing sightings, all resolved. \
         worst case: {} retry attempts / {:?}",
        max_attempts.load(Ordering::Relaxed),
        Duration::from_nanos(max_nanos.load(Ordering::Relaxed)),
    );

    // Guard against a vacuous pass. The decider sees thousands of sightings per
    // second, so a run that sees none did not exercise the contract at all.
    // (If this ever proves flaky on a very small CI runner, raise `WRITE_FOR`
    // rather than dropping the assertion — a silent no-op test is worse.)
    assert!(
        seen > 0,
        "no Missing was observed, so the retry contract went untested — the race is known \
         reachable on this codebase (see live_then_get_under_concurrent_writes)"
    );
}
