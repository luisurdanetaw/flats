//! SWMR-specific tests: the concurrent reader/writer races that the sequential
//! chaos harness does not exercise.
//!
//! NOTE on what these can and cannot catch: x86 is strongly ordered, so a
//! *missing* Release/Acquire would often still pass here. These tests prove the
//! design works (no panics, no torn/zero reads, no id↔vector mismatch) and act
//! as a regression guard on weakly-ordered targets (aarch64). The ordering
//! itself is argued by construction in `index.rs`. Miri would catch data races
//! directly but cannot map files (mmap), so it can't run these; that's the gap
//! these black-box stress tests fill.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flats::index::index::{FlatIndex, Reader, Writer};
use static_assertions::{assert_impl_all, assert_not_impl_any};

// (a) Compile-time: the single-writer invariant is type-enforced. If `Writer`
// ever gains `Clone`, this fails to compile. Readers are freely shareable.
assert_not_impl_any!(Writer: Clone);
assert_impl_all!(Writer: Send);
assert_impl_all!(Reader: Clone, Send, Sync);

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn readers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Deterministic per-ordinal pattern: a constant vector whose value is bounded
/// and exactly representable in f32, so a query of all-ones yields a score that
/// is an exact function of the ordinal. A torn or unpublished read would break
/// that relationship.
fn pattern(ordinal: usize, d: usize) -> Vec<f32> {
    vec![((ordinal % 7) + 1) as f32; d]
}
fn expected_score(id: u32, d: usize) -> f32 {
    (d as f32) * (((id as usize % 7) + 1) as f32)
}

#[test]
fn concurrent_reads_never_torn_or_misindexed() {
    let d = 16;
    let cap = 200_000;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.idx");
    let (mut writer, reader) = FlatIndex::create(&path, dim(d), cap).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let query = vec![1.0f32; d];

    // N reader threads: search in a tight loop; every result's score must equal
    // the exact value implied by its id — proving the vector at that id was
    // fully written before the id became visible, and that ids map to the right
    // stored vector.
    let mut handles = Vec::new();
    for _ in 0..readers() {
        let r: Reader = reader.clone();
        let stop = stop.clone();
        let q = query.clone();
        handles.push(std::thread::spawn(move || {
            let mut observed = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let hits = r.search(&q, 8).expect("search");
                for h in &hits {
                    assert_eq!(
                        h.score,
                        expected_score(h.id.0, d),
                        "torn/misindexed read at ordinal {}",
                        h.id.0
                    );
                    observed += 1;
                }
            }
            observed
        }));
    }

    // Single writer thread: append the pattern at the frontier for ~1s.
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut i = 0usize;
    while i < cap && Instant::now() < deadline {
        writer.write_at(i as u64, &pattern(i, d)).expect("write_at");
        i += 1;
    }
    stop.store(true, Ordering::Relaxed);

    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(i > 0, "writer made progress");
    assert!(total > 0, "readers observed committed vectors");
    // Final consistency: everything the writer published is visible & correct.
    let final_len = reader.len();
    assert_eq!(final_len, i, "reader sees exactly what the writer published");
}

#[test]
fn publication_never_exposes_a_zero_slot() {
    // Release/Acquire proof by construction: the writer fills a slot with a
    // non-zero sentinel BEFORE publishing the count. If a reader could observe
    // the incremented count before the bytes, it would read a zero vector
    // (score 0). We assert it never does, across many publish/observe rounds.
    let d = 8;
    let cap = 50_000;
    let sentinel = vec![9.0f32; d];
    let want = (d as f32) * 9.0;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.idx");
    let (mut writer, reader) = FlatIndex::create(&path, dim(d), cap).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let query = vec![1.0f32; d];

    let mut handles = Vec::new();
    for _ in 0..readers() {
        let r = reader.clone();
        let stop = stop.clone();
        let q = query.clone();
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                // Every visible vector is the sentinel; a zero (unpublished)
                // slot would score 0 and trip this.
                for h in r.search(&q, 4).expect("search") {
                    assert_eq!(h.score, want, "observed a zero/torn slot at {}", h.id.0);
                }
            }
        }));
    }

    let deadline = Instant::now() + Duration::from_secs(1);
    let mut i = 0usize;
    while i < cap && Instant::now() < deadline {
        writer.write_at(i as u64, &sentinel).expect("write_at");
        i += 1;
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
    assert!(i > 0);
}

/// Populate a read-only index and return its reader. The writer is dropped, so
/// the index is immutable for the measurement.
fn populated_reader(d: usize, count: usize) -> (tempfile::TempDir, Reader) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.idx");
    let (mut w, r) = FlatIndex::create(&path, dim(d), count).unwrap();
    for i in 0..count {
        let v: Vec<f32> = (0..d).map(|j| ((i + j) % 13) as f32).collect();
        w.write_at(i as u64, &v).unwrap();
    }
    drop(w); // read-only from here
    (dir, r)
}

/// Total searches/sec across `n` threads, each searching for `dur`.
fn measure_qps(reader: &Reader, query: &[f32], k: usize, n: usize, dur: Duration) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..n {
        let r = reader.clone();
        let q = query.to_vec();
        let stop = stop.clone();
        handles.push(std::thread::spawn(move || {
            let mut c = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let _ = r.search(&q, k).expect("search");
                c += 1;
            }
            c
        }));
    }
    std::thread::sleep(dur);
    stop.store(true, Ordering::Relaxed);
    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    total as f64 / start.elapsed().as_secs_f64()
}

#[test]
fn parallel_search_scales_with_threads() {
    let d = 64;
    let count = 4000;
    let k = 10;
    let (_dir, reader) = populated_reader(d, count);
    let query: Vec<f32> = (0..d).map(|j| (j % 13) as f32).collect();

    let cores = readers();
    let dur = Duration::from_millis(400);

    let single = measure_qps(&reader, &query, k, 1, dur);
    let multi = measure_qps(&reader, &query, k, cores, dur);
    let scaling = multi / single;
    eprintln!(
        "parallel search: 1 thread = {single:.0} q/s, {cores} threads = {multi:.0} q/s, \
         scaling = {scaling:.2}x"
    );

    // Conservative bounds to stay green on noisy/shared CI while still catching a
    // real regression (e.g. accidental serialization or count false-sharing):
    if cores >= 4 {
        assert!(
            scaling >= 1.8,
            "expected >=1.8x on {cores} cores (lock-free reads should scale), got {scaling:.2}x"
        );
    } else if cores >= 2 {
        assert!(
            scaling >= 1.2,
            "expected some scaling on {cores} cores, got {scaling:.2}x"
        );
    }
}
