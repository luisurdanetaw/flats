//! Memory pressure from concurrent long-running queries.
//!
//! Snapshot isolation means every open query holds an `Arc<LiveSet>` for its
//! whole life, and a `LiveSet` carries a flat bitset sized to the highest live
//! ordinal. Q concurrent queries pinned to Q distinct versions therefore hold Q
//! independent bitsets — this file measures what that actually costs, and
//! whether it comes back.
//!
//! # Why this is not driven through `Db`
//!
//! A 10M-capacity flat index is a ~320MB mmap, and those pages would dominate
//! RSS and drown the signal being measured. And 10M rows through `Db::insert`
//! is one fsync each — hours. The metadata index is the thing that owns
//! liveness, so the test drives it directly: with a vector-only schema an
//! `insert_row` is little more than a bitmap insert.
//!
//! # Why its own file
//!
//! RSS is process-wide, and `cargo test` runs tests within a binary in
//! parallel. One test per process is the only way the numbers mean anything.
//!
//! Linux-only (`/proc/self/status`), which is v1's only platform anyway.

#![cfg(target_os = "linux")]

use std::num::NonZeroUsize;
use std::sync::Arc;

use flats::metadata::common::Ordinal;
use flats::metadata::index::MetadataIndex;
use flats::{ColumnSpec, Schema};

/// An empty metadata row — the schema is vector-only, so every row is empty and
/// an `insert_row` is little more than a bitmap insert.
fn empty_row() -> flats::Row {
    Vec::new()
}

/// Live ordinals in the collection under test.
const ORDINALS: u32 = 10_000_000;
/// Concurrent long-running queries.
const QUERIES: usize = 8;
/// Bytes of flat bitset per snapshot: one bit per ordinal up to the highest.
const BITS_BYTES: usize = ORDINALS as usize / 8;

/// A field from `/proc/self/status`, in bytes. `VmRSS` is current resident set;
/// `VmHWM` is its high-water mark and NEVER decreases — which is exactly why
/// the return-to-baseline check below cannot use it.
fn proc_status_bytes(field: &str) -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field)
            && let Some(kb) = rest.trim_start_matches(':').trim().split_whitespace().next()
            && let Ok(kb) = kb.parse::<usize>()
        {
            return kb * 1024;
        }
    }
    panic!("{field} not found in /proc/self/status");
}

fn rss() -> usize {
    proc_status_bytes("VmRSS")
}

fn peak_rss() -> usize {
    proc_status_bytes("VmHWM")
}

fn mib(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn vector_only() -> Schema {
    Schema::from_columns(vec![ColumnSpec::Vector {
        name: "vector".into(),
        dim: NonZeroUsize::new(4).unwrap(),
    }])
    .unwrap()
}

#[test]
fn concurrent_snapshots_stay_within_budget_and_come_back() {
    let dir = tempfile::tempdir().unwrap();
    let (mut writer, reader, _lsn) =
        MetadataIndex::open_or_create(dir.path(), vector_only()).unwrap();

    for o in 0..ORDINALS {
        writer.insert_row(Ordinal(o), &empty_row()).unwrap();
    }
    let handle = reader.live_handle();

    let baseline = rss();
    let peak_before = peak_rss();
    eprintln!("baseline RSS {}", mib(baseline));

    // ---- Phase A: Q queries, Q distinct versions -------------------------
    // Each open is pinned to its own version, so none of them can share a
    // snapshot: this is the worst case the budget has to cover.
    let mut pinned: Vec<Arc<flats::metadata::index::LiveSet>> = Vec::new();
    for i in 0..QUERIES {
        if i > 0 {
            // One write between opens. The row is new, so the bitset grows by
            // a word at most — the cost being measured is the COPY, not the row.
            writer.insert_row(Ordinal(ORDINALS + i as u32), &empty_row()).unwrap();
            handle.bump();
        }
        pinned.push(handle.resolve());
    }

    assert_eq!(
        handle.materializations(),
        QUERIES as u64,
        "one materialization per distinct version"
    );
    for (i, a) in pinned.iter().enumerate() {
        for b in &pinned[i + 1..] {
            assert!(
                !Arc::ptr_eq(a, b),
                "snapshots at distinct versions must not be shared"
            );
        }
    }

    let peak = peak_rss();
    let held = rss();
    let growth = held.saturating_sub(baseline);
    let peak_growth = peak.saturating_sub(peak_before);
    eprintln!(
        "{QUERIES} pinned snapshots: RSS {} (+{}), peak +{} — budget {}",
        mib(held),
        mib(growth),
        mib(peak_growth),
        mib(QUERIES * BITS_BYTES * 3)
    );

    assert!(
        peak_growth < QUERIES * BITS_BYTES * 3,
        "peak grew by {} holding {QUERIES} snapshots; budget is {} ({} of bitset per \
         snapshot x3)",
        mib(peak_growth),
        mib(QUERIES * BITS_BYTES * 3),
        mib(BITS_BYTES),
    );

    // ---- Phase B: many queries, few versions ----------------------------
    // THE assertion that separates per-version materialization from per-query:
    // 64 opens across 8 versions must build 8 snapshots, not 64.
    let materializations_before = handle.materializations();
    let mut sharers = Vec::new();
    for v in 0..QUERIES {
        writer
            .insert_row(Ordinal(ORDINALS + 100 + v as u32), &empty_row())
            .unwrap();
        handle.bump();
        let at_this_version: Vec<_> = (0..QUERIES).map(|_| handle.resolve()).collect();
        for s in &at_this_version[1..] {
            assert!(
                Arc::ptr_eq(&at_this_version[0], s),
                "queries opened at ONE version must share ONE snapshot"
            );
        }
        sharers.push(at_this_version);
    }
    assert_eq!(
        handle.materializations() - materializations_before,
        QUERIES as u64,
        "{} queries across {QUERIES} versions must materialize {QUERIES} times",
        QUERIES * QUERIES
    );

    // ---- Drop everything: the Arcs must not leak -------------------------
    drop(pinned);
    drop(sharers);

    // Deterministic leak check, independent of what the allocator does with
    // freed pages: nothing but the handle's own cache may still hold a snapshot.
    let cached = handle.cached().expect("the last resolve is cached");
    assert_eq!(
        Arc::strong_count(&cached),
        2,
        "a dropped query's snapshot is still referenced somewhere"
    );
    drop(cached);

    let after_drop = rss();
    eprintln!(
        "after dropping every snapshot: RSS {} (+{} over baseline)",
        mib(after_drop),
        mib(after_drop.saturating_sub(baseline))
    );

    // ---- Phase C: steady state ------------------------------------------
    // RSS after a drop measures the ALLOCATOR, not us: glibc raises its mmap
    // threshold once it sees large frees, so pages from freed snapshots stay
    // mapped and RSS does not fall. Asserting on that would be asserting on
    // malloc's tuning heuristics.
    //
    // What a leak would actually look like is unbounded growth, so that is what
    // is asserted: run the whole pinning cycle again and require RSS not to
    // climb past the first round's peak. Reused memory means the first round's
    // snapshots really were freed.
    for round in 0..3 {
        let mut again = Vec::new();
        for i in 0..QUERIES {
            writer
                .insert_row(Ordinal(ORDINALS + 1_000 + (round * 100 + i) as u32), &empty_row())
                .unwrap();
            handle.bump();
            again.push(handle.resolve());
        }
        assert_eq!(again.len(), QUERIES);
        drop(again);
    }

    let steady = rss();
    eprintln!(
        "after 3 more rounds of {QUERIES} snapshots: RSS {} (+{} over post-drop)",
        mib(steady),
        mib(steady.saturating_sub(after_drop))
    );
    assert!(
        steady <= after_drop + BITS_BYTES,
        "RSS climbed from {} to {} across three more identical rounds — snapshots \
         are being retained, not reused",
        mib(after_drop),
        mib(steady)
    );
}

/// The worst-case SHAPE: one live row at a very high ordinal. `bits` is sized to
/// the highest live ordinal rather than the live count, so this snapshot costs
/// the full bitset to describe a single row.
///
/// Recorded deliberately. If the number is uncomfortable, the fix is for
/// `LiveSet` to skip the bitset when the set is too sparse to justify it and let
/// `search_filtered` fall back to a roaring probe — a change to a private
/// representation and one call site, which is why this measurement belongs
/// before the on-disk format is frozen.
#[test]
#[ignore = "diagnostic: run with --ignored to record the sparse worst case"]
fn sparse_high_ordinal_costs_a_full_bitset() {
    let dir = tempfile::tempdir().unwrap();
    let (mut writer, reader, _lsn) =
        MetadataIndex::open_or_create(dir.path(), vector_only()).unwrap();
    writer.insert_row(Ordinal(0), &empty_row()).unwrap();
    writer.insert_row(Ordinal(ORDINALS - 1), &empty_row()).unwrap();

    let handle = reader.live_handle();
    let before = rss();
    let set = handle.resolve();
    let after = rss();

    eprintln!(
        "2 live rows spanning {ORDINALS} ordinals: snapshot cost {} (bits {} words = {})",
        mib(after.saturating_sub(before)),
        set.bits().len(),
        mib(set.bits().len() * 8)
    );
    assert_eq!(set.len(), 2);
    assert_eq!(set.bits().len(), (ORDINALS as usize - 1) / 64 + 1);
}
