//! Integration tests for `FlatIndex` (CLAUDE.md §13).
//!
//! These differ from the inline unit tests in `src/index/index.rs` in scope,
//! not in "uses a real file" — both touch real files, and that's correct for
//! mmap-backed code (see CLAUDE.md §9). What belongs here instead:
//!
//! - Scenarios that cross more than one `FlatIndex` handle on the same file
//!   (the SWMR contract described in CLAUDE.md §4 / index.rs's module docs),
//!   which a single-struct unit test can't exercise.
//! - The hand-rolled property test required by CLAUDE.md §13: random `dim`,
//!   random vector counts, random `k`, checked against an in-memory
//!   reference. No `proptest`/`quickcheck` — those are third-party crates.

use std::num::NonZeroUsize;

use flats::index::index::{FlatIndex, Ordinal};

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("non-zero")
}

/// A second, independent `FlatIndex` handle opened on the same path after a
/// writer's `sync()` must see the writer's data — without the writer having
/// closed first. This is the actual multi-handle scenario CLAUDE.md §4
/// describes ("Readers go directly against the mmap"); it's distinct from
/// "close the writer, then reopen," which the inline `reopen_sees_synced_data`
/// unit test already covers and which hides bugs where a reader's state is
/// accidentally tied to the writer's handle being dropped.
#[test]
fn second_handle_sees_writer_data_after_sync() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx.bin");

    let mut writer = FlatIndex::create(&path, dim(4), 16).unwrap();
    writer.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
    writer.sync().unwrap();

    // Writer handle is still alive and mapped — this is not a "reopen after
    // close" test.
    let reader = FlatIndex::open(&path).unwrap();
    assert_eq!(reader.len(), 2);
    let results = reader.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, Ordinal(0));

    // A second batch, synced after the reader was opened. The reader's
    // `count` is cached at open() time from the on-disk header, not re-read
    // from the mmap on every call — so it should NOT observe this write
    // without reopening. This pins down current behavior; if that caching
    // strategy ever changes, this assertion is the one that should move.
    writer.insert(&[0.0, 0.0, 1.0, 0.0]).unwrap();
    writer.sync().unwrap();
    assert_eq!(
        reader.len(),
        2,
        "a handle opened before a later write+sync should stay pinned to the \
         count it saw at open() time, not observe later writes live"
    );

    // A fresh handle opened after the second sync sees everything.
    let reader2 = FlatIndex::open(&path).unwrap();
    assert_eq!(reader2.len(), 3);
}

/// CLAUDE.md §13: "Property tests (hand-rolled, no proptest/quickcheck):
/// random vectors, random dim, random batch sizes. Compare against in-memory
/// reference."
///
/// The reference here scores with the same `flats::dot` the index itself
/// uses — `dot`'s own bit-exactness across SIMD paths is `simd_parity.rs`'s
/// job, not this test's. What this test is actually checking is `FlatIndex`'s
/// plumbing: insert layout/offsets, the top-k heap, and that ids map back to
/// the right stored vectors. So scores compare bit-exact (`==`, no
/// tolerance — same fn, same inputs, must match), while id *order* on exact
/// score ties is intentionally not asserted: a heap-based partial selection
/// and a full stable sort are not guaranteed to break ties the same way, and
/// asserting that would make the test flaky for the wrong reason.
#[test]
fn random_search_matches_naive_reference() {
    // Seed from the clock so repeated runs explore new input space, but
    // print it on failure so a red run is reproducible by hardcoding it here.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut rng = SplitMix64::new(seed);

    for trial in 0..20 {
        let dim_n = 1 + rng.next_usize(64);
        let count = 1 + rng.next_usize(200);
        let k = 1 + rng.next_usize(count.min(20));

        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..dim_n).map(|_| rng.next_f32()).collect())
            .collect();
        let query: Vec<f32> = (0..dim_n).map(|_| rng.next_f32()).collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.bin");
        let mut idx = FlatIndex::create(&path, dim(dim_n), count).unwrap();
        for v in &vectors {
            idx.insert(v).unwrap();
        }

        let got = idx.search(&query, k).unwrap();

        // In-memory reference: brute-force every vector, full sort.
        let mut reference: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(id, v)| (id, flats::dot(&query, v)))
            .collect();
        reference.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let reference_topk = &reference[..k];

        let ctx = || format!("seed={seed} trial={trial} dim={dim_n} count={count} k={k}");

        assert_eq!(got.len(), reference_topk.len(), "{}", ctx());

        // Score multisets must match exactly (same scoring fn, so no
        // tolerance is appropriate here — a mismatch means the heap dropped
        // or duplicated a candidate).
        let mut got_scores: Vec<f32> = got.iter().map(|r| r.score).collect();
        let mut ref_scores: Vec<f32> = reference_topk.iter().map(|(_, s)| *s).collect();
        got_scores.sort_unstable_by(f32::total_cmp);
        ref_scores.sort_unstable_by(f32::total_cmp);
        assert_eq!(got_scores, ref_scores, "{}", ctx());

        // Every returned (id, score) pair must reflect the actually-stored
        // vector at that id — catches off-by-one / offset bugs in insert()
        // or vectors() that a pure score-multiset check could miss.
        for hit in &got {
            let expected = flats::dot(&query, &vectors[hit.id.0 as usize]);
            assert_eq!(hit.score, expected, "{}", ctx());
        }
    }
}

/// SplitMix64 — a tiny, deterministic, dependency-free PRNG. Not
/// cryptographically anything; just needs to scatter inputs across the
/// search space for the property test above (CLAUDE.md §13: "hand-rolled, no
/// proptest/quickcheck").
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in `[-1.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 11; // 53 usable bits, mirrors f64 mantissa tricks
        (bits as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }

    /// Uniform in `[0, bound)`. `bound` must be > 0.
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}
