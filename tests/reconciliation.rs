//! WAL ↔ index reconciliation and crash-mid-checkpoint convergence.
//!
//! These drive the WAL and a `FlatIndex` directly (rather than through `Db`) so
//! each individual checkpoint step (msync data → stage watermark → msync header
//! → truncate WAL) can be stopped at, modelling a crash at that exact boundary.
//! A counting applier reports how many records recovery actually replays, which
//! is what lets us assert "only the tail was replayed" rather than just "the end
//! state happens to be right."
//!
//! True torn-page writes can't be reproduced in-process (the page cache makes an
//! un-synced write visible on reopen), so the boundary where a header flush is
//! interrupted is modelled by corrupting that slot on disk — exactly the state a
//! torn flush leaves behind — and asserting the double-buffered header falls back
//! to the previous good slot.

use std::io;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use flats::index::index::{FlatIndex, Ordinal};
use flats::wal::wal::{Apply, Lsn, Record, Wal, WalHandle};

// Header geometry mirrored from index.rs for the torn-flush simulation. Slot 1
// lives on page 1; the first checkpoint writes it (seq 1) and makes it active.
const PAGE: u64 = 4096;
const SLOT1_OFFSET: u64 = PAGE;

/// Applies records into a real index and counts how many it applied. The count
/// is the whole point: recovery's replay count must equal the un-checkpointed
/// tail, not the full log.
struct CountingApplier {
    idx: Arc<Mutex<FlatIndex>>,
    applied: Arc<AtomicUsize>,
}

impl Apply for CountingApplier {
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
        let mut g = self.idx.lock().expect("index lock");
        match record {
            Record::Insert {
                ordinal, vector, ..
            } => g.write_at(*ordinal, vector).map_err(to_io)?,
            Record::Delete { ordinal, .. } => g.delete(*ordinal).map_err(to_io)?,
        }
        g.advance_applied_lsn(lsn.0);
        self.applied.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn to_io(e: flats::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn append_insert(h: &WalHandle, ordinal: u64) {
    h.append(Record::Insert {
        collection: 0,
        ordinal,
        vector: vec![ordinal as f32, 1.0],
    })
    .expect("append durable");
}

/// Full index checkpoint (steps a → b → c). Returns the durable watermark LSN.
fn checkpoint_index(idx: &Mutex<FlatIndex>) -> u64 {
    let mut g = idx.lock().expect("index lock");
    let (count, last_lsn) = g.begin_checkpoint();
    g.sync_data().expect("sync_data"); // a
    g.stage_watermark(count, last_lsn).expect("stage"); // b
    g.sync_header().expect("sync_header"); // c
    last_lsn
}

/// Assert the index holds exactly ordinals `0..expected`, none duplicated or
/// lost, by searching with a huge k and checking the id set.
fn assert_exactly(idx: &Mutex<FlatIndex>, expected: u64) {
    let g = idx.lock().expect("index lock");
    assert_eq!(g.len() as u64, expected, "high-water mark");
    let hits = g.search(&[0.0, 1.0], 1024).expect("search");
    assert_eq!(hits.len() as u64, expected, "result count (no dups, no losses)");
    let mut ids: Vec<u32> = hits.iter().map(|h| h.id.0).collect();
    ids.sort_unstable();
    let want: Vec<u32> = (0..expected as u32).collect();
    assert_eq!(ids, want, "exactly ordinals 0..expected present");
}

/// Recover from `wal_path` into a freshly-opened index, returning how many
/// records were replayed and the index for further assertions.
fn recover(idx_path: &std::path::Path, wal_path: &std::path::Path) -> (usize, Arc<Mutex<FlatIndex>>) {
    let idx = Arc::new(Mutex::new(FlatIndex::open(idx_path).expect("open index")));
    let skip = idx.lock().unwrap().checkpoint_lsn();
    let applied = Arc::new(AtomicUsize::new(0));
    let applier = CountingApplier {
        idx: idx.clone(),
        applied: applied.clone(),
    };
    let wal = Wal::start(wal_path, applier, skip).expect("wal start");
    wal.shutdown();
    (applied.load(Ordering::SeqCst), idx)
}

#[test]
fn reconciliation_replays_only_the_tail() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    // Session 1: write 3, checkpoint + truncate, then write 2 more (the tail).
    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let applied = Arc::new(AtomicUsize::new(0));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: applied.clone(),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();

        for ord in 0..3 {
            append_insert(&h, ord);
        }
        let watermark = checkpoint_index(&idx);
        assert_eq!(watermark, 3, "durable through LSN 3");
        h.truncate(watermark).unwrap(); // WAL sheds LSNs 1..=3

        for ord in 3..5 {
            append_insert(&h, ord); // LSNs 4, 5 — only these survive in the WAL
        }
        drop(h);
        wal.shutdown();

        assert_eq!(applied.load(Ordering::SeqCst), 5, "session 1 applied all 5 live");
        assert_exactly(&idx, 5);
    }

    // Session 2: recovery must skip the checkpointed prefix and replay only the
    // 2 tail records — yet the end state is all 5.
    let (replayed, idx) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 2, "only the un-checkpointed tail is replayed");
    assert_exactly(&idx, 5);
}

#[test]
fn crash_between_a_and_b_data_synced_watermark_not_written() {
    // Boundary a/b: vector pages durable, header watermark NOT advanced. Recovery
    // sees the old watermark (0) and replays everything; idempotent positional
    // writes land on the already-present data with no duplicates.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: Arc::new(AtomicUsize::new(0)),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            append_insert(&h, ord);
        }
        idx.lock().unwrap().sync_data().unwrap(); // step a ONLY — then "crash"
        drop(h);
        wal.shutdown();
    }

    let (replayed, idx) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 3, "no watermark => full replay");
    assert_eq!(idx.lock().unwrap().checkpoint_lsn(), 0);
    assert_exactly(&idx, 3);
}

#[test]
fn crash_between_b_and_c_torn_header_falls_back_to_previous_slot() {
    // Boundary b/c: the new watermark was written to the spare slot but its flush
    // was interrupted. Modelled by corrupting that slot on disk. The double
    // buffered header must fall back to the previous good slot (watermark 0), and
    // recovery replays everything to converge.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: Arc::new(AtomicUsize::new(0)),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            append_insert(&h, ord);
        }
        {
            let mut g = idx.lock().unwrap();
            let (count, last_lsn) = g.begin_checkpoint();
            g.sync_data().unwrap(); // a
            g.stage_watermark(count, last_lsn).unwrap(); // b — writes spare slot 1
            // crash before sync_header (c): slot 1 never durably committed
        }
        drop(h);
        wal.shutdown();
    }

    // Simulate the torn flush: clobber slot 1's identifying bytes so it fails
    // validation on reopen.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&idx_path).unwrap();
        f.seek(SeekFrom::Start(SLOT1_OFFSET)).unwrap();
        f.write_all(&[0xAB; 32]).unwrap();
        f.sync_all().unwrap();
    }

    let (replayed, idx) = recover(&idx_path, &wal_path);
    assert_eq!(idx.lock().unwrap().checkpoint_lsn(), 0, "fell back to prior slot");
    assert_eq!(replayed, 3, "fallback watermark => full replay");
    assert_exactly(&idx, 3);
}

#[test]
fn crash_between_c_and_d_header_durable_wal_not_truncated() {
    // Boundary c/d (and the pre-truncate half of d/e): header watermark is durable
    // but the WAL was not truncated. Recovery skips the whole (still-present)
    // prefix — replays nothing — and the index already has the data.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: Arc::new(AtomicUsize::new(0)),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            append_insert(&h, ord);
        }
        let watermark = checkpoint_index(&idx); // a, b, c
        assert_eq!(watermark, 3);
        // crash before truncate (d/e): WAL still holds LSNs 1..=3
        drop(h);
        wal.shutdown();
    }

    let (replayed, idx) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 0, "entire prefix already durable => nothing replayed");
    assert_eq!(idx.lock().unwrap().checkpoint_lsn(), 3);
    assert_exactly(&idx, 3);
}

#[test]
fn crash_after_d_e_truncated_wal_converges() {
    // Boundary d/e completed: header durable AND WAL truncated. Recovery has an
    // empty tail to replay and the index is already whole.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: Arc::new(AtomicUsize::new(0)),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            append_insert(&h, ord);
        }
        let watermark = checkpoint_index(&idx);
        h.truncate(watermark).unwrap(); // step e completed
        drop(h);
        wal.shutdown();
    }

    let (replayed, idx) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 0, "truncated WAL has no tail to replay");
    assert_exactly(&idx, 3);
}

#[test]
fn delete_replays_idempotently() {
    // A tombstone replayed onto an already-tombstoned slot stays a no-op, and the
    // deleted ordinal never reappears in search.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let idx = Arc::new(Mutex::new(FlatIndex::create(&idx_path, dim(2), 64).unwrap()));
        let wal = Wal::start(
            &wal_path,
            CountingApplier {
                idx: idx.clone(),
                applied: Arc::new(AtomicUsize::new(0)),
            },
            0,
        )
        .unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            append_insert(&h, ord);
        }
        h.append(Record::Delete {
            collection: 0,
            ordinal: 1,
        })
        .unwrap();
        // No checkpoint: the whole log (inserts + delete) replays on reopen.
        drop(h);
        wal.shutdown();
    }

    let (_replayed, idx) = recover(&idx_path, &wal_path);
    let g = idx.lock().unwrap();
    assert_eq!(g.len(), 3, "high-water unchanged by delete");
    let hits = g.search(&[0.0, 1.0], 1024).unwrap();
    let ids: Vec<u32> = hits.iter().map(|h| h.id.0).collect();
    assert!(!ids.contains(&1), "tombstoned ordinal stays hidden after replay");
    assert_eq!(hits.len(), 2);
    assert!(ids.contains(&0) && ids.contains(&2));
    let _ = Ordinal(0); // silence unused import if assertions change
}
