//! WAL ↔ index reconciliation and crash-mid-checkpoint convergence, ported to
//! the SWMR `(Writer, Reader)` split.
//!
//! Because the `Writer` is single-owner, these tests own it directly during the
//! "session 1" setup so they can stop at individual checkpoint steps (modelling
//! a crash at that exact boundary). WAL frames for that session are produced by
//! appending through a no-op (`LogOnly`) applier, while the test mirrors each
//! record into its own `Writer` — exactly what the real applier would do, split
//! apart so the test keeps step-level control. "Session 2" then recovers into a
//! fresh writer through the normal WAL path and reads via the matching `Reader`.
//!
//! True torn-page writes can't be reproduced in-process, so the interrupted
//! header-flush boundary is modelled by corrupting a slot on disk and asserting
//! the double-buffered header falls back to the previous good slot.

use std::io;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use flats::index::index::{FlatIndex, Reader, Writer};
use flats::wal::wal::{Apply, Lsn, Record, Wal, WalHandle};

// Slot 1 lives on page 1 (mirrored from index.rs for the torn-flush simulation).
const SLOT1_OFFSET: u64 = 4096;

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn to_io(e: flats::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

fn insert_rec(ordinal: u64) -> Record {
    Record::Insert {
        collection: 0,
        ordinal,
        vector: vec![ordinal as f32, 1.0],
        metadata: vec![],
    }
}

/// Session-1 applier: logs frames only. The test mirrors records into its own
/// `Writer`, so apply here is a no-op.
struct LogOnly;
impl Apply for LogOnly {
    fn apply(&mut self, _lsn: Lsn, _record: &Record) -> io::Result<()> {
        Ok(())
    }
    fn checkpoint(&mut self) -> io::Result<Option<u64>> {
        Ok(None)
    }
}

/// Session-2 applier: owns the recovery `Writer` and counts how many records it
/// actually replays — the key signal for "only the tail was replayed".
struct WriterApplier {
    writer: Writer,
    applied: Arc<AtomicUsize>,
}
impl Apply for WriterApplier {
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
        match record {
            Record::Insert {
                ordinal, vector, ..
            } => self.writer.write_at(*ordinal, vector).map_err(to_io)?,
            Record::Delete { ordinal, .. } => self.writer.delete(*ordinal).map_err(to_io)?,
            Record::CreateCollection { .. } => {
                unreachable!("this test never logs CreateCollection records")
            }
        }
        self.writer.advance_applied_lsn(lsn.0);
        self.applied.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn checkpoint(&mut self) -> io::Result<Option<u64>> {
        let wm = self.writer.sync().map_err(to_io)?;
        Ok(if wm == 0 { None } else { Some(wm) })
    }
}

/// Log a record AND mirror it into the test-owned writer, in lockstep (this is
/// what the real applier does; split so the test keeps the writer).
fn log_and_apply(h: &WalHandle, w: &mut Writer, rec: Record) {
    let lsn = h.append(rec.clone()).expect("append durable");
    match &rec {
        Record::Insert {
            ordinal, vector, ..
        } => w.write_at(*ordinal, vector).expect("write_at"),
        Record::Delete { ordinal, .. } => w.delete(*ordinal).expect("delete"),
        Record::CreateCollection { .. } => {
            unreachable!("this test never logs CreateCollection records")
        }
    }
    w.advance_applied_lsn(lsn.0);
}

/// Recover from `wal_path` into a freshly-opened index. Returns how many records
/// were replayed and the reader for assertions.
fn recover(idx_path: &std::path::Path, wal_path: &std::path::Path) -> (usize, Reader) {
    let (writer, reader) = FlatIndex::open(idx_path).expect("open index");
    let skip = writer.checkpoint_lsn();
    let applied = Arc::new(AtomicUsize::new(0));
    let applier = WriterApplier {
        writer,
        applied: applied.clone(),
    };
    let wal = Wal::start(wal_path, applier, skip).expect("wal start");
    wal.shutdown();
    (applied.load(Ordering::SeqCst), reader)
}

/// Assert the index holds exactly ordinals `0..expected` — none lost, none
/// duplicated, none phantom.
fn assert_exactly(r: &Reader, expected: u64) {
    assert_eq!(r.len() as u64, expected, "high-water mark");
    let hits = r.search(&[0.0, 1.0], 1024).expect("search");
    assert_eq!(hits.len() as u64, expected, "result count (no dups/losses)");
    let mut ids: Vec<u32> = hits.iter().map(|h| h.id.0).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..expected as u32).collect::<Vec<_>>());
}

fn assert_ordinal_dead(r: &Reader, dead: u32) {
    let hits = r.search(&[0.0, 1.0], 1024).expect("search");
    assert!(
        hits.iter().all(|h| h.id.0 != dead),
        "ordinal {dead} must not surface"
    );
}

#[test]
fn reconciliation_replays_only_the_tail() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();

        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        let watermark = w.sync().unwrap(); // full checkpoint
        assert_eq!(watermark, 3);
        h.truncate(watermark).unwrap(); // WAL sheds LSNs 1..=3

        for ord in 3..5 {
            log_and_apply(&h, &mut w, insert_rec(ord)); // LSNs 4, 5 survive
        }
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 2, "only the un-checkpointed tail is replayed");
    assert_exactly(&reader, 5);
}

#[test]
fn crash_between_a_and_b_data_synced_watermark_not_written() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        w.sync_data().unwrap(); // step a ONLY, then "crash"
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 3, "no watermark => full replay");
    assert_exactly(&reader, 3);
}

#[test]
fn crash_between_b_and_c_torn_header_falls_back_to_previous_slot() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        let (count, last_lsn) = w.begin_checkpoint();
        w.sync_data().unwrap(); // a
        w.stage_watermark(count, last_lsn).unwrap(); // b — writes spare slot 1
        // crash before sync_header (c): slot 1 never durably committed
        drop(h);
        wal.shutdown();
        drop(w);
    }

    // Model the torn flush: clobber slot 1 so it fails validation on reopen.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&idx_path)
            .unwrap();
        f.seek(SeekFrom::Start(SLOT1_OFFSET)).unwrap();
        f.write_all(&[0xAB; 32]).unwrap();
        f.sync_all().unwrap();
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 3, "fell back to prior slot (watermark 0) => full replay");
    assert_exactly(&reader, 3);
}

#[test]
fn crash_between_c_and_d_header_durable_wal_not_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        assert_eq!(w.sync().unwrap(), 3); // a, b, c — but DO NOT truncate
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 0, "entire prefix already durable => nothing replayed");
    assert_exactly(&reader, 3);
}

#[test]
fn crash_after_d_e_truncated_wal_converges() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        let watermark = w.sync().unwrap();
        h.truncate(watermark).unwrap(); // step e completed
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 0, "truncated WAL has no tail to replay");
    assert_exactly(&reader, 3);
}

#[test]
fn delete_replays_idempotently() {
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..3 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        log_and_apply(
            &h,
            &mut w,
            Record::Delete {
                collection: 0,
                ordinal: 1,
            },
        );
        // No checkpoint: the whole log (inserts + delete) replays on reopen.
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (_replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(reader.len(), 3, "high-water unchanged by delete");
    let hits = reader.search(&[0.0, 1.0], 1024).unwrap();
    let ids: Vec<u32> = hits.iter().map(|h| h.id.0).collect();
    assert!(!ids.contains(&1), "tombstoned ordinal stays hidden after replay");
    assert_eq!(hits.len(), 2);
    assert!(ids.contains(&0) && ids.contains(&2));
}

#[test]
fn tombstone_consistent_when_crash_before_watermark() {
    // insert 0..6, delete 5, sync_data (a) only, crash before watermark.
    // INVARIANT (not mechanism): ordinal 5 ends dead — here because the watermark
    // never advanced, so the Delete replays from the WAL.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..6 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        log_and_apply(
            &h,
            &mut w,
            Record::Delete {
                collection: 0,
                ordinal: 5,
            },
        );
        w.sync_data().unwrap(); // step a only; crash before b
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (_replayed, reader) = recover(&idx_path, &wal_path);
    assert_ordinal_dead(&reader, 5);
    assert_eq!(reader.search(&[0.0, 1.0], 1024).unwrap().len(), 5);
}

#[test]
fn tombstone_durable_before_watermark_isolated() {
    // Strict ordering proof: checkpoint PAST the delete AND truncate the WAL, so
    // the Delete is gone from the log. After a reopen with nothing to replay,
    // ordinal 5 can only be dead if sync_data made the bitset durable BEFORE the
    // watermark advanced.
    let dir = tempfile::tempdir().unwrap();
    let idx_path = dir.path().join("c0.idx");
    let wal_path = dir.path().join("wal.log");

    {
        let (mut w, _r) = FlatIndex::create(&idx_path, dim(2), 64).unwrap();
        let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
        let h = wal.handle();
        for ord in 0..6 {
            log_and_apply(&h, &mut w, insert_rec(ord));
        }
        log_and_apply(
            &h,
            &mut w,
            Record::Delete {
                collection: 0,
                ordinal: 5,
            },
        );
        let watermark = w.sync().unwrap(); // a, b, c — bitset durable, then watermark
        h.truncate(watermark).unwrap(); // WAL drops inserts + the Delete
        drop(h);
        wal.shutdown();
        drop(w);
    }

    let (replayed, reader) = recover(&idx_path, &wal_path);
    assert_eq!(replayed, 0, "WAL truncated past the delete; nothing replays");
    assert_eq!(reader.len(), 6, "high-water includes slot 5");
    assert_ordinal_dead(&reader, 5); // dead purely from the persisted bitset
    assert_eq!(reader.search(&[0.0, 1.0], 1024).unwrap().len(), 5);
}
