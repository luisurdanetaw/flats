//! Write-ahead log.
//!
//! Threading model (decided deliberately — read this before changing it):
//!
//!   * ONE thread owns the WAL file *and* drives index apply. Writers hand
//!     records over an mpsc queue and block on a per-record ack channel until
//!     the record is durable. This upholds the index's single-writer invariant
//!     for free (only this thread mutates the indexes) and makes "committed"
//!     and "visible" coincide, so read-your-writes works without extra work.
//!
//!   * GROUP COMMIT: the loop drains every record currently queued, writes all
//!     their frames, and fsyncs *once* for the whole batch before acking any of
//!     them. fsync is the expensive part; batching it is the only reason this
//!     hits useful throughput. Never fsync per record.
//!
//! Ordering invariant (do not reorder):
//!
//! ```text
//! assign LSN -> write frame -> FSYNC -> apply -> ACK
//! ```
//!
//!   fsync strictly before apply keeps the WAL >= the index at all times, which
//!   is what makes recovery "replay the tail onto the index" correct. ack
//!   strictly after fsync is what makes durability honest: we only promise a
//!   write survived once it actually did.
//!
//! Dependency direction: this module knows NOTHING about the vector or metadata
//! index. The engine injects an `Apply` impl. WAL depends on nobody.


use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------


/// Monotonic log sequence number. Assigned by the WAL thread, recorded in the
/// frame, and returned to the writer on ack. Recovery resumes the counter from
/// the highest LSN found in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lsn(pub u64);

/// The LOGICAL mutation — the intent, not the plan, not the physical bytes.
/// This is the *only* thing that goes durable. It must be replayable by a dumb
/// loop that doesn't have the query planner.
///
/// CRITICAL for idempotency: ordinal/position is assigned on the *logging*
/// side and carried here, NOT recomputed at apply time from the current count.
/// "Write vector at ordinal N" replays idempotently; "append next vector" does
/// not. Keep it positional.

//
// Serde-derived so the on-disk encoding lives in one place (`bincode`) instead
// of a hand-written byte layout. The enum's variant order is part of the wire
// format: bincode tags variants by their *positional index*, so APPEND new
// variants, never reorder or remove existing ones, or old logs stop decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Record {
    Insert {
        collection: u32,
        ordinal: u64,
        vector: Vec<f32>,
    },
    Delete {
        collection: u32,
        ordinal: u64,
    }
}

impl Record {
    /// Serialize the logical mutation into `buf` (bincode, little-endian).
    ///
    /// Appends to `buf` rather than returning a fresh `Vec` so the commit loop
    /// can reuse one scratch buffer across the whole batch. Serializing our
    /// plain data into an in-memory buffer has no fallible step (no IO, no size
    /// cap), so a failure here is a logic bug, not a runtime condition.
    fn encode(&self, buf: &mut Vec<u8>) {
        bincode::serialize_into(buf, self)
            .expect("Record serialization into an in-memory buffer cannot fail");
    }

    /// Inverse of `encode`. A malformed/garbage payload is a recoverable
    /// condition (corrupt frame), so it comes back as `InvalidData` rather than
    /// panicking — recovery turns that into "stop at the torn tail".
    fn decode(bytes: &[u8]) -> io::Result<Record> {
        bincode::deserialize(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
}

pub trait Apply {
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()>;
}

// ---------------------------------------------------------------------------
// On-disk framing
// ---------------------------------------------------------------------------
//
// Each frame:
//   [ len: u32 ][ crc32: u32 ][ lsn: u64 ][ payload: len bytes ]
//
// crc32 covers lsn + payload. Recovery walks frames in order and STOPS at the
// first frame that is short (torn tail) or fails its crc. Everything before the
// bad frame is the valid log; the torn tail was mid-fsync, never acked, and is
// discarded. This is non-negotiable — every crash-during-fsync leaves a torn
// tail, and without the checksum recovery would try to replay garbage.

const LEN_BYTES: usize = 4;
const CRC_BYTES: usize = 4;
const LSN_BYTES: usize = 8;
const HEADER_BYTES: usize = LEN_BYTES + CRC_BYTES + LSN_BYTES;

/// First LSN ever assigned. LSNs are 1-based so 0 can serve as the "nothing
/// durable" watermark sentinel (see `recover`).
const FIRST_LSN: u64 = 1;

fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn write_frame(out: &mut impl Write, lsn: Lsn, payload: &[u8]) -> io::Result<()> {
    let len = payload.len() as u32;

    // crc covers lsn || payload
    let mut crc_input = Vec::with_capacity(LSN_BYTES + payload.len());
    crc_input.extend_from_slice(&lsn.0.to_le_bytes());
    crc_input.extend_from_slice(payload);
    let crc = crc32(&crc_input);

    out.write_all(&len.to_le_bytes())?;
    out.write_all(&crc.to_le_bytes())?;
    out.write_all(&lsn.0.to_le_bytes())?;
    out.write_all(payload)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Writer-facing handle
// ---------------------------------------------------------------------------

/// A message to the WAL thread. Appends and truncations share one channel so
/// the thread that owns the file is the only one that ever touches it, and so
/// ordering between the two is well-defined (a truncate is processed only
/// between append batches, never interleaved with one).
///
/// (std mpsc's reply channels are used as one-shots; swap for crossbeam/tokio
/// oneshot later if the per-call alloc shows up in a profile.)
enum Command {
    /// Append a record; reply with its durable LSN.
    Append {
        record: Record,
        ack: Sender<io::Result<Lsn>>,
    },
    /// Drop every frame with `lsn <= up_to` from the front of the log. Issued by
    /// the flusher AFTER the corresponding index watermark is durable.
    Truncate {
        up_to: u64,
        ack: Sender<io::Result<()>>,
    },
}

/// Cloneable handle the engine hands to writer threads.
#[derive(Clone)]
pub struct WalHandle {
    tx: Sender<Command>,
}

impl WalHandle {
    /// Append a record and BLOCK until it is durable. Returns the assigned LSN.
    /// The caller acks the client only after this returns Ok — that ordering is
    /// the entire durability contract.
    pub fn append(&self, record: Record) -> io::Result<Lsn> {
        let (ack_tx, ack_rx) = mpsc::channel();
        self.tx
            .send(Command::Append {
                record,
                ack: ack_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal thread gone"))?;
        ack_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal thread dropped ack"))?
    }

    /// Ask the WAL thread to drop all frames with `lsn <= up_to` and BLOCK until
    /// done. Safe only once the index is durable up to `up_to` (the flusher
    /// guarantees this ordering); recovery's LSN-skip makes the truncation
    /// itself purely a space optimization, not a correctness dependency.
    pub fn truncate(&self, up_to: u64) -> io::Result<()> {
        let (ack_tx, ack_rx) = mpsc::channel();
        self.tx
            .send(Command::Truncate {
                up_to,
                ack: ack_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal thread gone"))?;
        ack_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal thread dropped ack"))?
    }

    // TODO: an async variant that returns a future instead of blocking, for the
    // engine's async front door.
}

// ---------------------------------------------------------------------------
// The WAL thread
// ---------------------------------------------------------------------------

pub struct Wal {
    handle: WalHandle,
    join: JoinHandle<()>,
}

impl Wal {
    /// Open (creating if needed) the log at `path`, run recovery by replaying
    /// any surviving tail into `applier`, then spawn the commit thread.
    ///
    /// Recovery runs on the caller's thread BEFORE the commit thread starts, so
    /// the index is caught up before any new write can race it.
    /// `skip_through` is the index's durable checkpoint watermark: recovery
    /// replays only frames with `lsn > skip_through` (everything at or below is
    /// already folded into the index), while still advancing the LSN counter
    /// past every frame it sees. Pass 0 for a fresh/un-checkpointed index.
    pub fn start<A: Apply + Send + 'static>(
        path: impl AsRef<Path>,
        mut applier: A,
        skip_through: u64,
    ) -> io::Result<Wal> {
        let path = path.as_ref().to_path_buf();

        // 1. Recover: replay the surviving tail. Returns the next LSN to assign.
        let next_lsn = recover(&path, &mut applier, skip_through)?;

        // 2. Open the log for appending. We keep the File for fsync; the batched
        //    frames are buffered in memory and written in one shot per batch.
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true) // appends only (implies write); truncation is in-thread
            .open(&path)?;

        let (tx, rx) = mpsc::channel::<Command>();

        let loop_path = path.clone();
        let join = std::thread::Builder::new()
            .name("wal-commit".into())
            .spawn(move || commit_loop(file, loop_path, rx, applier, next_lsn))
            .expect("spawn wal thread");

        Ok(Wal {
            handle: WalHandle { tx },
            join,
        })
    }

    pub fn handle(&self) -> WalHandle {
        self.handle.clone()
    }

    /// Clean shutdown: drop the sender so the loop sees the channel close,
    /// drains what's left, does a final fsync, and exits. Then join.
    pub fn shutdown(self) {
        drop(self.handle); // close the channel
        let _ = self.join.join();
    }
}

/// Max records folded into one fsync. A ceiling so a flood can't starve the
/// first waiter indefinitely. Tune against fsync latency.
const MAX_BATCH: usize = 1024;

/// One queued append awaiting commit: the record and the waiter to ack.
type Pending = (Record, Sender<io::Result<Lsn>>);

fn commit_loop<A: Apply>(
    mut file: File,
    path: PathBuf,
    rx: Receiver<Command>,
    mut applier: A,
    mut next_lsn: u64,
) {
    let mut batch: Vec<Pending> = Vec::with_capacity(MAX_BATCH);
    let mut frame_buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut payload: Vec<u8> = Vec::with_capacity(4 * 1024);

    loop {
        // Block for at least one command. recv() erroring == all senders dropped
        // == shutdown. There's nothing queued at that point, so just exit.
        let first = match rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break,
        };

        // A truncate is processed only between append batches, never folded into
        // one. If the first command is a truncate, save it; otherwise start a
        // batch and keep draining until we hit a truncate, the cap, or empty.
        let mut pending_truncate: Option<(u64, Sender<io::Result<()>>)> = None;
        match first {
            Command::Append { record, ack } => batch.push((record, ack)),
            Command::Truncate { up_to, ack } => pending_truncate = Some((up_to, ack)),
        }
        if pending_truncate.is_none() {
            while batch.len() < MAX_BATCH {
                match rx.try_recv() {
                    Ok(Command::Append { record, ack }) => batch.push((record, ack)),
                    Ok(Command::Truncate { up_to, ack }) => {
                        pending_truncate = Some((up_to, ack));
                        break;
                    }
                    Err(_) => break, // empty or disconnected; flush what we have
                }
            }
        }

        // Commit the batch (group fsync) first, so any truncate that follows
        // only ever runs against an already-durable log.
        if !batch.is_empty() {
            commit_batch(
                &mut file,
                &mut applier,
                &mut next_lsn,
                &mut batch,
                &mut frame_buf,
                &mut payload,
            );
        }

        if let Some((up_to, ack)) = pending_truncate {
            // Recovery's LSN-skip already makes truncation a no-op for
            // correctness, so a failure here is not durability-critical — just
            // report it; the next checkpoint retries.
            let _ = ack.send(truncate_wal(&mut file, &path, up_to));
        }
    }

    // Shutdown drain already handled by the loop (recv returns Err only when
    // empty AND disconnected). Final state is durable because every batch
    // fsynced before acking. Nothing buffered here.
}

/// Assign LSNs, write every frame, ONE fsync for the whole batch (group
/// commit), then apply post-fsync and ack each waiter. Drains `batch`.
fn commit_batch<A: Apply>(
    file: &mut File,
    applier: &mut A,
    next_lsn: &mut u64,
    batch: &mut Vec<Pending>,
    frame_buf: &mut Vec<u8>,
    payload: &mut Vec<u8>,
) {
    frame_buf.clear();
    let mut assigned: Vec<Lsn> = Vec::with_capacity(batch.len());
    let batch_start_lsn = *next_lsn;
    let mut frame_err: Option<io::Error> = None;
    for (record, _ack) in batch.iter() {
        let lsn = Lsn(*next_lsn);
        *next_lsn += 1;
        assigned.push(lsn);

        payload.clear();
        record.encode(payload);
        if let Err(e) = write_frame(frame_buf, lsn, payload) {
            frame_err = Some(e);
            break;
        }
    }
    if let Some(e) = frame_err {
        // Nothing hit the file; reclaim the LSNs (keeps the log gap-free) and
        // fail the whole batch loudly rather than silently dropping writes.
        *next_lsn = batch_start_lsn;
        fail_batch(batch, &e);
        return;
    }

    // --- one write + ONE fsync for the whole batch (group commit) ---
    let durable: io::Result<()> = (|| {
        file.write_all(frame_buf)?;
        // fdatasync: we need the data + the file-size growth, not mtime.
        // NOTE: on macOS sync_data/fsync does NOT flush the drive cache; use
        // F_FULLFSYNC there if you want real power-loss durability.
        file.sync_data()?;
        Ok(())
    })();
    if let Err(e) = durable {
        // Not durable -> these writes did NOT commit. Ack every waiter with the
        // error so callers do NOT ack their clients.
        fail_batch(batch, &e);
        return;
    }

    // --- COMMIT POINT crossed. Now apply (post-fsync) then ack. ---
    for ((record, ack), lsn) in batch.drain(..).zip(assigned) {
        // Apply is idempotent; an error here is a bug in apply, not a durability
        // failure — the record IS committed and will replay on restart. Surface
        // it to the waiter but keep going; the data is safe.
        let reply = applier.apply(lsn, &record).map(|()| lsn);
        let _ = ack.send(reply); // waiter gone == nobody to tell; fine
    }
}

fn fail_batch(batch: &mut Vec<Pending>, err: &io::Error) {
    for (_record, ack) in batch.drain(..) {
        let _ = ack.send(Err(io::Error::new(err.kind(), err.to_string())));
    }
}

/// Rewrite the log keeping only frames with `lsn > up_to`. Copies survivors to
/// a temp file, fsyncs, atomically renames over the log, and reopens the append
/// handle. Runs on the WAL thread (the sole file owner) between batches.
///
/// A torn tail (short final frame) is dropped during the copy — harmless, since
/// such a frame was never acked. Frames are copied verbatim, preserving LSNs.
fn truncate_wal(file: &mut File, path: &Path, up_to: u64) -> io::Result<()> {
    let mut kept: Vec<u8> = Vec::new();
    {
        let mut reader = BufReader::new(File::open(path)?);
        loop {
            let mut header = [0u8; HEADER_BYTES];
            match read_exact_or_eof(&mut reader, &mut header)? {
                ReadOutcome::Full => {}
                _ => break, // clean EOF or torn tail: stop
            }
            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header[8..16].try_into().unwrap());

            let mut payload = vec![0u8; len];
            match read_exact_or_eof(&mut reader, &mut payload)? {
                ReadOutcome::Full => {}
                _ => break, // torn payload at tail
            }
            if lsn > up_to {
                kept.extend_from_slice(&header);
                kept.extend_from_slice(&payload);
            }
        }
    }

    let tmp = path.with_extension("log.tmp");
    {
        let mut tf = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        tf.write_all(&kept)?;
        tf.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;

    // The old fd points at the now-unlinked inode; reopen on the new file.
    *file = OpenOptions::new()
        .read(true)
        .create(true)
        .append(true)
        .open(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

/// Replay the surviving WAL tail into `applier`, in LSN order, stopping at the
/// first torn/corrupt frame. Returns the next LSN to assign (max seen + 1, or 0
/// for an empty/absent log).
///
/// Frames with `lsn <= skip_through` are already folded into the index's
/// durable state (that's exactly what the checkpoint watermark means), so they
/// are NOT re-applied — but they still advance the LSN counter, since the log
/// may not have been truncated up to the watermark yet. Idempotent apply makes
/// over-replay harmless; skipping is just cheaper and avoids redundant work.
fn recover<A: Apply>(path: &Path, applier: &mut A, skip_through: u64) -> io::Result<u64> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(FIRST_LSN), // fresh db
        Err(e) => return Err(e),
    };
    let mut reader = BufReader::new(file);
    // LSNs are 1-based: LSN 0 is never assigned, so a durable watermark of 0
    // unambiguously means "nothing checkpointed yet" and the recovery skip
    // (`lsn <= skip_through`) never swallows a real record.
    let mut next_lsn = FIRST_LSN;

    loop {
        // Read the fixed header. A short read here == clean EOF (no partial
        // header) => end of valid log.
        let mut header = [0u8; HEADER_BYTES];
        match read_exact_or_eof(&mut reader, &mut header)? {
            ReadOutcome::Eof => break,
            ReadOutcome::Torn => break, // partial header = torn tail, stop
            ReadOutcome::Full => {}
        }

        let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let lsn = u64::from_le_bytes(header[8..16].try_into().unwrap());

        // Read the payload. Short read = torn tail, stop.
        let mut payload = vec![0u8; len];
        match read_exact_or_eof(&mut reader, &mut payload)? {
            ReadOutcome::Full => {}
            _ => break, // torn payload at tail
        }

        // Verify checksum. A mismatch is a torn/corrupt frame — treat as the
        // end of the valid log (same as a torn tail) and stop.
        let mut crc_input = Vec::with_capacity(LSN_BYTES + len);
        crc_input.extend_from_slice(&lsn.to_le_bytes());
        crc_input.extend_from_slice(&payload);
        if crc32(&crc_input) != crc {
            break;
        }

        // Advance the counter past every valid frame we see, even skipped ones,
        // so LSNs never get reused after a partially-truncated log.
        next_lsn = lsn + 1;

        // Already durable in the index — don't re-apply.
        if lsn <= skip_through {
            continue;
        }

        let record = Record::decode(&payload)?;
        applier.apply(Lsn(lsn), &record)?; // same apply path as live writes
    }

    Ok(next_lsn)
}

enum ReadOutcome {
    Full,
    Torn, // some bytes but not all — partial frame at the tail
    Eof,  // zero bytes — clean end
}

fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8]) -> io::Result<ReadOutcome> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return Ok(if filled == 0 {
                    ReadOutcome::Eof
                } else {
                    ReadOutcome::Torn
                });
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Full)
}

// ---------------------------------------------------------------------------
// Checkpoint / truncation — driven by the flusher thread, not here.
// ---------------------------------------------------------------------------
//
// The flusher (separate thread, every X seconds) does:
//   1. msync both index mmaps  -> index state on disk is now durable up to LSN C
//   2. record checkpoint LSN C
//   3. truncate the WAL up to C
//
// Ordering matters: msync the indexes BEFORE truncating the WAL. If you
// truncate first and crash before the msync lands, you've thrown away records
// the indexes hadn't durably absorbed -> data loss. msync, THEN truncate.
//
// TODO: expose a `Checkpoint` channel/handle so the flusher can tell the WAL
// thread "safe to truncate up to C" and the WAL thread performs the truncation
// itself (it owns the file). Don't truncate the file from another thread.
pub struct Checkpoint {
    pub lsn: Lsn,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn insert(collection: u32, ordinal: u64, vector: &[f32]) -> Record {
        Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
        }
    }

    /// Collects every applied `(lsn, record)` into a shared vec so a test can
    /// inspect what the WAL thread / recovery actually replayed.
    #[derive(Clone)]
    struct CollectingApplier {
        log: Arc<Mutex<Vec<(u64, Record)>>>,
    }

    impl CollectingApplier {
        fn new() -> Self {
            CollectingApplier {
                log: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn snapshot(&self) -> Vec<(u64, Record)> {
            self.log.lock().expect("lock not poisoned").clone()
        }
    }

    impl Apply for CollectingApplier {
        fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
            self.log
                .lock()
                .expect("lock not poisoned")
                .push((lsn.0, record.clone()));
            Ok(())
        }
    }

    #[test]
    fn encode_decode_round_trips_each_variant() {
        for record in [
            insert(0, 0, &[]),
            insert(7, 42, &[1.0, -2.5, 3.25, f32::MIN, f32::MAX]),
            Record::Delete {
                collection: 3,
                ordinal: 99,
            },
        ] {
            let mut buf = Vec::new();
            record.encode(&mut buf);
            let decoded = Record::decode(&buf).expect("decode clean payload");
            assert_eq!(record, decoded);
        }
    }

    #[test]
    fn decode_rejects_garbage() {
        // A bogus enum discriminant must surface as InvalidData, not panic.
        let err = Record::decode(&[0xFF; 4]).expect_err("garbage must not decode");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn appends_are_durable_and_recover_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let records = [
            insert(0, 0, &[1.0, 2.0]),
            insert(0, 1, &[3.0, 4.0]),
            Record::Delete {
                collection: 0,
                ordinal: 0,
            },
        ];

        // First session: append, observing the LSNs handed back, then shut down
        // cleanly (drains + final fsync + join).
        {
            let live = CollectingApplier::new();
            let wal = Wal::start(&path, live.clone(), 0).unwrap();
            let handle = wal.handle();
            for (i, rec) in records.iter().enumerate() {
                let lsn = handle.append(rec.clone()).unwrap();
                assert_eq!(lsn.0, (i + 1) as u64, "LSNs are 1-based, monotonic, gap-free");
            }
            // Drop our handle clone BEFORE shutdown: the commit thread only
            // exits once every sender is gone, so a lingering clone would hang
            // the join. (This is a genuine sharp edge of the handle API.)
            drop(handle);
            wal.shutdown();

            // The live applier saw every record, post-fsync, in order.
            let seen = live.snapshot();
            assert_eq!(seen.len(), records.len());
            for (i, (lsn, rec)) in seen.iter().enumerate() {
                assert_eq!(*lsn, (i + 1) as u64);
                assert_eq!(rec, &records[i]);
            }
        }

        // Second session: a fresh applier must be brought fully up to date by
        // recovery replaying the surviving log before any new write is possible.
        {
            let recovered = CollectingApplier::new();
            let wal = Wal::start(&path, recovered.clone(), 0).unwrap();

            let replayed = recovered.snapshot();
            assert_eq!(replayed.len(), records.len());
            for (i, (lsn, rec)) in replayed.iter().enumerate() {
                assert_eq!(*lsn, (i + 1) as u64);
                assert_eq!(rec, &records[i]);
            }

            // Recovery resumes the counter past the highest LSN seen.
            let next = wal.handle().append(insert(0, 2, &[5.0, 6.0])).unwrap();
            assert_eq!(next.0, records.len() as u64 + 1);
            wal.shutdown();
        }
    }

    #[test]
    fn recover_stops_at_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        {
            let wal = Wal::start(&path, CollectingApplier::new(), 0).unwrap();
            let h = wal.handle();
            h.append(insert(0, 0, &[1.0])).unwrap();
            h.append(insert(0, 1, &[2.0])).unwrap();
            drop(h); // release the clone so shutdown's join can complete
            wal.shutdown();
        }

        // Simulate a crash mid-fsync: append a partial frame to the tail. It has
        // a length header promising bytes that aren't there.
        {
            use std::io::Write as _;
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&999u32.to_le_bytes()).unwrap(); // len says 999
            f.write_all(&0u32.to_le_bytes()).unwrap(); // crc
            f.write_all(&7u64.to_le_bytes()).unwrap(); // lsn
            f.write_all(b"partial").unwrap(); // far fewer than 999 bytes
            f.sync_all().unwrap();
        }

        // Recovery must replay the two good frames (LSN 1, 2) and silently
        // discard the torn tail — no error, next LSN is 3.
        let recovered = CollectingApplier::new();
        let wal = Wal::start(&path, recovered.clone(), 0).unwrap();
        assert_eq!(recovered.snapshot().len(), 2);
        let next = wal.handle().append(insert(0, 2, &[3.0])).unwrap();
        assert_eq!(next.0, 3);
        wal.shutdown();
    }
}


