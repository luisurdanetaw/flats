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
//!   Because fsync comes first, an apply failure lands on an already-committed
//!   record — durable, but absent from the index, with no way to say so in a
//!   watermark. The log halts there rather than carrying on: see `Poison`.
//!
//!   fsync strictly before apply keeps the WAL >= the index at all times, which
//!   is what makes recovery "replay the tail onto the index" correct. ack
//!   strictly after fsync is what makes durability honest: we only promise a
//!   write survived once it actually did.
//!
//! Dependency direction: this module knows NOTHING about the vector or metadata
//! index. The engine injects an `Apply` impl. WAL depends only on the plain
//! data types in `metadata::common` (`Record::Insert` carries the metadata
//! row); the stores themselves are never imported here.


use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::metadata::common::{CollectionConfig, ColumnId, Value};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------


/// Monotonic log sequence number. Assigned by the WAL thread, recorded in the
/// frame, and returned to the writer on ack. Recovery resumes the counter from
/// the highest LSN found in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lsn(pub u64);

/// A `RoaringBitmap` that can ride inside a `Record`.
///
/// `roaring` ships no serde impls (and we do not enable the feature — its
/// encoding is a serde structural one, fatter than the format below), so the
/// bridge is hand-written and delegates to roaring's own PORTABLE
/// serialization — byte-for-byte the format `metadata.snap` already stores.
///
/// Why that matters here and not just for tidiness: a bulk `DELETE` retires a
/// long contiguous run of ordinals. A structural encoding would spell out every
/// ordinal, and this bitmap has to fit in one frame, one CRC, one fsync.
///
/// `RoaringBitmap::optimize` is NOT called here. `Serialize` takes `&self`, so doing it on
/// the encode path would mean cloning the bitmap every time; the logging side
/// calls it once when it builds the retire set, where the run structure is
/// known. Un-optimized is already ~16 KiB per 100k contiguous ordinals (bitmap
/// containers) against ~400 KiB structural, so this is a tuning lever, not a
/// correctness one.
#[derive(Debug, Clone, PartialEq)]
pub struct Bitmap(pub RoaringBitmap);

impl Serialize for Bitmap {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(self.0.serialized_size());
        // Writing into a Vec cannot fail; roaring's signature is io-shaped.
        self.0
            .serialize_into(&mut bytes)
            .map_err(serde::ser::Error::custom)?;
        s.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Bitmap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        // A corrupt/hostile payload reaches this from the recovery path, so a
        // malformed bitmap must be an error, never a panic — the caller turns
        // it into "stop at the torn tail".
        let bytes = <Vec<u8>>::deserialize(d)?;
        RoaringBitmap::deserialize_from(&bytes[..])
            .map(Bitmap)
            .map_err(serde::de::Error::custom)
    }
}

/// One row installed by a `Record::Mutate`: the copy-on-write replacement an
/// UPDATE writes, or a plain new row. Same payload as `Record::Insert` minus
/// the collection, which the enclosing record carries once for the whole
/// statement.
///
/// `ordinal` is assigned on the LOGGING side, like every other ordinal here —
/// see the `Record` doc. Replay must place the row where the original run put
/// it, not wherever the count happens to be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Install {
    pub ordinal: u64,
    pub vector: Vec<f32>,
    /// == `metadata::common::Row`, already validated against the schema.
    pub metadata: Vec<(ColumnId, Value)>,
}

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
        /// The metadata row, exactly as validated against the schema on the
        /// logging side. Applied to MetadataIndex (insert_row) and TupleStore
        /// (write_row). == `metadata::common::Row`.
        metadata: Vec<(ColumnId, Value)>,
    },
    Delete {
        collection: u32,
        ordinal: u64,
    },
    /// DDL is a mutation too (Phase 6): CREATE COLLECTION rides the same
    /// durable path as inserts. The record carries the FULL config (id
    /// assigned on the logging side, like ordinals) so a dumb replay loop can
    /// materialize the collection with no catalog to consult. Apply is
    /// idempotent: if the id/name already exists, it's a no-op.
    CreateCollection { config: CollectionConfig },
    /// ONE STATEMENT, ONE RECORD (Phase 7): everything a `DELETE` or `UPDATE`
    /// changes, in a single frame with a single CRC behind a single fsync.
    ///
    /// This variant exists because the row-at-a-time records above cannot
    /// express a multi-row statement atomically. `DELETE ... WHERE p` matching
    /// N rows would be N frames and N commit points, so a crash halfway leaves
    /// half a statement applied. UPDATE is worse: it is copy-on-write, so the
    /// retire of the old ordinal and the install of the new one MUST land
    /// together — a crash between them either loses the row (retired, never
    /// installed) or duplicates it (installed, never retired).
    ///
    /// `retire` is a bitmap because that is the shape the predicate resolves to
    /// on the logging side, and because a contiguous run of tens of thousands
    /// of ordinals costs tens of bytes in it.
    ///
    /// APPLY IS ONE PUBLISH, not a loop of publishes: see `Apply` and the
    /// engine's `apply_mutate`. Installed payload is capped (see
    /// `Error::StatementTooLarge`) — a record must fit in memory on the commit
    /// thread and on every future replay.
    Mutate {
        collection: u32,
        retire: Bitmap,
        install: Vec<Install>,
    },
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

    /// Make all applied state durable and return the LSN through which the log
    /// may now be truncated (the minimum durable watermark), or `None` if there
    /// is nothing to truncate.
    ///
    /// This runs ON THE COMMIT THREAD, the same thread as `apply`. That is the
    /// whole point under the SWMR index: the index's single `Writer` lives in
    /// the applier, so checkpointing (which mutates the index header) must
    /// happen here, not on a separate flusher thread that would need a second
    /// mutator. The commit thread truncates the WAL up to the returned LSN.
    fn checkpoint(&mut self) -> io::Result<Option<u64>>;
}

// ---------------------------------------------------------------------------
// On-disk framing
// ---------------------------------------------------------------------------
//
// The file opens with a fixed 8-byte header:
//   [ magic: b"FWAL" ][ version: u32 LE ]
//
// Frames carry no version of their own (bincode tags positionally), so the
// header is the ONLY thing that lets recovery refuse an incompatible log
// cleanly instead of misdecoding it as a torn tail. Version 2 = the
// metadata-carrying `Record::Insert` (Phase 4c). Version 1 logs predate the
// header entirely and surface as "bad magic": the upgrade path is a clean
// shutdown on the old build (its final checkpoint folds everything into the
// indexes and truncates the WAL), then start the new build.
//
// Each frame after the header:
//   [ len: u32 ][ crc32: u32 ][ lsn: u64 ][ payload: len bytes ]
//
// crc32 covers lsn + payload. Recovery walks frames in order and STOPS at the
// first frame that is short (torn tail) or fails its crc. Everything before the
// bad frame is the valid log; the torn tail was mid-fsync, never acked, and is
// discarded. This is non-negotiable — every crash-during-fsync leaves a torn
// tail, and without the checksum recovery would try to replay garbage.

const WAL_MAGIC: &[u8; 4] = b"FWAL";
/// Version written to new logs. 2 = metadata-carrying Insert (Phase 4c);
/// 3 = + CreateCollection (Phase 6); 4 = + Mutate (Phase 7).
const WAL_VERSION: u32 = 4;
/// Oldest version this build still reads. Every bump so far only APPENDED a
/// variant to the Record enum, so bincode's positional tags for the older
/// variants are unchanged and an old log is a strict subset of a new one —
/// reading it is safe. The reverse is not: an old build meeting a variant it
/// has no tag for would misdecode, which is the whole reason the written
/// version bumps.
const WAL_MIN_VERSION: u32 = 2;
const WAL_HEADER_BYTES: usize = 8;

/// Ceiling on one frame's payload, enforced when READING.
///
/// The length field is read straight off disk and used to size a buffer, so
/// without a bound a corrupt byte becomes a 4 GiB allocation on the recovery
/// path. Nothing legal comes close: installed payload is capped at
/// `MAX_INSTALL_BYTES` (256 MiB) on the logging side, and the rest of a record
/// — the retire bitmap, framing, bincode overhead — is orders of magnitude
/// smaller. A frame claiming more than this cannot have been written by any
/// build of this code, so recovery treats it as the torn tail it is.
const MAX_FRAME_BYTES: usize = 384 * 1024 * 1024;

fn wal_header() -> [u8; WAL_HEADER_BYTES] {
    let mut h = [0u8; WAL_HEADER_BYTES];
    h[0..4].copy_from_slice(WAL_MAGIC);
    h[4..8].copy_from_slice(&WAL_VERSION.to_le_bytes());
    h
}

const LEN_BYTES: usize = 4;
const CRC_BYTES: usize = 4;
const LSN_BYTES: usize = 8;
const HEADER_BYTES: usize = LEN_BYTES + CRC_BYTES + LSN_BYTES;

// LSNs are 1-based: LSN 0 is never assigned, so a durable watermark of 0 means
// "nothing checkpointed yet" and the recovery skip never swallows a real record
// (see `recover`, where the counter resumes at `skip_through + 1`).

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

/// A message to the WAL thread. Everything that touches the log file flows
/// through this one channel, so the commit thread is the sole file owner and
/// control ops are well-ordered against append batches (processed only between
/// batches, never interleaved with one).
///
/// (std mpsc's reply channels are used as one-shots; swap for crossbeam/tokio
/// oneshot later if the per-call alloc shows up in a profile.)
enum Command {
    /// Append a record; reply with its durable LSN.
    Append {
        record: Record,
        ack: Sender<io::Result<Lsn>>,
    },
    /// Drop every frame with `lsn <= up_to` from the front of the log. Used by
    /// callers that manage durability themselves (e.g. tests).
    Truncate {
        up_to: u64,
        ack: Sender<io::Result<()>>,
    },
    /// Run `Apply::checkpoint` on the commit thread (making the index durable),
    /// then truncate the log up to the watermark it returns. This keeps all
    /// index mutation on the single writer thread.
    Checkpoint {
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
    /// done. Safe only once the index is durable up to `up_to`; recovery's
    /// LSN-skip makes the truncation itself purely a space optimization, not a
    /// correctness dependency.
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

    /// Run a checkpoint on the commit thread (`Apply::checkpoint` + truncate) and
    /// BLOCK until done. This is how the flusher and `Db::checkpoint` make the
    /// index durable, since the index's sole `Writer` lives on that thread.
    pub fn checkpoint(&self) -> io::Result<()> {
        let (ack_tx, ack_rx) = mpsc::channel();
        self.tx
            .send(Command::Checkpoint { ack: ack_tx })
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

/// Why the log stopped accepting writes.
///
/// Set exactly once, by the first `Apply::apply` that fails AFTER its record
/// was fsynced. That record is committed — it will replay on the next `open` —
/// but the index does not reflect it, and there is no way to express "applied
/// through 7 except 5": a watermark is a single high-water number. So the only
/// safe move is to stop:
///
///   * no further writes — they would apply on top of the hole, and their
///     watermark would claim the record that never landed;
///   * no checkpoint and no truncation — either one sheds the very frame that
///     recovery needs, turning a recoverable failure into silent data loss.
///
/// Reads keep working: the index is stale, not inconsistent. Reopening the
/// database replays the tail and clears the condition (apply is idempotent).
/// This is the same posture as Postgres' redo `PANIC` — the log is the truth,
/// the index is a derived cache, and you re-derive rather than repair in place.
#[derive(Clone, Debug)]
pub struct Poison {
    /// The durable record the index never absorbed.
    pub lsn: Lsn,
    /// The apply error, rendered where it happened.
    pub cause: String,
}

impl Poison {
    fn to_io(&self) -> io::Error {
        io::Error::other(format!(
            "wal poisoned at lsn {}: {} (the record is durable but unapplied; \
             reopen the database to replay it)",
            self.lsn.0, self.cause
        ))
    }
}

/// The error to hand a waiter that was refused because the log is poisoned.
fn poison_io(cell: &OnceLock<Poison>) -> io::Error {
    match cell.get() {
        Some(p) => p.to_io(),
        // Unreachable: every caller checks `get().is_some()` first. Degrade to
        // a truthful message rather than panicking on the commit thread.
        None => io::Error::other("wal poisoned"),
    }
}

pub struct Wal {
    handle: WalHandle,
    join: JoinHandle<()>,
    /// Set once by the commit thread when a post-fsync apply fails; read by the
    /// engine to turn a refused write into a typed error. See `Poison`.
    poison: Arc<OnceLock<Poison>>,
    /// Test-only fault-injection point: when set, the next commit batch fails as
    /// if its fsync errored, exercising callers' durability-failure paths. The
    /// flag is shared with the commit thread; it is never consulted in non-test
    /// builds (the check in `commit_batch` is `#[cfg(test)]`).
    #[allow(dead_code)]
    fail_next: Arc<AtomicBool>,
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
        let mut file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true) // appends only (implies write); truncation is in-thread
            .open(&path)?;

        // Fresh (or crash-emptied) file: stamp the header before any frame.
        // Recovery already validated it on non-empty files.
        if file.metadata()?.len() == 0 {
            file.write_all(&wal_header())?;
            file.sync_data()?;
        }

        let (tx, rx) = mpsc::channel::<Command>();

        let fail_next = Arc::new(AtomicBool::new(false));
        let loop_fail = fail_next.clone();
        let poison: Arc<OnceLock<Poison>> = Arc::new(OnceLock::new());
        let loop_poison = poison.clone();
        let loop_path = path.clone();
        let join = std::thread::Builder::new()
            .name("wal-commit".into())
            .spawn(move || {
                commit_loop(
                    file,
                    loop_path,
                    rx,
                    applier,
                    next_lsn,
                    loop_fail,
                    loop_poison,
                )
            })
            .expect("spawn wal thread");

        Ok(Wal {
            handle: WalHandle { tx },
            join,
            poison,
            fail_next,
        })
    }

    /// Why the log stopped accepting writes, if it has. `None` is the healthy
    /// state. Once `Some`, it stays `Some` for the life of this `Wal`.
    pub fn poison(&self) -> Option<Poison> {
        self.poison.get().cloned()
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

    /// Test-only: make the next committed batch fail as though its fsync errored.
    /// Consumed by that batch (one-shot).
    #[cfg(test)]
    pub fn fail_next_append(&self) {
        self.fail_next.store(true, Ordering::SeqCst);
    }
}

/// Max records folded into one fsync. A ceiling so a flood can't starve the
/// first waiter indefinitely. Tune against fsync latency.
const MAX_BATCH: usize = 1024;

/// One queued append awaiting commit: the record and the waiter to ack.
type Pending = (Record, Sender<io::Result<Lsn>>);

/// A control op pulled off the command stream, deferred until the current append
/// batch has been committed.
enum Control {
    Truncate {
        up_to: u64,
        ack: Sender<io::Result<()>>,
    },
    Checkpoint {
        ack: Sender<io::Result<()>>,
    },
}

impl Control {
    /// Both variants ack the same way; a poisoned log refuses either without
    /// caring which it was.
    fn into_ack(self) -> Sender<io::Result<()>> {
        match self {
            Control::Truncate { ack, .. } => ack,
            Control::Checkpoint { ack } => ack,
        }
    }
}

fn commit_loop<A: Apply>(
    mut file: File,
    path: PathBuf,
    rx: Receiver<Command>,
    mut applier: A,
    mut next_lsn: u64,
    fail_next: Arc<AtomicBool>,
    poison: Arc<OnceLock<Poison>>,
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

        // A control op (truncate / checkpoint) is processed only between append
        // batches, never folded into one. If the first command is a control op,
        // save it; otherwise start a batch and keep draining until we hit a
        // control op, the cap, or empty.
        let mut pending_control: Option<Control> = None;
        match first {
            Command::Append { record, ack } => batch.push((record, ack)),
            Command::Truncate { up_to, ack } => {
                pending_control = Some(Control::Truncate { up_to, ack })
            }
            Command::Checkpoint { ack } => pending_control = Some(Control::Checkpoint { ack }),
        }
        if pending_control.is_none() {
            while batch.len() < MAX_BATCH {
                match rx.try_recv() {
                    Ok(Command::Append { record, ack }) => batch.push((record, ack)),
                    Ok(Command::Truncate { up_to, ack }) => {
                        pending_control = Some(Control::Truncate { up_to, ack });
                        break;
                    }
                    Ok(Command::Checkpoint { ack }) => {
                        pending_control = Some(Control::Checkpoint { ack });
                        break;
                    }
                    Err(_) => break, // empty or disconnected; flush what we have
                }
            }
        }

        // Commit the batch (group fsync) first, so any control op that follows
        // runs against a log that is durable and — unless the batch itself
        // poisoned us — fully applied.
        // Test fault point: pretend this batch's fsync failed. Decided before
        // any LSN/IO work so it models "the durable write never happened", and
        // only consumed by a batch that would actually have committed.
        // Compiled out of non-test builds entirely.
        #[cfg(test)]
        let injected_failure = !batch.is_empty() && fail_next.swap(false, Ordering::SeqCst);
        #[cfg(not(test))]
        let injected_failure = {
            let _ = &fail_next; // unused in non-test builds
            false
        };

        if poison.get().is_some() {
            // Closed for business. Refuse without writing a frame or burning an
            // LSN: anything appended past the hole is a record the index can
            // never absorb in order.
            fail_batch(&mut batch, &poison_io(&poison));
        } else if injected_failure {
            fail_batch(
                &mut batch,
                &io::Error::other("injected WAL failure (test fault point)"),
            );
        } else if !batch.is_empty() {
            commit_batch(
                &mut file,
                &mut applier,
                &mut next_lsn,
                &mut batch,
                &mut frame_buf,
                &mut payload,
                &poison,
            );
        }

        match pending_control {
            // Checked AFTER the batch, because the batch may have poisoned us
            // just now. Both control ops shed log the index still needs: a
            // checkpoint would stamp a durable watermark over the hole and
            // truncate the frame away, and truncate does the second half on its
            // own. Refuse both while poisoned.
            Some(control) if poison.get().is_some() => {
                let _ = control.into_ack().send(Err(poison_io(&poison)));
            }
            Some(Control::Truncate { up_to, ack }) => {
                // Recovery's LSN-skip already makes truncation a no-op for
                // correctness, so a failure here is not durability-critical —
                // just report it; the next checkpoint retries.
                let _ = ack.send(truncate_wal(&mut file, &path, up_to));
            }
            Some(Control::Checkpoint { ack }) => {
                // Make the index durable, then shed the now-redundant prefix.
                // Both steps run here on the single writer thread.
                let res = match applier.checkpoint() {
                    Ok(Some(up_to)) => truncate_wal(&mut file, &path, up_to),
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                };
                let _ = ack.send(res);
            }
            None => {}
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
    poison: &OnceLock<Poison>,
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
        // Every record from here on is committed. An apply failure is therefore
        // NOT a durability failure — the record survives and replays — but the
        // index is now missing it, and nothing downstream can represent that
        // gap. So the first failure poisons the log and the rest of this batch
        // is refused unapplied: applying past the hole would let their
        // watermark claim the record that never landed, and the next checkpoint
        // would truncate it away for good.
        if poison.get().is_some() {
            let _ = ack.send(Err(poison_io(poison)));
            continue;
        }
        match applier.apply(lsn, &record) {
            Ok(()) => {
                let _ = ack.send(Ok(lsn)); // waiter gone == nobody to tell; fine
            }
            Err(e) => {
                let _ = poison.set(Poison {
                    lsn,
                    cause: e.to_string(),
                });
                let _ = ack.send(Err(poison_io(poison)));
            }
        }
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
        // Skip the file header (recovery validated it when this file was
        // opened); the temp file below gets a fresh copy.
        let mut file_header = [0u8; WAL_HEADER_BYTES];
        match read_exact_or_eof(&mut reader, &mut file_header)? {
            ReadOutcome::Full => {}
            _ => {} // empty/short file: no frames to keep
        }
        loop {
            let mut header = [0u8; HEADER_BYTES];
            match read_exact_or_eof(&mut reader, &mut header)? {
                ReadOutcome::Full => {}
                _ => break, // clean EOF or torn tail: stop
            }
            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header[8..16].try_into().unwrap());
            if len > MAX_FRAME_BYTES {
                break; // impossible length: torn/corrupt tail, not an allocation
            }

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
        tf.write_all(&wal_header())?; // the rewritten log keeps its header
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
///
/// MULTI-COLLECTION CAUTION: `skip_through` is a single scalar here. With more
/// than one collection sharing this WAL, the caller MUST pass the *minimum*
/// durable watermark across collections (so this never skips a frame a lagging
/// collection still needs), and the real fix is a per-collection watermark keyed
/// on each frame's collection id. Never pass a global `max(last_lsn)` — that
/// silently drops records for collections that haven't caught up. See
/// `engine::Db::open`.
fn recover<A: Apply>(path: &Path, applier: &mut A, skip_through: u64) -> io::Result<u64> {
    // The LSN counter must never resume at or below the checkpoint watermark.
    // A checkpoint can truncate the WAL down to EMPTY (everything <= watermark
    // was folded into the index and dropped), so the surviving frames alone
    // can't tell us how far the counter had advanced — `skip_through` can. If we
    // restarted from 1 here, the next appends would be assigned LSNs <=
    // skip_through and the *following* recovery would skip them as "already
    // durable", silently losing them. So seed the counter past the watermark and
    // only ever ratchet it upward. (skip_through == 0 for a fresh db => 1.)
    let resume_floor = skip_through + 1;

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(resume_floor), // fresh db
        Err(e) => return Err(e),
    };
    let mut reader = BufReader::new(file);

    // File header first. Refusing here must be LOUD: without the check, an
    // incompatible log would misdecode as a torn tail and recovery would
    // silently stop early — worse than an error.
    let mut file_header = [0u8; WAL_HEADER_BYTES];
    match read_exact_or_eof(&mut reader, &mut file_header)? {
        // Zero bytes: created-then-crashed before the header landed. Nothing
        // was ever acked from it; treat as fresh (start() stamps the header).
        ReadOutcome::Eof => return Ok(resume_floor),
        ReadOutcome::Torn => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "wal file shorter than its header and not empty — not a flats v2 WAL",
            ));
        }
        ReadOutcome::Full => {}
    }
    if &file_header[0..4] != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wal has no FWAL header — pre-4c log or foreign file; upgrade path: \
             clean shutdown on the previous build (final checkpoint folds the log \
             into the indexes), then start this build",
        ));
    }
    let version = u32::from_le_bytes(
        file_header[4..8]
            .try_into()
            .expect("slice of fixed len 4"),
    );
    if !(WAL_MIN_VERSION..=WAL_VERSION).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported wal version {version} (this build reads {WAL_MIN_VERSION}..={WAL_VERSION})"
            ),
        ));
    }

    let mut next_lsn = resume_floor;

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

        // A length no writer could have produced is corruption, and sizing a
        // buffer from it would turn one bad byte into a 4 GiB allocation.
        // Stop here exactly as for any other malformed tail.
        if len > MAX_FRAME_BYTES {
            break;
        }

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

        // Ratchet the counter past every valid frame we see, even skipped ones,
        // so LSNs are never reused. `max` (not assignment) keeps the
        // `resume_floor` seed intact when surviving frames sit below it.
        next_lsn = next_lsn.max(lsn + 1);

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
// Checkpoint / truncation — driven on the commit thread.
// ---------------------------------------------------------------------------
//
// A `Command::Checkpoint` (sent by the engine's flusher on a timer, or by
// `Db::checkpoint`) is handled between append batches by the commit loop:
//   1. `Apply::checkpoint` makes the index durable (msync data, then advance the
//      durable header watermark) -> returns the LSN `C` it is now durable up to.
//   2. the commit thread truncates the WAL up to `C`.
//
// Ordering matters: msync the index BEFORE truncating the WAL. If you truncated
// first and crashed before the msync landed, you'd throw away records the index
// hadn't durably absorbed -> data loss. The applier syncs, THEN we truncate.
// Both steps run here on the file-owning thread; no other thread touches the
// file or the index.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    fn test_config() -> CollectionConfig {
        CollectionConfig {
            id: 4,
            name: "docs".into(),
            capacity: 1_000_000,
            schema: crate::metadata::common::Schema::from_columns(vec![
                crate::metadata::common::ColumnSpec::Vector {
                    name: "vector".into(),
                    dim: std::num::NonZeroUsize::new(768).unwrap(),
                },
                crate::metadata::common::ColumnSpec::Scalar {
                    name: "author".into(),
                    ty: crate::metadata::common::ColumnType::Text,
                },
                crate::metadata::common::ColumnSpec::Scalar {
                    name: "year".into(),
                    ty: crate::metadata::common::ColumnType::Int,
                },
            ])
            .unwrap(),
        }
    }

    fn insert(collection: u32, ordinal: u64, vector: &[f32]) -> Record {
        Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
            metadata: vec![],
        }
    }

    /// Collects every applied `(lsn, record)` into a shared vec so a test can
    /// inspect what the WAL thread / recovery actually replayed.
    #[derive(Clone)]
    struct CollectingApplier {
        log: Arc<Mutex<Vec<(u64, Record)>>>,
        /// When set, `apply` fails for this one LSN. The fault fixture for the
        /// apply-error halt: by the time apply runs the record is already
        /// fsynced, so this models the single failure the WAL cannot ack
        /// honestly — durable, but not reflected in the index.
        fail_at: Option<u64>,
        /// How many times `checkpoint` was called. A poisoned log must never
        /// call it: checkpoint advances the durable watermark and truncates.
        checkpoints: Arc<AtomicUsize>,
    }

    impl CollectingApplier {
        fn new() -> Self {
            CollectingApplier {
                log: Arc::new(Mutex::new(Vec::new())),
                fail_at: None,
                checkpoints: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// An applier that fails to apply exactly `lsn`, post-fsync.
        fn failing_at(lsn: u64) -> Self {
            CollectingApplier {
                fail_at: Some(lsn),
                ..CollectingApplier::new()
            }
        }

        fn snapshot(&self) -> Vec<(u64, Record)> {
            self.log.lock().expect("lock not poisoned").clone()
        }

        fn checkpoints(&self) -> usize {
            self.checkpoints.load(Ordering::SeqCst)
        }
    }

    impl Apply for CollectingApplier {
        fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
            if self.fail_at == Some(lsn.0) {
                return Err(io::Error::other(format!("injected apply failure at {}", lsn.0)));
            }
            self.log
                .lock()
                .expect("lock not poisoned")
                .push((lsn.0, record.clone()));
            Ok(())
        }
        fn checkpoint(&mut self) -> io::Result<Option<u64>> {
            // These WAL tests never checkpoint; nothing to make durable. The
            // count is what the poison tests assert on.
            self.checkpoints.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }
    }

    /// Poison the log at `path` by letting LSN `fail_at` fail apply, and hand
    /// back the running WAL plus a handle. Every poison test starts here.
    fn poisoned_wal(path: &Path, fail_at: u64) -> (Wal, WalHandle, CollectingApplier) {
        let applier = CollectingApplier::failing_at(fail_at);
        let wal = Wal::start(path, applier.clone(), 0).unwrap();
        let handle = wal.handle();
        // LSNs are 1-based, so priming `fail_at - 1` records leaves the next
        // append landing on exactly `fail_at`.
        for ordinal in 0..fail_at.saturating_sub(1) {
            handle.append(insert(0, ordinal, &[1.0])).unwrap();
        }
        handle
            .append(insert(0, fail_at, &[2.0]))
            .expect_err("the failing apply must be reported to its waiter");
        (wal, handle, applier)
    }

    #[test]
    fn apply_failure_poisons_the_wal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let (wal, handle, _applier) = poisoned_wal(&path, 2);

        // The record IS durable — the fsync happened before apply ran. What
        // failed is the derived index, so the log must remember which record
        // the index is missing rather than carrying on as if nothing happened.
        let poison = wal.poison().expect("a failed apply poisons the log");
        assert_eq!(poison.lsn, Lsn(2));

        drop(handle);
        wal.shutdown();
    }

    #[test]
    fn writes_after_poison_are_refused_without_touching_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let (wal, handle, _applier) = poisoned_wal(&path, 2);
        let len_at_poison = std::fs::metadata(&path).unwrap().len();

        handle
            .append(insert(0, 3, &[3.0]))
            .expect_err("a poisoned log refuses further writes");

        // Refused means refused: no frame, no LSN burned. Anything written past
        // the hole would be a record the index can never absorb in order.
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            len_at_poison,
            "a refused append must not reach the file"
        );

        drop(handle);
        wal.shutdown();
    }

    #[test]
    fn checkpoint_is_refused_while_poisoned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let (wal, handle, applier) = poisoned_wal(&path, 2);
        let len_at_poison = std::fs::metadata(&path).unwrap().len();

        handle
            .checkpoint()
            .expect_err("a poisoned log refuses to checkpoint");

        // This is the whole point of the halt. Checkpoint persists the durable
        // watermark and then truncates every frame at or below it — with an
        // unapplied record in the tail that turns a recoverable failure into
        // permanent, silent data loss.
        assert_eq!(
            applier.checkpoints(),
            0,
            "the index must not be made durable past a record it never absorbed"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            len_at_poison,
            "the unapplied record must survive in the log for recovery"
        );

        drop(handle);
        wal.shutdown();
    }

    #[test]
    fn truncate_is_refused_while_poisoned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let (wal, handle, _applier) = poisoned_wal(&path, 2);
        let len_at_poison = std::fs::metadata(&path).unwrap().len();

        // Same reasoning as checkpoint, via the other door: an explicit
        // truncate would shed the frame recovery needs.
        handle
            .truncate(u64::MAX)
            .expect_err("a poisoned log refuses to truncate");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len_at_poison);

        drop(handle);
        wal.shutdown();
    }

    /// The bitmap encoding is roaring's own portable format, not a serde
    /// element-by-element fallback — the same format `metadata.snap` already
    /// writes. That is not cosmetic: a bulk DELETE retires a long contiguous
    /// run, and this bitmap has to fit in one frame behind one fsync.
    ///
    /// Two facts pinned here. First, the default encoding is already an order
    /// of magnitude better than per-element: 100k contiguous ordinals span two
    /// 64k blocks, so roaring stores two BITMAP containers — 8 KiB each,
    /// ~16 KiB total, against the ~400 KiB a structural encoding would spend.
    /// Second, `RoaringBitmap::optimize` collapses those runs into RUN containers and
    /// buys another ~20x. It is deliberately NOT called in `Serialize` (that
    /// takes `&self`, so it would mean cloning the bitmap on every encode) —
    /// the logging side calls it when it builds the retire set, which is where
    /// the run structure is known.
    #[test]
    fn bitmap_round_trips_in_the_portable_roaring_encoding() {
        let dense: RoaringBitmap = (0..100_000u32).collect();
        let bitmap = Bitmap(dense.clone());

        let encoded = bincode::serialize(&bitmap).expect("bitmap serializes");
        assert!(
            encoded.len() < 32 * 1024,
            "100k ordinals must stay in bitmap containers, got {} bytes",
            encoded.len()
        );

        // The lever step 4's DELETE path pulls: same set, run containers.
        let mut optimized = dense.clone();
        assert!(optimized.optimize(), "a contiguous run is run-compressible");
        let opt_encoded = bincode::serialize(&Bitmap(optimized)).unwrap();
        assert!(
            opt_encoded.len() < 1024,
            "run containers must collapse a contiguous run, got {} bytes",
            opt_encoded.len()
        );
        assert_eq!(
            bincode::deserialize::<Bitmap>(&opt_encoded).unwrap().0,
            dense,
            "run optimization is an encoding choice, never a change of contents"
        );

        let decoded: Bitmap = bincode::deserialize(&encoded).expect("bitmap deserializes");
        assert_eq!(decoded.0, dense);

        // Sparse and empty are the other two shapes the record carries.
        for set in [RoaringBitmap::new(), (0..50_000u32).map(|i| i * 977).collect()] {
            let round: Bitmap =
                bincode::deserialize(&bincode::serialize(&Bitmap(set.clone())).unwrap()).unwrap();
            assert_eq!(round.0, set);
        }
    }

    /// bincode tags enum variants by POSITION, so the on-disk meaning of a tag
    /// is decided by where a variant sits in the enum. `Mutate` was appended;
    /// this pins that the three older variants kept tags 0/1/2, which is the
    /// entire reason a v2/v3 log still decodes on this build. Inserting a
    /// variant above them would silently reinterpret every existing log.
    #[test]
    fn appending_mutate_keeps_old_variant_tags_stable() {
        let cases: [(Record, u32); 4] = [
            (insert(0, 0, &[1.0]), 0),
            (
                Record::Delete {
                    collection: 0,
                    ordinal: 0,
                },
                1,
            ),
            (
                Record::CreateCollection {
                    config: test_config(),
                },
                2,
            ),
            (
                Record::Mutate {
                    collection: 0,
                    retire: Bitmap(RoaringBitmap::new()),
                    install: vec![],
                },
                3,
            ),
        ];
        for (record, tag) in cases {
            let mut buf = Vec::new();
            record.encode(&mut buf);
            // bincode writes the variant index as a u32 LE prefix.
            assert_eq!(
                u32::from_le_bytes(buf[0..4].try_into().unwrap()),
                tag,
                "variant tag moved for {record:?} — every existing log just changed meaning"
            );
        }
    }

    /// Older logs are a strict SUBSET of the current format: every variant was
    /// APPENDED, so a v2 or v3 writer's frames still decode to the same
    /// variants here. Refusing them would strand every existing database on
    /// upgrade; misdecoding them would be worse.
    #[test]
    fn v2_and_v3_logs_still_recover_on_this_build() {
        let records = [
            insert(0, 0, &[1.0, 2.0]),
            Record::Delete {
                collection: 0,
                ordinal: 0,
            },
        ];

        // Adding a variant means an OLD build must refuse a NEW log rather
        // than misdecode it, and the header version is the only thing that can
        // tell it to (CLAUDE.md §6). Pinned as a literal so appending a variant
        // without bumping fails here rather than in production.
        assert_eq!(
            WAL_VERSION, 4,
            "Record::Mutate was appended; the written version must bump with it"
        );

        // Every version this build claims to accept must actually replay.
        for version in WAL_MIN_VERSION..=WAL_VERSION {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wal.log");

            let mut bytes = Vec::new();
            bytes.extend_from_slice(WAL_MAGIC);
            bytes.extend_from_slice(&version.to_le_bytes());
            for (i, rec) in records.iter().enumerate() {
                let mut payload = Vec::new();
                rec.encode(&mut payload);
                write_frame(&mut bytes, Lsn(i as u64 + 1), &payload).unwrap();
            }
            std::fs::write(&path, &bytes).unwrap();

            let applier = CollectingApplier::new();
            let wal = Wal::start(&path, applier.clone(), 0).unwrap();
            assert_eq!(
                applier.snapshot().len(),
                records.len(),
                "a v{version} log must replay on this build"
            );
            wal.shutdown();
        }

        // ...and the floor still bites: v1 predates the metadata-carrying
        // Insert, so its frames would misdecode rather than fail.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(WAL_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            Wal::start(&path, CollectingApplier::new(), 0).is_err(),
            "a v1 log must be refused, not misdecoded"
        );
    }

    /// A frame's length is read straight off disk, so a corrupt one used to
    /// become `vec![0u8; len]` — an allocation sized by a byte that a crash (or
    /// a hostile file) chose, up to 4 GiB. A frame that cannot legally exist is
    /// a torn tail, which is exactly how every other malformed tail is treated.
    #[test]
    fn recovery_treats_an_absurd_frame_length_as_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&wal_header());
        // One good frame first, to prove recovery keeps everything before the
        // garbage rather than discarding the whole log.
        let mut payload = Vec::new();
        insert(0, 0, &[1.0]).encode(&mut payload);
        write_frame(&mut bytes, Lsn(1), &payload).unwrap();
        // Then a header claiming a preposterous payload.
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // len
        bytes.extend_from_slice(&0u32.to_le_bytes()); // crc
        bytes.extend_from_slice(&2u64.to_le_bytes()); // lsn
        std::fs::write(&path, &bytes).unwrap();

        let applier = CollectingApplier::new();
        let wal = Wal::start(&path, applier.clone(), 0).unwrap();
        let seen = applier.snapshot();
        assert_eq!(seen.len(), 1, "the good frame survives the garbage that follows");
        assert_eq!(seen[0].0, 1);
        wal.shutdown();
    }

    #[test]
    fn mutate_round_trips_through_encode_decode() {
        for record in [
            // The DELETE shape: retire many, install nothing.
            Record::Mutate {
                collection: 2,
                retire: Bitmap((0..10_000u32).collect()),
                install: vec![],
            },
            // The UPDATE shape: retire the old ordinals, install their
            // copy-on-write replacements at fresh ones.
            Record::Mutate {
                collection: 7,
                retire: Bitmap([3u32, 9, 27].into_iter().collect()),
                install: vec![
                    Install {
                        ordinal: 100,
                        vector: vec![1.0, -2.5, f32::MIN],
                        metadata: vec![(0, Value::Int(-7)), (1, Value::Text("héllo".into()))],
                    },
                    Install {
                        ordinal: 101,
                        vector: vec![0.0, 0.0, 0.0],
                        metadata: vec![(0, Value::Float(2.25))],
                    },
                ],
            },
            // Degenerate but legal on the wire: the engine refuses to LOG an
            // empty mutate, but decode must not care.
            Record::Mutate {
                collection: 0,
                retire: Bitmap(RoaringBitmap::new()),
                install: vec![],
            },
        ] {
            let mut buf = Vec::new();
            record.encode(&mut buf);
            let decoded = Record::decode(&buf).expect("decode clean payload");
            assert_eq!(record, decoded);
        }
    }

    #[test]
    fn encode_decode_round_trips_each_variant() {
        for record in [
            insert(0, 0, &[]),
            insert(7, 42, &[1.0, -2.5, 3.25, f32::MIN, f32::MAX]),
            // Metadata-carrying inserts: empty row and a multi-type row.
            Record::Insert {
                collection: 1,
                ordinal: 5,
                vector: vec![0.5],
                metadata: vec![
                    (0, Value::Int(-7)),
                    (1, Value::Float(2.25)),
                    (2, Value::Text("héllo wörld".into())),
                ],
            },
            Record::Delete {
                collection: 3,
                ordinal: 99,
            },
            Record::CreateCollection {
                config: CollectionConfig {
                    id: 4,
                    name: "docs".into(),
                    capacity: 1_000_000,
                    schema: crate::metadata::common::Schema::from_columns(vec![
                        crate::metadata::common::ColumnSpec::Vector {
                            name: "vector".into(),
                            dim: std::num::NonZeroUsize::new(768).unwrap(),
                        },
                        crate::metadata::common::ColumnSpec::Scalar {
                            name: "author".into(),
                            ty: crate::metadata::common::ColumnType::Text,
                        },
                        crate::metadata::common::ColumnSpec::Scalar {
                            name: "year".into(),
                            ty: crate::metadata::common::ColumnType::Int,
                        },
                    ])
                    .unwrap(),
                },
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
    fn header_round_trips_and_survives_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");

        {
            let wal = Wal::start(&path, CollectingApplier::new(), 0).unwrap();
            let h = wal.handle();
            h.append(insert(0, 0, &[1.0])).unwrap();
            h.append(insert(0, 1, &[2.0])).unwrap();
            // Truncate rewrites the whole file — the header must be re-emitted.
            h.truncate(1).unwrap();
            drop(h);
            wal.shutdown();
        }

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], WAL_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            WAL_VERSION
        );

        // And the truncated log still recovers: only LSN 2 survives.
        let recovered = CollectingApplier::new();
        let wal = Wal::start(&path, recovered.clone(), 0).unwrap();
        let seen = recovered.snapshot();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, 2);
        wal.shutdown();
    }

    #[test]
    fn recovery_refuses_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        // A pre-header (or foreign) file: no FWAL magic.
        std::fs::write(&path, b"not a wal at all, definitely long enough").unwrap();
        let err = match Wal::start(&path, CollectingApplier::new(), 0) {
            Err(e) => e,
            Ok(_) => panic!("bad magic must refuse, not misdecode"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("FWAL"), "error names the header: {err}");
    }

    #[test]
    fn recovery_refuses_future_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(WAL_MAGIC);
        bytes.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = match Wal::start(&path, CollectingApplier::new(), 0) {
            Err(e) => e,
            Ok(_) => panic!("future version must refuse cleanly"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("99"), "error names the version: {err}");
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


