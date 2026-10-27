//! Compaction: rebuild one collection densely, off the commit thread, then
//! flip to the rebuilt copy.
//!
//! Retiring a row destroys nothing (see the `Record::Delete` arm), and ordinals
//! are never reused, so without this a collection only ever fills: dead rows
//! keep their vectors, their values, and their slot of `capacity` forever.
//! Compaction copies the LIVE rows into a fresh set of stores at ordinals
//! `0..n` and swaps them in. Nothing about the three invariants moves:
//!
//!   * **One owner of liveness.** The dead set is the complement of the `live`
//!     bitmap. Nothing else is consulted.
//!   * **Append-only `count`.** The new flat index is written by its own
//!     writer from zero; the old one is never rewritten below its `count`.
//!   * **Lock-free reads.** Readers never wait on a compaction. A query holds
//!     `Arc`s to ONE generation for its whole life, and the old generation's
//!     files stay mapped until the last of them drops — that is the cursor
//!     drain, and it costs nothing.
//!
//! # The flip
//!
//! The expensive part — copying every live row — runs on the compactor's own
//! thread against a liveness snapshot taken at the start (`live_x`), while
//! writes keep landing in the old generation. Then:
//!
//! 1. **Close the write gate** ([`WriteGate`]). Every write holds the gate's
//!    read side from the moment it resolves an ordinal to the moment the WAL
//!    acks it, so closing it waits out every in-flight write. After that, no
//!    record anywhere names an old ordinal that has not already applied. This
//!    is what lets renumbering need no generation tag in the WAL records.
//! 2. **On the commit thread** (`Apply::control`, between batches):
//!    - catch up: retire `live_x \ live_now` at their new ordinals
//!      (`rank - 1`) and append `live_now \ live_x`. Pure bitmap arithmetic,
//!      because liveness has one owner and ordinals are never reused;
//!    - make the new stores durable, stamped with `last_applied`. Nothing
//!      else is checkpointed: the old generation recovers from the log;
//!    - rename `collection-{id}` → `.old`, then `.compact` → `collection-{id}`
//!      — the COMMIT POINT;
//!    - point the copy's stores at the renamed directory, swap its writers
//!      in, and publish its readers.
//! 3. Reopen the gate, then free the old generation on the compactor's
//!    thread — the slowest step of all, and one no write has to wait for.
//!
//! Records from before the flip still name OLD ordinals, and the WAL may keep
//! some of them (truncation is a global minimum across every collection). They
//! never reach the new generation: recovery skips any record at or below its
//! collection's own durable watermark (`IndexApplier::floors`), and the new
//! generation is stamped with the flip's LSN.
//!
//! # Crashes
//!
//! [`settle`] runs before every collection opens and resolves the directory
//! state a crash can leave:
//!
//! | on disk                         | meaning                  | action              |
//! |---------------------------------|--------------------------|---------------------|
//! | `collection-{id}` + `.compact`  | crashed before the swap  | delete `.compact`   |
//! | `.old` + `.compact`, no canonical | crashed between renames | roll back `.old`    |
//! | `collection-{id}` + `.old`      | crashed after the commit | delete `.old`       |
//!
//! Rolling back is always safe: the old generation's watermark is in the
//! global truncation minimum, so the log still holds every record above it,
//! each naming an old ordinal — exactly what replay onto the old generation
//! needs. The gate guaranteed nothing was logged after the flip.

use std::any::Any;
use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use super::{
    CatalogMap, Collection, CollectionConfig, CollectionWriters, IndexApplier, SharedCatalog,
    catalog_snapshot, collection_dir, open_collection_in, to_io,
};
use crate::error::{Error, Result};
use crate::metadata::common::{ColumnId, Ordinal, Row};
use crate::metadata::index::LiveSet;
use crate::metadata::tuples::RowGet;
use crate::wal::wal::{Lsn, WalHandle};

/// When the background compactor rebuilds a collection on its own.
///
/// Two triggers, either one sufficient:
///
///   * **waste** — dead / allocated ≥ `waste_ratio` AND dead ≥
///     `waste_min_dead`. Deletes and copy-on-write updates slowly bloat the
///     files and make every search scan dead blocks. The floor stops a tiny
///     collection rebuilding over a handful of rows. Checked on the
///     checkpoint interval.
///   * **headroom** — allocated ≥ `headroom_ratio` × capacity AND dead ≥
///     `headroom_min_dead_ratio` × capacity. Ordinals are about to run out and
///     compacting would actually free some. Checked as an insert crosses the
///     line (and on the interval), so a burst does not have to wait a tick.
///
/// "Allocated" is the ordinal high-water mark; "dead" is allocated minus live,
/// which also counts ordinals burned by failed appends.
///
/// If neither fires and the collection fills anyway, inserts are refused with
/// `CapacityExceeded` — the backstop, not the plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionOptions {
    /// Run the background compactor at all. `false` leaves the triggers off;
    /// nothing else changes.
    pub enabled: bool,
    /// Waste trigger: fraction of allocated ordinals that are dead.
    pub waste_ratio: f64,
    /// Waste trigger: minimum number of dead rows.
    pub waste_min_dead: u64,
    /// Headroom trigger: fraction of capacity allocated.
    pub headroom_ratio: f64,
    /// Headroom trigger: minimum dead rows, as a fraction of capacity.
    pub headroom_min_dead_ratio: f64,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        CompactionOptions {
            enabled: true,
            waste_ratio: 0.30,
            waste_min_dead: 10_000,
            headroom_ratio: 0.90,
            headroom_min_dead_ratio: 0.05,
        }
    }
}

impl CompactionOptions {
    /// Does a collection in this state warrant a rebuild?
    fn should_compact(&self, capacity: usize, allocated: u64, live: u64) -> bool {
        let dead = allocated.saturating_sub(live);
        if dead == 0 {
            return false;
        }
        let waste =
            dead >= self.waste_min_dead && dead as f64 >= self.waste_ratio * allocated as f64;
        let headroom = allocated as f64 >= self.headroom_ratio * capacity as f64
            && dead as f64 >= self.headroom_min_dead_ratio * capacity as f64;
        waste || headroom
    }

    /// The allocated count at which the headroom trigger starts to apply.
    fn headroom_line(&self, capacity: usize) -> u64 {
        (self.headroom_ratio * capacity as f64).ceil() as u64
    }
}

// ---------------------------------------------------------------------------
// The write gate and the flip epoch
// ---------------------------------------------------------------------------

thread_local! {
    /// Gates this thread currently holds, by address. Lets a write that is
    /// already inside the gate call another gated write — `execute` holds it
    /// across a whole statement whose VM calls `Db::insert`/`Db::mutate` —
    /// without taking the read lock twice, which deadlocks once a flip is
    /// waiting for the write side.
    static HELD: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Every write holds this shared while it is in flight; a flip holds it
/// exclusively. See the module doc, step 1. Readers never touch it.
pub(crate) struct WriteGate {
    lock: RwLock<()>,
}

/// One write's hold on the gate. Re-entrant per thread: an inner hold on the
/// same gate takes no lock.
pub(crate) struct GateGuard<'a> {
    gate: usize,
    _shared: Option<RwLockReadGuard<'a, ()>>,
}

impl WriteGate {
    pub(crate) fn new() -> WriteGate {
        WriteGate {
            lock: RwLock::new(()),
        }
    }

    /// Enter as a writer. Blocks only while a flip is in progress.
    pub(crate) fn enter(&self) -> GateGuard<'_> {
        let gate = self as *const WriteGate as usize;
        let nested = HELD.with(|held| held.borrow().contains(&gate));
        let shared = if nested {
            None
        } else {
            Some(self.lock.read().unwrap_or_else(|e| e.into_inner()))
        };
        HELD.with(|held| held.borrow_mut().push(gate));
        GateGuard {
            gate,
            _shared: shared,
        }
    }

    /// Wait out every in-flight write and keep new ones out until the guard
    /// drops.
    fn close(&self) -> RwLockWriteGuard<'_, ()> {
        self.lock.write().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        HELD.with(|held| {
            let mut held = held.borrow_mut();
            if let Some(at) = held.iter().rposition(|g| *g == self.gate) {
                held.remove(at);
            }
        });
    }
}

/// A seqlock-style counter over generation flips: odd while one is being
/// published, bumped twice per flip.
///
/// A query that resolves several handles for one collection — the live set,
/// then the metadata reader, then the flat reader — must get them all from the
/// SAME generation, or it reads old ordinals out of new stores and returns the
/// wrong rows without an error. Rather than lock, it notes the epoch before it
/// starts and checks it after; if a flip landed in between, it starts again.
pub(crate) struct Epoch(AtomicU64);

impl Epoch {
    pub(crate) fn new() -> Epoch {
        Epoch(AtomicU64::new(0))
    }

    /// The current epoch, waiting out a publish in progress (an in-memory
    /// pointer swap, so microseconds).
    pub(crate) fn begin(&self) -> u64 {
        loop {
            let seen = self.0.load(Ordering::SeqCst);
            if seen.is_multiple_of(2) {
                return seen;
            }
            std::thread::yield_now();
        }
    }

    /// Did no flip start since `begin` returned `seen`?
    pub(crate) fn unchanged(&self, seen: u64) -> bool {
        self.0.load(Ordering::SeqCst) == seen
    }

    fn publish(&self, swap: impl FnOnce()) {
        self.0.fetch_add(1, Ordering::SeqCst);
        swap();
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Directories
// ---------------------------------------------------------------------------

fn staging_dir(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("collection-{id}.compact"))
}

fn retired_dir(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("collection-{id}.old"))
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    fs::File::open(dir)?.sync_all()
}

/// Resolve whatever an interrupted compaction left behind, before the
/// collection opens. A no-op when there is nothing to resolve. See the
/// module-level table.
pub(super) fn settle(dir: &Path, id: u32) -> Result<()> {
    let live = collection_dir(dir, id);
    let staged = staging_dir(dir, id);
    let retired = retired_dir(dir, id);
    let mut changed = false;

    if !live.exists() && retired.exists() {
        // Crashed between the two renames: put the old generation back.
        fs::rename(&retired, &live)?;
        changed = true;
    } else if !live.exists() && staged.exists() {
        // The protocol renames the canonical directory away before the staged
        // one moves in, so a staged copy with neither beside it was not left
        // by a compaction. Refuse rather than open an empty collection.
        return Err(Error::CorruptSnapshot(format!(
            "collection-{id}: only a compaction staging directory exists"
        )));
    }
    if staged.exists() {
        fs::remove_dir_all(&staged)?;
        changed = true;
    }
    if retired.exists() {
        fs::remove_dir_all(&retired)?;
        changed = true;
    }
    if changed {
        sync_dir(dir)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Building the new generation
// ---------------------------------------------------------------------------

/// Every scalar column, in `ColumnId` order — a copied row carries all of them.
fn scalar_columns(config: &CollectionConfig) -> Vec<ColumnId> {
    (0..config.schema.columns.len() as ColumnId).collect()
}

/// Copy one row from `src` at `old` into `dst` at `new`, through the same
/// three writes apply uses — data first, liveness last.
fn copy_row(
    src: &Collection,
    dst: &mut CollectionWriters,
    columns: &[ColumnId],
    old: u32,
    new: u32,
) -> Result<()> {
    let missing = || Error::SnapshotRowMissing {
        collection: src.config.id,
        ordinal: old,
    };
    let vector = src.reader.vector_at(Ordinal(old)).ok_or_else(missing)?;
    let values = match src.tuple.get(Ordinal(old), columns)? {
        RowGet::Live(values) => values,
        RowGet::Missing => return Err(missing()),
    };
    let row: Row = values
        .into_iter()
        .enumerate()
        .map(|(id, value)| (id as ColumnId, value))
        .collect();
    dst.flat.write_at(new as u64, vector)?;
    dst.tuple.write_row(Ordinal(new), &row)?;
    dst.meta.insert_row(Ordinal(new), &row)?;
    Ok(())
}

/// How many rows the builder copies between checks of the stop flag.
const STOP_CHECK_EVERY: usize = 4096;

/// Write every row of `live_x` into fresh stores in the staging directory, at
/// ordinals `0..live_x.len()` in ascending old-ordinal order — so a row's new
/// ordinal is its rank in `live_x` minus one, which is what the catch-up step
/// relies on.
///
/// Runs off the commit thread: it only reads the old generation, through the
/// same lock-free handles any query uses, and writes stores no one else can
/// see yet.
fn build(
    dir: &Path,
    src: &Collection,
    live_x: &LiveSet,
    stop: &AtomicBool,
) -> Result<(CollectionWriters, Collection)> {
    let staged = staging_dir(dir, src.config.id);
    if staged.exists() {
        fs::remove_dir_all(&staged)?;
    }
    let (mut writers, readers, _watermark) = open_collection_in(&staged, &src.config)?;
    let columns = scalar_columns(&src.config);
    for (new, old) in live_x.iter().enumerate() {
        if new % STOP_CHECK_EVERY == 0 && stop.load(Ordering::Relaxed) {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::Interrupted,
                "compaction abandoned: the database is closing",
            )));
        }
        copy_row(src, &mut writers, &columns, old, new as u32)?;
    }
    // Flush the copied vectors now, while writes are still flowing. The flip
    // has to make the copy durable with the gate closed, and this is nearly
    // all of it: afterwards that sync only has the catch-up's pages to write.
    writers.flat.sync_data()?;
    Ok((writers, readers))
}

/// The commit-thread half of a compaction, sent through `WalHandle::control`.
pub(super) struct Flip {
    id: u32,
    /// The snapshot the builder copied.
    live_x: Arc<LiveSet>,
    /// The builder's writers over the staging directory.
    writers: CollectionWriters,
    /// Their read handles — what readers get once the flip publishes.
    collection: Collection,
    /// Where the flip sends the generation it replaced, to be freed by the
    /// compactor once the gate is open again (see [`Retired`]).
    graves: Sender<Retired>,
    /// Test fault point: die between the two renames, as a crash would.
    #[cfg(test)]
    crash_between_renames: bool,
}

/// The generation a flip replaced. Freeing it — the old stores, and the
/// mapping of the old flat index — is the slowest part of a flip, slower than
/// everything else together, and nothing needs it gone before writes resume.
/// So the commit thread hands it to the compactor instead of dropping it with
/// the gate closed. Queries still holding the old generation keep it alive
/// either way; whichever drops last frees it.
struct Retired {
    _writers: Option<CollectionWriters>,
    _catalog: Arc<CatalogMap>,
}

impl IndexApplier {
    /// Run a control job. The only kind there is: a compaction's flip.
    pub(super) fn run_control(&mut self, job: Box<dyn Any + Send>) -> io::Result<Option<u64>> {
        match job.downcast::<Flip>() {
            Ok(flip) => self.flip(*flip),
            Err(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown control job",
            )),
        }
    }

    /// Step 2 of the module doc. Runs on the commit thread with the write
    /// gate closed, so nothing applies while it runs and nothing in flight
    /// names an old ordinal.
    ///
    /// Returns no truncation point: nothing else is checkpointed here, so the
    /// log keeps everything until the next regular checkpoint.
    fn flip(&mut self, flip: Flip) -> io::Result<Option<u64>> {
        #[cfg(test)]
        let crash = flip.crash_between_renames;
        #[cfg(not(test))]
        let crash = false;
        let Flip {
            id,
            live_x,
            mut writers,
            collection,
            graves,
            ..
        } = flip;
        // Up to the commit point nothing the database depends on has changed;
        // a failure just throws the staged copy away.
        if let Err(e) = self.stage(id, &live_x, &mut writers) {
            drop(writers);
            let _ = fs::remove_dir_all(staging_dir(&self.dir, id));
            return Err(e);
        }
        self.swap_dirs(id, crash)?;
        let retired = self.install(id, writers, collection);
        // The compactor is blocked on this job's ack, so it is there to
        // receive; if it is somehow gone, freeing here is merely slower.
        let _ = graves.send(retired);
        Ok(None)
    }

    /// Catch the staged copy up with everything that happened during the
    /// build and make it durable at `last_applied`.
    ///
    /// Nothing else is checkpointed. The old generation does not need to be
    /// durable at the flip for a rollback to be safe: its own watermark is
    /// already in the global truncation minimum, so the log still holds every
    /// record above it, and recovery replays them onto it at their old
    /// ordinals — which is what they name.
    fn stage(
        &mut self,
        id: u32,
        live_x: &LiveSet,
        staged: &mut CollectionWriters,
    ) -> io::Result<()> {
        let current = catalog_snapshot(&self.catalog)
            .get(&id)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("unknown collection {id}"))
            })?;
        let live_now = current.meta.live();
        let before = live_x.bitmap();

        // Retired since the snapshot: find each at its NEW ordinal. The builder
        // wrote `live_x` densely in ascending order, so that is its rank.
        for old in before - &live_now {
            let new = before.rank(old) - 1;
            staged.meta.remove_row(Ordinal(new as u32)).map_err(to_io)?;
        }
        // Installed since the snapshot: append, in ascending old order. Every
        // one of them is a distinct allocated ordinal, so this never runs past
        // capacity.
        let columns = scalar_columns(&current.config);
        let base = before.len() as u32;
        for (i, old) in (&live_now - before).iter().enumerate() {
            copy_row(&current, staged, &columns, old, base + i as u32).map_err(to_io)?;
        }

        // Stamp the copy with the flip's LSN — it reflects every record so far
        // — and make it durable.
        let lsn = self.last_applied;
        staged.flat.advance_applied_lsn(lsn);
        staged.meta.advance_applied_lsn(Lsn(lsn));
        staged.tuple.advance_applied_lsn(Lsn(lsn));
        staged.checkpoint()?;
        sync_dir(&staging_dir(&self.dir, id))
    }

    /// The two renames. After the second one returns, the collection
    /// IS the new generation on disk.
    ///
    /// `crash` is the test fault point: stop after the first rename and leave
    /// the directories exactly as a crash there would.
    fn swap_dirs(&mut self, id: u32, crash: bool) -> io::Result<()> {
        let live = collection_dir(&self.dir, id);
        let staged = staging_dir(&self.dir, id);
        let retired = retired_dir(&self.dir, id);

        if let Err(e) = fs::rename(&live, &retired).and_then(|()| sync_dir(&self.dir)) {
            // If the rename itself landed, put it back; the old generation's
            // writers are still the ones in use.
            if !live.exists() && fs::rename(&retired, &live).is_err() {
                self.halt(id, "could not restore the collection directory", &e);
                return Err(e);
            }
            let _ = fs::remove_dir_all(&staged);
            return Err(e);
        }
        if crash {
            let e = io::Error::other("injected crash between the renames");
            self.halt(id, "test fault point", &e);
            return Err(e);
        }
        if let Err(e) = fs::rename(&staged, &live) {
            if fs::rename(&retired, &live).is_err() {
                self.halt(id, "could not restore the collection directory", &e);
            } else {
                let _ = fs::remove_dir_all(&staged);
            }
            return Err(e);
        }
        // COMMITTED. A dir fsync failure now leaves the rename's durability
        // unknown, so the only safe move is to log nothing further.
        if let Err(e) = sync_dir(&self.dir) {
            self.halt(id, "could not make the compacted collection durable", &e);
            return Err(e);
        }
        // The old generation's files stay mapped by whatever still holds them.
        // Left behind on failure, `settle` removes it on the next open.
        let _ = fs::remove_dir_all(&retired);
        Ok(())
    }

    /// After the renames: swap the copy's writers in and publish its readers. The old
    /// generation's handles stay alive in whatever queries still hold them;
    /// this thread's are returned, for the compactor to free.
    fn install(
        &mut self,
        id: u32,
        mut writers: CollectionWriters,
        collection: Collection,
    ) -> Retired {
        // The stores were built in the staging directory, which is now the
        // collection's; their next checkpoint must land there.
        let live = collection_dir(&self.dir, id);
        writers.meta.relocate(&live);
        writers.tuple.relocate(&live);
        // The catch-up changed liveness after the handle was minted.
        writers.live.bump();
        collection
            .next_ordinal
            .store(collection.reader.len() as u64, Ordering::Release);
        self.floors.insert(id, self.last_applied);
        let old_writers = self.writers.insert(id, writers);
        let catalog = Arc::clone(&self.catalog);
        let mut old_catalog = None;
        self.epoch.publish(|| {
            let mut guard = catalog.write().unwrap_or_else(|e| e.into_inner());
            let mut map: CatalogMap = (**guard).clone();
            map.insert(id, Arc::new(collection));
            old_catalog = Some(std::mem::replace(&mut *guard, Arc::new(map)));
        });
        Retired {
            _writers: old_writers,
            // `publish` runs its closure exactly once.
            _catalog: old_catalog.unwrap_or_default(),
        }
    }

    /// Stop absorbing records: memory and disk disagree about which generation
    /// is the collection. The log halts before writing another frame; reopening
    /// the database resolves the directories and recovers.
    fn halt(&mut self, id: u32, what: &str, e: &io::Error) {
        self.halted
            .get_or_insert_with(|| format!("compaction of collection {id}: {what}: {e}"));
    }
}

// ---------------------------------------------------------------------------
// Running a compaction
// ---------------------------------------------------------------------------

/// What every compaction needs, shared by `Db::compact_now` and the
/// background thread. Deliberately holds no `WalHandle`: a live sender would
/// keep the commit thread from ever seeing its channel close.
pub(super) struct Shared {
    pub(super) dir: PathBuf,
    pub(super) catalog: SharedCatalog,
    pub(super) gate: Arc<WriteGate>,
    pub(super) opts: CompactionOptions,
    /// One compaction at a time, per database.
    pub(super) running: Mutex<()>,
    /// Set on close: a build in progress gives up.
    pub(super) stop: AtomicBool,
}

impl Shared {
    /// Rebuild `id` if it has anything to reclaim. `Ok(false)` when every
    /// allocated ordinal is live, so there is nothing to do.
    pub(super) fn compact(&self, wal: &WalHandle, id: u32) -> Result<bool> {
        let _one = self.running.lock().unwrap_or_else(|e| e.into_inner());
        let coll = catalog_snapshot(&self.catalog)
            .get(&id)
            .cloned()
            .ok_or(Error::UnknownCollection { id })?;
        let live_x = coll.live.resolve();
        if live_x.len() == coll.next_ordinal.load(Ordering::Acquire) {
            return Ok(false);
        }
        let (writers, collection) = match build(&self.dir, &coll, &live_x, &self.stop) {
            Ok(built) => built,
            Err(e) => {
                let _ = fs::remove_dir_all(staging_dir(&self.dir, id));
                return Err(e);
            }
        };
        drop(coll);
        let (graves, retired) = mpsc::channel();
        let flipped = {
            let _closed = self.gate.close();
            wal.control(Box::new(Flip {
                id,
                live_x,
                writers,
                collection,
                graves,
                #[cfg(test)]
                crash_between_renames: false,
            }))
        };
        // Writes are flowing again; only now free the old generation.
        drop(retired.try_iter().collect::<Vec<_>>());
        flipped?;
        Ok(true)
    }

    /// Check one collection against the triggers and compact it if one fires.
    fn evaluate(&self, wal: &WalHandle, id: u32) {
        let Some(coll) = catalog_snapshot(&self.catalog).get(&id).cloned() else {
            return;
        };
        let allocated = coll.next_ordinal.load(Ordering::Acquire);
        let live = coll.meta.live_count();
        let fire = self
            .opts
            .should_compact(coll.config.capacity, allocated, live);
        drop(coll);
        if fire {
            // Best-effort, like a checkpoint: a failed compaction changed
            // nothing and the next trigger tries again.
            let _ = self.compact(wal, id);
        }
    }
}

/// The background compactor: wakes on the checkpoint interval to check every
/// collection, and early when an insert crosses the headroom line.
pub(super) struct Compactor {
    nudge: Sender<u32>,
    join: JoinHandle<()>,
    opts: CompactionOptions,
}

impl Compactor {
    pub(super) fn spawn(shared: Arc<Shared>, wal: WalHandle, interval: Duration) -> Compactor {
        let (nudge, rx) = mpsc::channel::<u32>();
        let opts = shared.opts;
        let join = std::thread::Builder::new()
            .name("flats-compactor".into())
            .spawn(move || {
                loop {
                    match rx.recv_timeout(interval) {
                        Ok(id) => shared.evaluate(&wal, id),
                        Err(RecvTimeoutError::Timeout) => {
                            let mut ids: Vec<u32> =
                                catalog_snapshot(&shared.catalog).keys().copied().collect();
                            ids.sort_unstable();
                            for id in ids {
                                if shared.stop.load(Ordering::Relaxed) {
                                    break;
                                }
                                shared.evaluate(&wal, id);
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .expect("spawn compactor thread");
        Compactor { nudge, join, opts }
    }

    /// Called by a write that allocated ordinals `[from, to)`: wake the
    /// compactor if that crossed the headroom line. Only the crossing nudges,
    /// so a stream of inserts above the line sends one message, not one each.
    pub(super) fn allocated(&self, id: u32, capacity: usize, from: u64, to: u64) {
        let line = self.opts.headroom_line(capacity);
        if from < line && to >= line {
            let _ = self.nudge.send(id);
        }
    }

    pub(super) fn stop(self) {
        drop(self.nudge);
        let _ = self.join.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waste_needs_both_the_ratio_and_the_floor() {
        let o = CompactionOptions::default();
        // 30% dead of a large collection: fires.
        assert!(o.should_compact(1_000_000, 100_000, 70_000));
        // 30% dead but only 3 rows: the floor holds it.
        assert!(!o.should_compact(1_000_000, 10, 7));
        // Plenty of dead rows, but only 10% of what is allocated.
        assert!(!o.should_compact(1_000_000, 200_000, 180_000));
    }

    #[test]
    fn headroom_needs_the_line_and_something_to_free() {
        let o = CompactionOptions::default();
        // 90% allocated, 5% of capacity dead: fires.
        assert!(o.should_compact(1000, 900, 850));
        // 90% allocated but almost nothing dead: compacting gains nothing.
        assert!(!o.should_compact(1000, 900, 899));
        // Below the line.
        assert!(!o.should_compact(1000, 899, 800));
    }

    #[test]
    fn nothing_dead_never_fires() {
        let o = CompactionOptions::default();
        assert!(!o.should_compact(10, 10, 10));
        assert!(!o.should_compact(10, 0, 0));
    }

    #[test]
    fn the_gate_is_reentrant_on_one_thread() {
        let gate = WriteGate::new();
        let outer = gate.enter();
        // Would deadlock against a waiting flip if this took the lock again.
        let inner = gate.enter();
        drop(inner);
        drop(outer);
        // Fully released: the write side is available.
        drop(gate.close());
    }

    #[test]
    fn closing_the_gate_waits_for_writers() {
        let gate = Arc::new(WriteGate::new());
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let writer = {
            let (gate, entered, release) = (gate.clone(), entered.clone(), release.clone());
            std::thread::spawn(move || {
                let _g = gate.enter();
                entered.store(true, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
            })
        };
        while !entered.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let closed = Arc::new(AtomicBool::new(false));
        let closer = {
            let (gate, closed) = (gate.clone(), closed.clone());
            std::thread::spawn(move || {
                let _w = gate.close();
                closed.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !closed.load(Ordering::SeqCst),
            "closed over an in-flight write"
        );
        release.store(true, Ordering::SeqCst);
        writer.join().unwrap();
        closer.join().unwrap();
        assert!(closed.load(Ordering::SeqCst));
    }

    #[test]
    fn the_epoch_moves_across_a_publish() {
        let epoch = Epoch::new();
        let seen = epoch.begin();
        assert!(epoch.unchanged(seen));
        epoch.publish(|| {});
        assert!(!epoch.unchanged(seen));
        assert!(epoch.unchanged(epoch.begin()));
    }

    #[test]
    fn settle_is_a_no_op_on_a_clean_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(collection_dir(dir.path(), 0)).unwrap();
        settle(dir.path(), 0).unwrap();
        assert!(collection_dir(dir.path(), 0).exists());
        // A collection that does not exist yet is fine too.
        settle(dir.path(), 7).unwrap();
    }

    #[test]
    fn settle_discards_an_uncommitted_copy() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(collection_dir(dir.path(), 0)).unwrap();
        fs::create_dir(staging_dir(dir.path(), 0)).unwrap();
        settle(dir.path(), 0).unwrap();
        assert!(collection_dir(dir.path(), 0).exists());
        assert!(!staging_dir(dir.path(), 0).exists());
    }

    #[test]
    fn settle_rolls_back_between_the_renames() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(retired_dir(dir.path(), 0)).unwrap();
        fs::write(retired_dir(dir.path(), 0).join("marker"), b"old").unwrap();
        fs::create_dir(staging_dir(dir.path(), 0)).unwrap();
        settle(dir.path(), 0).unwrap();
        assert_eq!(
            fs::read(collection_dir(dir.path(), 0).join("marker")).unwrap(),
            b"old"
        );
        assert!(!staging_dir(dir.path(), 0).exists());
        assert!(!retired_dir(dir.path(), 0).exists());
    }

    #[test]
    fn settle_finishes_a_committed_swap() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(collection_dir(dir.path(), 0)).unwrap();
        fs::write(collection_dir(dir.path(), 0).join("marker"), b"new").unwrap();
        fs::create_dir(retired_dir(dir.path(), 0)).unwrap();
        settle(dir.path(), 0).unwrap();
        assert_eq!(
            fs::read(collection_dir(dir.path(), 0).join("marker")).unwrap(),
            b"new"
        );
        assert!(!retired_dir(dir.path(), 0).exists());
    }

    #[test]
    fn settle_refuses_a_lone_staging_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(staging_dir(dir.path(), 0)).unwrap();
        assert!(matches!(
            settle(dir.path(), 0),
            Err(Error::CorruptSnapshot(_))
        ));
    }

    // -- end to end, through `Db` ---------------------------------------------

    use crate::engine::{Db, DbOptions};
    use crate::vm::exec::RegValue;
    use std::collections::BTreeMap;

    /// No timer, no background compactor: every compaction below is asked for.
    fn manual() -> DbOptions {
        DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
            compaction: CompactionOptions {
                enabled: false,
                ..Default::default()
            },
        }
    }

    /// `docs` is collection 0. Row `id` carries `n = id * 10` and the vector
    /// `[id, 1]`, so every row can be checked against its key alone.
    fn create_docs(db: &Db, capacity: usize) {
        db.execute(&format!(
            "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(2), n INT) \
             WITH (capacity = {capacity});"
        ))
        .expect("create");
    }

    fn put(db: &Db, id: i64) -> std::result::Result<(), crate::QueryError> {
        db.execute(&format!(
            "INSERT INTO docs (id, vector, n) VALUES ({id}, [{id}.0, 1.0], {});",
            id * 10
        ))
        .map(drop)
    }

    fn del(db: &Db, id: i64) {
        db.execute(&format!("DELETE FROM docs WHERE id = {id};"))
            .expect("delete");
    }

    /// Every row, by key: `(n, vector)`.
    fn rows(db: &Db) -> BTreeMap<i64, (i64, Vec<f32>)> {
        db.execute("SELECT id, n, vector FROM docs;")
            .expect("select")
            .map(|row| match row.expect("row").0.as_slice() {
                [RegValue::Int(id), RegValue::Int(n), RegValue::Vector(v)] => {
                    (*id, (*n, v.to_vec()))
                }
                other => panic!("unexpected row {other:?}"),
            })
            .collect()
    }

    /// The rows `ids` should produce, with `n` overridden where given.
    fn expect(ids: impl IntoIterator<Item = i64>) -> BTreeMap<i64, (i64, Vec<f32>)> {
        ids.into_iter()
            .map(|id| (id, (id * 10, vec![id as f32, 1.0])))
            .collect()
    }

    fn high_water(db: &Db) -> usize {
        db.reader(0).expect("docs").len()
    }

    #[test]
    fn compaction_reclaims_dead_rows_and_their_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        create_docs(&db, 10);
        for id in 0..10 {
            put(&db, id).unwrap();
        }
        for id in [0, 2, 4, 6, 8] {
            del(&db, id);
        }
        // Full: retiring a row does not give its ordinal back.
        assert!(put(&db, 10).is_err());

        assert!(db.compact_now(0).unwrap());
        assert_eq!(rows(&db), expect([1, 3, 5, 7, 9]));
        assert_eq!(high_water(&db), 5, "live rows packed into 0..5");

        for id in 10..15 {
            put(&db, id).unwrap();
        }
        assert!(put(&db, 15).is_err(), "capacity still holds");
        assert_eq!(rows(&db), expect([1, 3, 5, 7, 9, 10, 11, 12, 13, 14]));
        db.close().unwrap();
    }

    #[test]
    fn nothing_dead_means_nothing_to_do() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        create_docs(&db, 10);
        for id in 0..3 {
            put(&db, id).unwrap();
        }
        assert!(!db.compact_now(0).unwrap());
        assert_eq!(rows(&db), expect(0..3));
        db.close().unwrap();
    }

    #[test]
    fn a_compacted_collection_survives_reopen_and_leaves_no_debris() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[], manual()).unwrap();
            create_docs(&db, 16);
            for id in 0..8 {
                put(&db, id).unwrap();
            }
            del(&db, 3);
            del(&db, 4);
            assert!(db.compact_now(0).unwrap());
            db.close().unwrap();
        }
        assert!(!staging_dir(dir.path(), 0).exists());
        assert!(!retired_dir(dir.path(), 0).exists());

        let db = Db::open(dir.path(), &[], manual()).unwrap();
        assert_eq!(rows(&db), expect([0, 1, 2, 5, 6, 7]));
        assert_eq!(high_water(&db), 6);
        db.close().unwrap();
    }

    /// THE renumbering hazard. A second, never-written collection pins the
    /// global truncation point at 0, so the WAL keeps every record — including
    /// the ones from before the flip, which name OLD ordinals. A crash then
    /// replays the whole log; only the per-collection floor stops those records
    /// landing on the wrong rows of the new generation.
    #[test]
    fn records_from_before_the_flip_are_not_replayed_onto_the_new_generation() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[], manual()).unwrap();
            create_docs(&db, 32);
            db.execute(
                "CREATE COLLECTION other (id INT PRIMARY KEY, vector VECTOR(2)) \
                 WITH (capacity = 4);",
            )
            .unwrap();
            for id in 0..10 {
                put(&db, id).unwrap();
            }
            for id in [1, 2, 5] {
                del(&db, id);
            }
            assert!(db.compact_now(0).unwrap());
            // After the flip: new ordinals.
            put(&db, 10).unwrap();
            del(&db, 7);
            db.execute("UPDATE docs SET n = 990 WHERE id = 9;").unwrap();
            // A crash: no final checkpoint.
            drop(db);
        }
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        let mut want = expect([0, 3, 4, 6, 8, 9, 10]);
        want.get_mut(&9).unwrap().0 = 990;
        assert_eq!(rows(&db), want);
        db.close().unwrap();
    }

    /// A crash between the two renames rolls back to the old generation — and
    /// the flip no longer checkpoints it, so it is BEHIND: everything since
    /// its last checkpoint exists only in the log, at old ordinals. Recovery
    /// must replay all of it onto the rolled-back directory.
    #[test]
    fn a_crash_between_the_renames_rolls_back_and_replays() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[], manual()).unwrap();
            create_docs(&db, 32);
            for id in 0..10 {
                put(&db, id).unwrap();
            }
            db.checkpoint().unwrap();
            // Only in the log from here on.
            for id in [1, 2, 5] {
                del(&db, id);
            }
            put(&db, 10).unwrap();
            db.execute("UPDATE docs SET n = 990 WHERE id = 9;").unwrap();

            let shared = Arc::clone(&db.compaction);
            let wal = db.wal_handle().unwrap();
            let coll = catalog_snapshot(&shared.catalog).get(&0).cloned().unwrap();
            let live_x = coll.live.resolve();
            let (writers, collection) =
                build(&shared.dir, &coll, &live_x, &shared.stop).unwrap();
            drop(coll);
            {
                let _closed = shared.gate.close();
                let flipped = wal.control(Box::new(Flip {
                    id: 0,
                    live_x,
                    writers,
                    collection,
                    graves: mpsc::channel().0,
                    crash_between_renames: true,
                }));
                assert!(flipped.is_err());
            }
            drop(wal);
            assert!(!collection_dir(dir.path(), 0).exists());
            assert!(retired_dir(dir.path(), 0).exists());
            assert!(staging_dir(dir.path(), 0).exists());
            // The crash: no final checkpoint.
            drop(db);
        }

        let mut want = expect([0, 3, 4, 6, 7, 8, 9, 10]);
        want.get_mut(&9).unwrap().0 = 990;
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        assert!(!retired_dir(dir.path(), 0).exists());
        assert!(!staging_dir(dir.path(), 0).exists());
        assert_eq!(rows(&db), want);
        // Still the old generation: 10 puts, 1 more, and the update's install.
        assert_eq!(high_water(&db), 12);

        // And it compacts cleanly from there.
        assert!(db.compact_now(0).unwrap());
        assert_eq!(rows(&db), want);
        db.close().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        assert_eq!(rows(&db), want);
        db.close().unwrap();
    }

    /// Writes that land while the builder is copying are folded in at the flip:
    /// retires found at their new ordinals, installs appended. Driven by hand so
    /// the writes provably fall between the snapshot and the flip.
    #[test]
    fn writes_during_the_build_are_caught_up_at_the_flip() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        create_docs(&db, 64);
        for id in 0..10 {
            put(&db, id).unwrap();
        }
        del(&db, 0);
        del(&db, 5);

        // `Shared::compact`, split open at the gap.
        let shared = Arc::clone(&db.compaction);
        let wal = db.wal_handle().unwrap();
        let coll = catalog_snapshot(&shared.catalog).get(&0).cloned().unwrap();
        let live_x = coll.live.resolve();
        let (writers, collection) = build(&shared.dir, &coll, &live_x, &shared.stop).unwrap();
        drop(coll);

        del(&db, 3); // a row the builder copied
        put(&db, 20).unwrap(); // a row it never saw
        put(&db, 21).unwrap();
        del(&db, 20); // ...and one of those, gone again
        db.execute("UPDATE docs SET n = 770 WHERE id = 7;").unwrap(); // retire + install

        {
            let _closed = shared.gate.close();
            wal.control(Box::new(Flip {
                id: 0,
                live_x,
                writers,
                collection,
                graves: mpsc::channel().0,
                crash_between_renames: false,
            }))
            .unwrap();
        }
        // A live handle would keep the commit thread from ever shutting down.
        drop(wal);

        let mut want = expect([1, 2, 4, 6, 7, 8, 9, 21]);
        want.get_mut(&7).unwrap().0 = 770;
        assert_eq!(rows(&db), want);
        db.close().unwrap();

        let db = Db::open(dir.path(), &[], manual()).unwrap();
        assert_eq!(rows(&db), want);
        db.close().unwrap();
    }

    #[test]
    fn the_key_stays_unique_after_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        create_docs(&db, 16);
        for id in 0..4 {
            put(&db, id).unwrap();
        }
        del(&db, 0);
        assert!(db.compact_now(0).unwrap());
        assert!(put(&db, 2).is_err(), "2 is still held by a live row");
        put(&db, 0).unwrap(); // freed by the delete, before compaction
        del(&db, 2);
        put(&db, 2).unwrap();
        assert_eq!(rows(&db), expect(0..4));
        db.close().unwrap();
    }

    /// A stream opened before the flip is pinned to the old generation's
    /// handles, so it keeps returning exactly the rows it was opened over.
    #[test]
    fn an_open_stream_keeps_reading_the_old_generation() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        create_docs(&db, 32);
        for id in 0..12 {
            put(&db, id).unwrap();
        }
        for id in [0, 1, 2, 3] {
            del(&db, id);
        }
        let stream = db.execute("SELECT id, n FROM docs;").unwrap();
        assert!(db.compact_now(0).unwrap());
        put(&db, 100).unwrap();

        let seen: Vec<(i64, i64)> = stream
            .map(|row| match row.unwrap().0.as_slice() {
                [RegValue::Int(id), RegValue::Int(n)] => (*id, *n),
                other => panic!("unexpected row {other:?}"),
            })
            .collect();
        assert_eq!(seen, (4..12).map(|id| (id, id * 10)).collect::<Vec<_>>());
        db.close().unwrap();
    }

    /// Readers, a writer, and repeated compactions at once. Every row a reader
    /// sees must be internally consistent (`n` and the vector both derive from
    /// the key) — a query that mixed two generations' handles would pair a key
    /// with some other row's values.
    #[test]
    fn readers_and_writers_run_through_compactions() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Db::open(dir.path(), &[], manual()).unwrap());
        create_docs(&db, 4096);
        for id in 0..200 {
            put(&db, id).unwrap();
        }
        let done = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (db, done) = (Arc::clone(&db), Arc::clone(&done));
                std::thread::spawn(move || {
                    let mut reads = 0u64;
                    while !done.load(Ordering::SeqCst) {
                        for (id, (n, v)) in rows(&db) {
                            assert_eq!(n, id * 10, "row {id} has another row's n");
                            assert_eq!(
                                v,
                                vec![id as f32, 1.0],
                                "row {id} has another row's vector"
                            );
                        }
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();

        let writer = {
            let (db, done) = (Arc::clone(&db), Arc::clone(&done));
            std::thread::spawn(move || {
                let mut next = 200i64;
                let mut oldest = 0i64;
                while !done.load(Ordering::SeqCst) {
                    put(&db, next).unwrap();
                    next += 1;
                    del(&db, oldest);
                    oldest += 1;
                }
                (oldest, next)
            })
        };

        let mut flips = 0;
        while flips < 10 {
            if db.compact_now(0).unwrap() {
                flips += 1;
            }
        }
        done.store(true, Ordering::SeqCst);
        let (oldest, next) = writer.join().unwrap();
        for r in readers {
            assert!(r.join().unwrap() > 0, "a reader never completed a scan");
        }

        assert_eq!(rows(&db), expect(oldest..next));
        let db = Arc::into_inner(db).expect("threads joined");
        db.close().unwrap();
        let db = Db::open(dir.path(), &[], manual()).unwrap();
        assert_eq!(rows(&db), expect(oldest..next));
        db.close().unwrap();
    }

    fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn the_waste_trigger_compacts_in_the_background() {
        let dir = tempfile::tempdir().unwrap();
        let opts = DbOptions {
            checkpoint_interval: Duration::from_millis(20),
            compaction: CompactionOptions {
                // Fires only once all five deletes are in; a lower floor could
                // compact after four and leave the fifth below the ratio.
                waste_min_dead: 5,
                ..Default::default()
            },
        };
        let db = Db::open(dir.path(), &[], opts).unwrap();
        create_docs(&db, 1000);
        for id in 0..10 {
            put(&db, id).unwrap();
        }
        for id in 0..5 {
            del(&db, id);
        }
        wait_for("the waste trigger", || high_water(&db) == 5);
        assert_eq!(rows(&db), expect(5..10));
        db.close().unwrap();
    }

    /// No timer at all: only the insert that crosses the headroom line can
    /// start this compaction.
    #[test]
    fn crossing_the_headroom_line_nudges_the_compactor() {
        let dir = tempfile::tempdir().unwrap();
        let opts = DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
            compaction: CompactionOptions {
                waste_min_dead: u64::MAX,
                headroom_ratio: 0.5,
                headroom_min_dead_ratio: 0.1,
                ..Default::default()
            },
        };
        let db = Db::open(dir.path(), &[], opts).unwrap();
        create_docs(&db, 20);
        for id in 0..6 {
            put(&db, id).unwrap();
        }
        for id in 0..3 {
            del(&db, id);
        }
        // Below the line, nothing happens however long we wait.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(high_water(&db), 6);
        // Allocated reaches 10 of 20 here.
        for id in 6..10 {
            put(&db, id).unwrap();
        }
        wait_for("the headroom nudge", || high_water(&db) == 7);
        assert_eq!(rows(&db), expect(3..10));
        db.close().unwrap();
    }
}
