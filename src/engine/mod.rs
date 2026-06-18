//! Execution layer: the thin engine that wires the WAL to the flat index.
//!
//! This is the minimal v1 executor — no SQL, no planner, no metadata filtering.
//! It owns the durability story end to end:
//!
//!   * WRITE PATH (`insert`/`delete`): allocate the next ordinal from the
//!     collection's high-water mark, build a *positional* WAL record carrying
//!     that ordinal, and block until the WAL has made it durable AND applied it
//!     into the index. The ordinal is assigned exactly once, on the logging
//!     side; apply/replay never recompute it.
//!
//!   * APPLY (`IndexApplier`): the WAL thread calls this post-fsync. It folds a
//!     record into the index with `write_at`/`delete` (both idempotent) so that
//!     replaying an already-applied record is a no-op.
//!
//!   * RECOVERY: on open, each collection's durable checkpoint watermark
//!     (`FlatIndex::checkpoint_lsn`) tells the WAL which prefix is already in the
//!     index; recovery replays only the tail.
//!
//!   * CHECKPOINT (`Flusher`): periodically makes the index durable and lets the
//!     WAL shed the now-redundant prefix, in a strict crash-safe order.
//!
//! Multi-collection catalog persistence is out of scope here: the caller hands
//! the collection set to `open` each time. The data structures are already
//! keyed by collection id so the catalog can be layered on without reshaping.

use std::collections::HashMap;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::index::index::{FlatIndex, Ordinal, SearchResult};
use crate::wal::wal::{Apply, Lsn, Record, Wal, WalHandle};

/// How a single collection is configured at open time. (Until the catalog is
/// persisted separately, the embedder supplies this on every open.)
#[derive(Debug, Clone, Copy)]
pub struct CollectionConfig {
    pub id: u32,
    pub dim: NonZeroUsize,
    pub capacity: usize,
}

/// Engine tunables.
#[derive(Debug, Clone, Copy)]
pub struct DbOptions {
    /// How often the background flusher runs a checkpoint. Set very large to
    /// effectively disable it and drive checkpoints manually via `checkpoint`.
    pub checkpoint_interval: Duration,
}

impl Default for DbOptions {
    fn default() -> Self {
        DbOptions {
            checkpoint_interval: Duration::from_secs(5),
        }
    }
}

/// One open collection: its index plus the live ordinal allocator.
struct Collection {
    index: Arc<Mutex<FlatIndex>>,
    /// Next ordinal to hand out. Seeded from the index high-water after
    /// recovery; advanced once per insert.
    next_ordinal: AtomicU64,
    dim: usize,
    capacity: usize,
}

impl Collection {
    /// Reserve the next ordinal, or fail if the collection is at capacity.
    /// CAS loop so a full collection never overshoots the high-water mark.
    fn alloc_ordinal(&self) -> Result<u64> {
        loop {
            let cur = self.next_ordinal.load(Ordering::Acquire);
            if cur >= self.capacity as u64 {
                return Err(Error::CapacityExceeded {
                    capacity: self.capacity,
                });
            }
            if self
                .next_ordinal
                .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(cur);
            }
        }
    }
}

/// Shared, read-only-after-open map of collections. Cloned (Arc) into the apply
/// path, the flusher, and the engine's own read path.
type Catalog = Arc<HashMap<u32, Collection>>;

// ---------------------------------------------------------------------------
// Apply: fold WAL records into the index (runs on the WAL commit thread)
// ---------------------------------------------------------------------------

struct IndexApplier {
    catalog: Catalog,
}

impl Apply for IndexApplier {
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
        match record {
            Record::Insert {
                collection,
                ordinal,
                vector,
            } => {
                let coll = self.collection(*collection)?;
                let mut idx = lock(&coll.index)?;
                idx.write_at(*ordinal, vector).map_err(to_io)?;
                idx.advance_applied_lsn(lsn.0);
            }
            Record::Delete {
                collection,
                ordinal,
            } => {
                let coll = self.collection(*collection)?;
                let mut idx = lock(&coll.index)?;
                idx.delete(*ordinal).map_err(to_io)?;
                idx.advance_applied_lsn(lsn.0);
            }
        }
        Ok(())
    }
}

impl IndexApplier {
    fn collection(&self, id: u32) -> io::Result<&Collection> {
        self.catalog.get(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("record references unknown collection {id}"),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Run one checkpoint across all collections, then truncate the WAL.
///
/// The per-index order (a → b → c) and the cross-component order (index durable
/// BEFORE WAL truncate) are the whole crash-safety argument — do not reorder:
///   a. `sync_data`       — vector + tombstone pages durable
///   b. `stage_watermark` — write `(count, last_lsn)` into the spare header slot
///   c. `sync_header`     — that slot's page durable, then it becomes active
///   d/e. truncate the WAL up to the *minimum* durable watermark across
///        collections (a frame at LSN L is redundant only once every collection
///        that might own it is durable past L).
///
/// A crash between any two steps is safe: the index is durable exactly up to its
/// last successfully-synced header, and the WAL still holds everything after
/// that (truncation happens strictly last).
fn run_checkpoint(catalog: &Catalog, wal: &WalHandle) -> io::Result<()> {
    let mut min_durable = u64::MAX;
    for coll in catalog.values() {
        let mut idx = lock(&coll.index)?;
        // Snapshot BEFORE syncing so we never persist a watermark/count that
        // covers bytes the sync didn't flush.
        let (count, last_lsn) = idx.begin_checkpoint();
        idx.sync_data().map_err(to_io)?; // a
        idx.stage_watermark(count, last_lsn).map_err(to_io)?; // b
        idx.sync_header().map_err(to_io)?; // c
        min_durable = min_durable.min(last_lsn);
    }
    // d + e: only frames at or below every collection's durable watermark are
    // safe to drop. `> 0` guards the fresh/never-checkpointed case.
    if min_durable != u64::MAX && min_durable > 0 {
        wal.truncate(min_durable)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Background flusher
// ---------------------------------------------------------------------------

struct Flusher {
    stop: Sender<()>,
    join: JoinHandle<()>,
}

impl Flusher {
    fn spawn(catalog: Catalog, wal: WalHandle, interval: Duration) -> Flusher {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let join = std::thread::Builder::new()
            .name("flats-flusher".into())
            .spawn(move || {
                // Tick until an explicit stop or a dropped handle (any non-
                // Timeout result). A failed checkpoint is best-effort: it's
                // retried next tick and correctness never depends on it (the WAL
                // is the source of truth). The final checkpoint, if any, is
                // driven by `Db::close`.
                while let Err(RecvTimeoutError::Timeout) = stop_rx.recv_timeout(interval) {
                    let _ = run_checkpoint(&catalog, &wal);
                }
            })
            .expect("spawn flusher thread");
        Flusher { stop: stop_tx, join }
    }

    fn stop(self) {
        drop(self.stop);
        let _ = self.join.join();
    }
}

// ---------------------------------------------------------------------------
// Db
// ---------------------------------------------------------------------------

/// The embeddable vector database handle.
pub struct Db {
    catalog: Catalog,
    wal: Option<Wal>,
    flusher: Option<Flusher>,
}

impl Db {
    /// Open (creating if missing) the database rooted at `dir` with the given
    /// collections. Runs WAL recovery — replaying the post-checkpoint tail into
    /// each index — before returning, then starts the background flusher.
    pub fn open(dir: impl AsRef<Path>, collections: &[CollectionConfig], opts: DbOptions) -> Result<Db> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // 1. Open/create every collection index.
        let mut map: HashMap<u32, Collection> = HashMap::with_capacity(collections.len());
        for cfg in collections {
            let path = Self::index_path(dir, cfg.id);
            let index = if path.exists() {
                FlatIndex::open(&path)?
            } else {
                FlatIndex::create(&path, cfg.dim, cfg.capacity)?
            };
            map.insert(
                cfg.id,
                Collection {
                    index: Arc::new(Mutex::new(index)),
                    next_ordinal: AtomicU64::new(0), // re-seeded after recovery
                    dim: cfg.dim.get(),
                    capacity: cfg.capacity,
                },
            );
        }
        let catalog: Catalog = Arc::new(map);

        // 2. Recovery skips everything already folded into the indexes. The safe
        //    global threshold is the minimum durable watermark: a frame at LSN L
        //    is already applied in *its* collection only if every collection is
        //    durable past L. Over-replaying the rest is harmless (idempotent).
        let skip_through = catalog
            .values()
            .map(|c| lock(&c.index).map(|g| g.checkpoint_lsn()))
            .collect::<io::Result<Vec<_>>>()?
            .into_iter()
            .min()
            .unwrap_or(0);

        // 3. Start the WAL; recovery replays the tail into the indexes (via the
        //    applier's Arc clone of the catalog — the same indexes we hold).
        let wal_path = Self::wal_path(dir);
        let applier = IndexApplier {
            catalog: catalog.clone(),
        };
        let wal = Wal::start(&wal_path, applier, skip_through)?;

        // 4. Seed each ordinal allocator from the post-recovery high-water mark.
        for coll in catalog.values() {
            let hw = lock(&coll.index)?.len() as u64;
            coll.next_ordinal.store(hw, Ordering::Release);
        }

        // 5. Background checkpoints.
        let flusher = Flusher::spawn(catalog.clone(), wal.handle(), opts.checkpoint_interval);

        Ok(Db {
            catalog,
            wal: Some(wal),
            flusher: Some(flusher),
        })
    }

    /// Insert `vector` into `collection`. Blocks until durable. Returns the
    /// vector's ordinal (its stable id within the collection).
    pub fn insert(&self, collection: u32, vector: &[f32]) -> Result<Ordinal> {
        let coll = self.collection(collection)?;
        // Validate dim BEFORE we log: a record that can't apply must never reach
        // the durable WAL, or it would fail apply forever on every replay.
        if vector.len() != coll.dim {
            return Err(Error::DimensionMismatch {
                expected: coll.dim,
                got: vector.len(),
            });
        }
        let ordinal = coll.alloc_ordinal()?;
        self.wal_handle()?.append(Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
        })?;
        Ok(Ordinal(ordinal as u32))
    }

    /// Tombstone `ordinal` in `collection`. Blocks until durable. Idempotent.
    pub fn delete(&self, collection: u32, ordinal: u64) -> Result<()> {
        // Validate the collection exists before logging.
        let _ = self.collection(collection)?;
        self.wal_handle()?.append(Record::Delete {
            collection,
            ordinal,
        })?;
        Ok(())
    }

    /// Brute-force top-`k` search within `collection`. Tombstoned vectors are
    /// excluded. Results are most-similar (highest dot product) first.
    pub fn search(&self, collection: u32, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let coll = self.collection(collection)?;
        lock(&coll.index)?.search(query, k)
    }

    /// Force a checkpoint now (index durable + WAL truncated). Mostly for tests
    /// and graceful shutdown; the flusher does this on a timer otherwise.
    pub fn checkpoint(&self) -> Result<()> {
        run_checkpoint(&self.catalog, &self.wal_handle()?)?;
        Ok(())
    }

    /// Graceful shutdown: stop the flusher, take one final checkpoint so reopen
    /// is fast and the WAL is trimmed, then drain and join the WAL thread.
    pub fn close(mut self) -> Result<()> {
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        let wal = self.wal.take().expect("wal present until close");
        run_checkpoint(&self.catalog, &wal.handle())?;
        wal.shutdown();
        Ok(())
    }

    fn collection(&self, id: u32) -> Result<&Collection> {
        self.catalog
            .get(&id)
            .ok_or(Error::UnknownCollection { id })
    }

    fn wal_handle(&self) -> Result<WalHandle> {
        self.wal
            .as_ref()
            .map(|w| w.handle())
            .ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::BrokenPipe, "db is closing")))
    }

    fn index_path(dir: &Path, id: u32) -> PathBuf {
        dir.join(format!("collection-{id}.idx"))
    }

    fn wal_path(dir: &Path) -> PathBuf {
        dir.join("wal.log")
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        // If the caller didn't `close()`, still tear down cleanly: stop the
        // flusher and join the WAL thread. We skip the final checkpoint here —
        // the WAL is durable, so the next open just replays a longer tail.
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        if let Some(wal) = self.wal.take() {
            wal.shutdown();
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn lock<T>(m: &Mutex<T>) -> io::Result<MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| io::Error::other("index mutex poisoned"))
}

fn to_io(e: Error) -> io::Error {
    match e {
        Error::Io(io) => io,
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(id: u32, dim: usize, capacity: usize) -> CollectionConfig {
        CollectionConfig {
            id,
            dim: NonZeroUsize::new(dim).unwrap(),
            capacity,
        }
    }

    // Large interval => the background flusher never fires; tests drive
    // checkpoints explicitly for determinism.
    fn manual_opts() -> DbOptions {
        DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
        }
    }

    #[test]
    fn insert_search_delete_e2e() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();

        let a = db.insert(0, &[1.0, 0.0]).unwrap();
        let b = db.insert(0, &[2.0, 0.0]).unwrap();
        let _c = db.insert(0, &[0.0, 1.0]).unwrap();
        assert_eq!(a, Ordinal(0));
        assert_eq!(b, Ordinal(1));

        let hits = db.search(0, &[1.0, 0.0], 3).unwrap();
        assert_eq!(hits[0].id, Ordinal(1)); // [2,0] best

        // Delete the winner; it must vanish from results.
        db.delete(0, 1).unwrap();
        let hits = db.search(0, &[1.0, 0.0], 3).unwrap();
        assert!(hits.iter().all(|h| h.id != Ordinal(1)));
        assert_eq!(hits[0].id, Ordinal(0)); // [1,0] now best

        db.close().unwrap();
    }

    #[test]
    fn reopen_recovers_full_state() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
            db.insert(0, &[1.0, 0.0]).unwrap();
            db.insert(0, &[2.0, 0.0]).unwrap();
            db.checkpoint().unwrap(); // durable + WAL truncated
            db.insert(0, &[3.0, 0.0]).unwrap(); // lives only in the WAL tail
            db.close().unwrap();
        }
        // Reopen: checkpointed prefix comes from the index, the tail from the WAL.
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        let hits = db.search(0, &[1.0, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, Ordinal(2)); // [3,0]
        // The allocator resumed past the recovered high-water mark.
        let next = db.insert(0, &[4.0, 0.0]).unwrap();
        assert_eq!(next, Ordinal(3));
        db.close().unwrap();
    }

    #[test]
    fn capacity_is_enforced_before_logging() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 1, 2)], manual_opts()).unwrap();
        db.insert(0, &[1.0]).unwrap();
        db.insert(0, &[2.0]).unwrap();
        assert!(matches!(
            db.insert(0, &[3.0]),
            Err(Error::CapacityExceeded { .. })
        ));
        db.close().unwrap();
    }

    #[test]
    fn unknown_collection_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 8)], manual_opts()).unwrap();
        assert!(matches!(
            db.insert(7, &[1.0, 2.0]),
            Err(Error::UnknownCollection { id: 7 })
        ));
        db.close().unwrap();
    }
}
