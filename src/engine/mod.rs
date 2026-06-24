//! Execution layer: the thin engine that wires the WAL to the flat index.
//!
//! This is the minimal v1 executor — no SQL, no planner, no metadata filtering.
//! It owns the durability story end to end.
//!
//! # SWMR ownership
//! Each collection's index is split into a single `Writer` and cloneable
//! `Reader`s (see `index.rs`). The `Writer` lives in the `IndexApplier` on the
//! WAL commit thread — the one and only mutator. The catalog holds the `Reader`s
//! for the (lock-free, parallel) read path. This is why checkpointing rides on
//! the WAL thread (`Apply::checkpoint`): all index mutation must stay on the
//! thread that owns the `Writer`.
//!
//!   * WRITE PATH (`insert`/`delete`): allocate the next ordinal from the
//!     collection's high-water mark, build a *positional* WAL record carrying
//!     that ordinal, and block until the WAL has made it durable AND applied it
//!     into the index. The ordinal is assigned exactly once, on the logging
//!     side; apply/replay never recompute it.
//!
//!   * APPLY (`IndexApplier`): the WAL thread calls this post-fsync. It folds a
//!     record into the index's `Writer` with `write_at`/`delete` (both
//!     idempotent) so that replaying an already-applied record is a no-op.
//!
//!   * RECOVERY: on open, each collection's durable checkpoint watermark
//!     (`Writer::checkpoint_lsn`) tells the WAL which prefix is already in the
//!     index; recovery replays only the tail.
//!
//!   * CHECKPOINT: the `Flusher` thread pokes the WAL commit thread on a timer;
//!     that thread runs `Apply::checkpoint` (index durable) then truncates the
//!     log, in a strict crash-safe order.
//!
//! Multi-collection catalog persistence is out of scope here: the caller hands
//! the collection set to `open` each time. The data structures are already
//! keyed by collection id so the catalog can be layered on without reshaping.

use std::collections::HashMap;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::index::index::{FlatIndex, Ordinal, Reader, SearchResult, Writer};
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

/// One open collection on the read/executor side: the index's `Reader` plus the
/// live ordinal allocator. The matching `Writer` lives in the `IndexApplier` on
/// the WAL commit thread — the single mutator.
struct Collection {
    reader: Reader,
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

/// Shared, read-only-after-open map of collections (readers + allocators).
/// Cloned (Arc) into the engine's read path; never holds a writer.
type Catalog = Arc<HashMap<u32, Collection>>;

// ---------------------------------------------------------------------------
// Apply: fold WAL records into the index (runs on the WAL commit thread)
// ---------------------------------------------------------------------------

/// Owns every collection's single `Writer`. Lives entirely on the WAL commit
/// thread, which is what upholds the single-writer invariant: only this thread,
/// via these uniquely-owned `Writer`s, ever mutates an index.
struct IndexApplier {
    writers: HashMap<u32, Writer>,
}

impl Apply for IndexApplier {
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
        match record {
            Record::Insert {
                collection,
                ordinal,
                vector,
            } => {
                let w = self.writer(*collection)?;
                w.write_at(*ordinal, vector).map_err(to_io)?;
                w.advance_applied_lsn(lsn.0);
            }
            Record::Delete {
                collection,
                ordinal,
            } => {
                let w = self.writer(*collection)?;
                w.delete(*ordinal).map_err(to_io)?;
                w.advance_applied_lsn(lsn.0);
            }
        }
        Ok(())
    }

    /// Checkpoint every collection, returning the WAL truncation point.
    ///
    /// Per-index order (a → b → c) is the crash-safety argument — do not reorder:
    ///   a. `sync_data`       — vector + tombstone pages durable
    ///   b. `stage_watermark` — write `(count, last_lsn)` into the spare slot
    ///   c. `sync_header`     — that slot's page durable, then it becomes active
    /// The commit thread then truncates the WAL up to the returned LSN (strictly
    /// after this), so a crash between any two steps stays safe. The truncation
    /// point is the *minimum* durable watermark across collections: a frame at
    /// LSN L is redundant only once every collection that might own it is durable
    /// past L (see the per-collection note in `Db::open`).
    fn checkpoint(&mut self) -> io::Result<Option<u64>> {
        let mut min_durable = u64::MAX;
        for w in self.writers.values_mut() {
            // Snapshot BEFORE syncing so we never persist a watermark/count that
            // covers bytes the sync didn't flush.
            let (count, last_lsn) = w.begin_checkpoint();
            w.sync_data().map_err(to_io)?; // a
            w.stage_watermark(count, last_lsn).map_err(to_io)?; // b
            w.sync_header().map_err(to_io)?; // c
            min_durable = min_durable.min(last_lsn);
        }
        Ok(if min_durable == u64::MAX || min_durable == 0 {
            None
        } else {
            Some(min_durable)
        })
    }
}

impl IndexApplier {
    fn writer(&mut self, id: u32) -> io::Result<&mut Writer> {
        self.writers.get_mut(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("record references unknown collection {id}"),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Background flusher
// ---------------------------------------------------------------------------

struct Flusher {
    stop: Sender<()>,
    join: JoinHandle<()>,
}

impl Flusher {
    fn spawn(wal: WalHandle, interval: Duration) -> Flusher {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let join = std::thread::Builder::new()
            .name("flats-flusher".into())
            .spawn(move || {
                // Tick until an explicit stop or a dropped handle (any non-
                // Timeout result). The checkpoint runs on the WAL commit thread;
                // we just ask for it. Best-effort: a failed checkpoint is retried
                // next tick and correctness never depends on it (the WAL is the
                // source of truth). The final checkpoint, if any, is driven by
                // `Db::close`.
                while let Err(RecvTimeoutError::Timeout) = stop_rx.recv_timeout(interval) {
                    let _ = wal.checkpoint();
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

        // 1. Open/create every collection index, splitting each into its Reader
        //    (kept in the catalog) and Writer (handed to the applier).
        let mut map: HashMap<u32, Collection> = HashMap::with_capacity(collections.len());
        let mut writers: HashMap<u32, Writer> = HashMap::with_capacity(collections.len());
        for cfg in collections {
            let path = Self::index_path(dir, cfg.id);
            let (writer, reader) = if path.exists() {
                FlatIndex::open(&path)?
            } else {
                FlatIndex::create(&path, cfg.dim, cfg.capacity)?
            };
            map.insert(
                cfg.id,
                Collection {
                    reader,
                    next_ordinal: AtomicU64::new(0), // re-seeded after recovery
                    dim: cfg.dim.get(),
                    capacity: cfg.capacity,
                },
            );
            writers.insert(cfg.id, writer);
        }
        let catalog: Catalog = Arc::new(map);

        // 2. Recovery skips everything already folded into the indexes.
        //
        //    CORRECTNESS — skip-through is conceptually PER-COLLECTION: a frame
        //    at LSN L may be dropped only if *its own* collection is durable
        //    past L. The right model is `skip iff L <= last_lsn[frame.collection]`.
        //
        //    With a single shared WAL and no per-frame collection routing in
        //    recovery yet, we use a conservative global *minimum* across
        //    collections: anything <= the slowest collection's watermark is
        //    durable everywhere, so skipping it is always safe; frames above it
        //    are replayed and idempotent apply absorbs any that a faster
        //    collection had already seen. This only over-replays — never skips a
        //    record a collection still needs.
        //
        //    DO NOT change this to a global `max`: that would skip frames for a
        //    lagging collection that has NOT durably applied them — silent data
        //    loss. When the catalog-wiring phase lands, replace this scalar with
        //    a real per-collection watermark keyed on each frame's collection id;
        //    never collapse it to one global high-water mark.
        let skip_through = writers
            .values()
            .map(|w| w.checkpoint_lsn())
            .min()
            .unwrap_or(0);

        // 3. Start the WAL; recovery replays the tail into the writers (which the
        //    applier owns). The catalog's readers observe the same inners.
        let wal_path = Self::wal_path(dir);
        let applier = IndexApplier { writers };
        let wal = Wal::start(&wal_path, applier, skip_through)?;

        // 4. Seed each ordinal allocator from the post-recovery high-water mark
        //    (read via the reader — recovery has already run on this thread).
        for coll in catalog.values() {
            coll.next_ordinal
                .store(coll.reader.len() as u64, Ordering::Release);
        }

        // 5. Background checkpoints (the flusher just pokes the commit thread).
        let flusher = Flusher::spawn(wal.handle(), opts.checkpoint_interval);

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
        match self.wal_handle()?.append(Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
        }) {
            Ok(_lsn) => Ok(Ordinal(ordinal as u32)),
            Err(e) => {
                // The append never became durable, so there is nothing to log —
                // but the allocator already burned `ordinal`, leaving a
                // zero-filled slot that a later insert will pull into search
                // range and surface with score 0 (a phantom). Tombstone it in
                // memory: a pure bit-flip, NOT a WAL Delete (the ordinal was
                // never durable, so there is nothing to replay). Best-effort —
                // if the lock is poisoned we still surface the original error.
                //
                // TODO(durability): this in-memory tombstone is lost on a crash
                // before the next checkpoint flushes the bitset. It fully covers
                // the common case (a WAL append failure is effectively terminal —
                // no later append succeeds, so the high-water mark never advances
                // past `ordinal` and the gap is unreachable). The residual hole:
                // a *transient* append failure, followed by a *successful* insert
                // (which advances the high-water mark past `ordinal`), followed by
                // a crash before any checkpoint — recovery would then rebuild the
                // high-water mark over the gap with no tombstone, resurfacing the
                // phantom. Closing it requires making the tombstone durable
                // (e.g. logging a WAL `Delete { ordinal }` here, or persisting a
                // "burned ordinals" set), which we deliberately deferred. Revisit
                // if WAL failures ever become recoverable/retryable mid-session.
                //
                // The Writer lives on the WAL thread, so we flip the bit through
                // the Reader — sound because the tombstone bitset is atomic (see
                // `Reader::tombstone_uncommitted`).
                let _ = coll.reader.tombstone_uncommitted(ordinal);
                Err(Error::from(e))
            }
        }
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
    /// excluded. Results are most-similar (highest dot product) first. Runs
    /// lock-free against the reader; concurrent searches do not serialize.
    pub fn search(&self, collection: u32, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let coll = self.collection(collection)?;
        coll.reader.search(query, k)
    }

    /// A cloneable read handle for `collection`, for issuing searches from other
    /// threads in parallel. Returns `None` for an unknown collection.
    pub fn reader(&self, collection: u32) -> Option<Reader> {
        self.catalog.get(&collection).map(|c| c.reader.clone())
    }

    /// Force a checkpoint now (index durable + WAL truncated). Runs on the WAL
    /// commit thread. Mostly for tests and graceful shutdown; the flusher does
    /// this on a timer otherwise.
    pub fn checkpoint(&self) -> Result<()> {
        self.wal_handle()?.checkpoint()?;
        Ok(())
    }

    /// Graceful shutdown: stop the flusher, take one final checkpoint so reopen
    /// is fast and the WAL is trimmed, then drain and join the WAL thread.
    pub fn close(mut self) -> Result<()> {
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        let wal = self.wal.take().expect("wal present until close");
        wal.handle().checkpoint()?; // final checkpoint on the commit thread
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

    #[test]
    fn failed_insert_does_not_leak_phantom_ordinal() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0]).unwrap(); // ord 0
        db.insert(0, &[1.0, 0.0]).unwrap(); // ord 1

        // Force the next durability to fail; its ordinal (2) is burned by the
        // allocator but its slot is never written.
        db.wal.as_ref().unwrap().fail_next_append();
        let failed = db.insert(0, &[9.0, 9.0]);
        assert!(failed.is_err(), "durability failure must surface to caller");

        // A subsequent successful insert takes ordinal 3 and pushes the
        // high-water mark to 4, pulling the burned slot 2 (zero-filled) into
        // search range. It must NOT appear — the error path tombstoned it.
        let later = db.insert(0, &[1.0, 0.0]).unwrap();
        assert_eq!(later, Ordinal(3));

        let hits = db.search(0, &[1.0, 1.0], 64).unwrap();
        let ids: std::collections::BTreeSet<u32> = hits.iter().map(|h| h.id.0).collect();
        assert!(!ids.contains(&2), "burned ordinal 2 must stay hidden");
        assert_eq!(
            ids,
            [0u32, 1, 3].into_iter().collect(),
            "only the real ordinals are visible"
        );
        db.close().unwrap();
    }
}
