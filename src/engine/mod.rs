//! Execution layer: the thin engine that wires the WAL to the indexes —
//! per collection: the flat vector index, the metadata index (4a), and the
//! tuple store (4b), all fed from one durable record stream.
//!
//! This is the minimal v1 executor — no SQL, no planner, no predicate AST.
//! The metadata read path is primitive plumbing (`metadata_reader` /
//! `tuple_reader`); the SEARCH..WHERE executor composes it later.
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
//! The collection set is durable: `catalog.snap` in the root dir registers
//! every collection's config (id, name, dim, capacity, schema), stamped with
//! the LSN it reflects. `Db::open` merges caller-supplied configs into it at
//! open time (first mention registers; `&[]` and everything re-emerges); at
//! RUNTIME, `create_collection` logs `Record::CreateCollection` through the
//! WAL like any other mutation — DDL commits the same way DML does, and
//! replays idempotently (apply checks the catalog before materializing).
//! The catalog participates in checkpoint/recovery as a fourth watermark
//! alongside the three per-collection stores.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

pub mod cursor;

use crate::engine::cursor::Cursor;
use crate::error::{Error, Result};
use crate::index::index::{FlatIndex, Ordinal, Reader, SearchResult, Writer};
use roaring::RoaringBitmap;

use crate::metadata::common::{ColumnId, Row, Schema, Value};
use crate::metadata::index::MetadataIndex;
use crate::metadata::tuples::TupleStore;
use crate::metadata::{crc32, index as meta, tuples, write_snapshot_atomic};
use crate::wal::wal::{Apply, Bitmap, Install, Lsn, Outcome, Record, Wal, WalHandle};

/// Ceiling on the installed payload of one `Record::Mutate`, in bytes.
///
/// One statement is one record, so the whole thing has to fit in memory on the
/// commit thread and on every future replay. 256 MiB is far above any
/// reasonable predicate-scoped UPDATE and far below anything that threatens a
/// small deployment's memory. See `Error::StatementTooLarge`.
pub const MAX_INSTALL_BYTES: usize = 256 * 1024 * 1024;

/// Re-exported from `metadata::common` (it moved there in Phase 6 so
/// `Record::CreateCollection` can carry it — the WAL may only depend on plain
/// data types). Registered on first mention (a `Db::open` config or a
/// `create_collection` call) and persisted in `catalog.snap` from then on.
pub use crate::metadata::common::CollectionConfig;

/// A collection's stable numeric identity — what WAL records key on and what
/// every `Db` method takes. Assigned once (open-time registration or
/// `create_collection`) and never reused. [`Db::collection_id`] is the bridge
/// from the NAME compiled bytecode carries to this.
///
/// An ALIAS, not a newtype: the id is `u32` in `CollectionConfig` and inside the
/// bincode-encoded WAL frame, so wrapping it is a change to the persisted
/// format's type surface, not a local rename. The alias exists so signatures can
/// SAY collection id where they currently say `u32`; promoting it to a real
/// newtype is a separate, deliberate change.
pub type CollectionId = u32;

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
    /// Read handles for the two metadata stores (4a/4b). The matching writers
    /// live in the `IndexApplier` next to the flat one.
    meta: meta::Reader,
    tuple: tuples::Reader,
    /// The registered config (dim, capacity, schema) — the same value that is
    /// persisted in catalog.snap. `insert` validates rows against its schema
    /// before logging.
    config: CollectionConfig,
    /// Next ordinal to hand out. Seeded from the index high-water after
    /// recovery; advanced once per insert.
    next_ordinal: AtomicU64,
    /// Version-tagged liveness snapshots for this collection. The applier
    /// bumps its version once per liveness-changing record; readers resolve an
    /// `Arc<LiveSet>` and carry it for the life of a query.
    ///
    /// Shared with this collection's `CollectionWriters` on the WAL thread,
    /// which is the only side that bumps.
    live: Arc<meta::LiveHandle>,
}

impl Collection {
    /// Raise the allocator's high-water mark to `at_least`, so ordinals a
    /// statement placed itself are never handed out again.
    ///
    /// `Db::mutate` installs rows at ordinals chosen on the logging side rather
    /// than drawn from `alloc_ordinal`, so without this the next `insert` would
    /// allocate straight over a freshly installed row. Monotonic and racy-safe:
    /// a concurrent `alloc_ordinal` either loses the CAS and retries above the
    /// new mark, or wins and is already above it.
    fn reserve_through(&self, at_least: u64) {
        let mut cur = self.next_ordinal.load(Ordering::Acquire);
        while cur < at_least {
            match self.next_ordinal.compare_exchange_weak(
                cur,
                at_least,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Reserve `count` CONSECUTIVE ordinals, or fail if they would not fit.
    ///
    /// One CAS for the whole run, which is what makes an UPDATE's capacity
    /// check meaningful: a per-row loop could pass its check, allocate half
    /// the rows, and then fail — leaving a statement that must be abandoned
    /// with ordinals already burned. Either the whole run is reserved or
    /// nothing is, and the check happens before the record is built at all.
    fn alloc_ordinals(&self, count: u64) -> Result<u64> {
        loop {
            let cur = self.next_ordinal.load(Ordering::Acquire);
            let end = cur
                .checked_add(count)
                .filter(|end| *end <= self.config.capacity as u64)
                .ok_or(Error::CapacityExceeded {
                    capacity: self.config.capacity,
                })?;
            if self
                .next_ordinal
                .compare_exchange_weak(cur, end, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(cur);
            }
        }
    }

    /// Reserve the next ordinal, or fail if the collection is at capacity.
    /// CAS loop so a full collection never overshoots the high-water mark.
    fn alloc_ordinal(&self) -> Result<u64> {
        loop {
            let cur = self.next_ordinal.load(Ordering::Acquire);
            if cur >= self.config.capacity as u64 {
                return Err(Error::CapacityExceeded {
                    capacity: self.config.capacity,
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

/// The live collection map. Entries are `Arc<Collection>` so a published
/// snapshot shares the allocators/readers with every previous snapshot.
type CatalogMap = HashMap<u32, Arc<Collection>>;

/// Shared, VERSIONED map of collections. The read path takes a cheap snapshot
/// (`Arc` clone under a brief read lock) and never holds the lock while
/// working; the WAL commit thread publishes a whole new map when a
/// CreateCollection applies (copy-on-write — clones of `Arc`s, so it's cheap
/// and DDL-rare). It never holds a writer.
type SharedCatalog = Arc<RwLock<Arc<CatalogMap>>>;

fn catalog_snapshot(catalog: &SharedCatalog) -> Arc<CatalogMap> {
    catalog.read().unwrap_or_else(|e| e.into_inner()).clone()
}

fn collection_dir(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("collection-{id}"))
}

/// Open (or create) all three stores of one collection. Every artifact is
/// open-if-exists-else-create, which is what makes both `Db::open` and
/// CreateCollection replay idempotent at the file level: a half-materialized
/// collection from a crash is absorbed. Returns the writers, the read-side
/// `Collection`, and the min durable watermark across the three stores.
fn open_collection(dir: &Path, cfg: &CollectionConfig) -> Result<(CollectionWriters, Collection, u64)> {
    let cdir = collection_dir(dir, cfg.id);
    std::fs::create_dir_all(&cdir)?;

    let idx_path = cdir.join("vectors.idx");
    let (flat_w, flat_r) = if idx_path.exists() {
        FlatIndex::open(&idx_path)?
    } else {
        FlatIndex::create(&idx_path, cfg.schema.vector().dim, cfg.capacity)?
    };
    // The metadata stores' open() treats a corrupt snapshot as "empty at
    // Lsn(0)" — the WAL replays them back to life. A corrupt store's 0 drags
    // the min watermark down, forcing the full-tail replay that heals it
    // while the healthy stores absorb the replay idempotently.
    let (meta_w, meta_r, meta_lsn) = MetadataIndex::open_or_create(&cdir, cfg.schema.clone())?;
    let (tuple_w, tuple_r, tuple_lsn) = TupleStore::open_or_create(&cdir, cfg.schema.clone())?;

    let watermark = flat_w.checkpoint_lsn().min(meta_lsn.0).min(tuple_lsn.0);
    // ONE handle, shared by both sides. The applier could reach the read-side
    // Collection through the catalog instead, but that is an RwLock read plus a
    // HashMap lookup per record, on the write path, forever.
    let live = Arc::new(meta_r.live_handle());
    Ok((
        CollectionWriters {
            flat: flat_w,
            meta: meta_w,
            tuple: tuple_w,
            live: Arc::clone(&live),
        },
        Collection {
            live,
            reader: flat_r,
            meta: meta_r,
            tuple: tuple_r,
            config: cfg.clone(),
            next_ordinal: AtomicU64::new(0), // re-seeded after recovery
        },
        watermark,
    ))
}

// ---------------------------------------------------------------------------
// Apply: fold WAL records into the index (runs on the WAL commit thread)
// ---------------------------------------------------------------------------

/// One collection's full mutation surface: the flat vector index plus the two
/// metadata stores (4a/4b). All three writers travel together — every record
/// fans out to all of them.
struct CollectionWriters {
    flat: Writer,
    meta: meta::Writer,
    tuple: tuples::Writer,
    /// Liveness version counter, shared with the read-side `Collection`. Bumped
    /// once per liveness-changing record — the WAL thread is the only bumper,
    /// which is the same single-writer argument the three writers above rest on.
    live: Arc<meta::LiveHandle>,
}

/// Owns every collection's writers. Lives entirely on the WAL commit thread,
/// which is what upholds the single-writer invariant: only this thread, via
/// these uniquely-owned writers, ever mutates an index. (The Mutex inside the
/// metadata stores exists for their *readers*, not for a second mutator.)
///
/// Since Phase 6 it also owns the CATALOG's write side: CreateCollection
/// applies here (materialize stores → persist catalog.snap → publish to the
/// shared read map), and the catalog participates in checkpoint/recovery as a
/// FOURTH store with its own watermark — see `catalog_watermark`.
struct IndexApplier {
    dir: PathBuf,
    writers: HashMap<u32, CollectionWriters>,
    /// What catalog.snap holds: every registered config, sorted by id.
    registry: Vec<CollectionConfig>,
    /// Shared with the read path; a successful CreateCollection publishes a
    /// new snapshot into it.
    catalog: SharedCatalog,
    /// Highest LSN successfully applied to everything (the catalog's
    /// in-memory watermark; stores keep their own). Seeded with `skip_through`
    /// at open — everything at or below it is already durable everywhere.
    last_applied: u64,
    /// First CreateCollection LSN that failed to materialize this session.
    /// Freezes `catalog_watermark` below it, so neither truncation nor the
    /// persisted catalog stamp can ever claim a create that didn't happen —
    /// restart replays it.
    ///
    /// Belt to the halt's braces, and deliberately kept: any apply failure now
    /// poisons the log (`wal::Poison`), which refuses every later write and
    /// every checkpoint, so no watermark can advance past the hole in the
    /// first place. This stays because it is the local, checkable version of
    /// that argument — a create has no store watermark of its own to stall,
    /// unlike an insert, so without it the invariant would rest entirely on a
    /// guarantee made two modules away.
    failed_create: Option<u64>,
}

impl Apply for IndexApplier {
    /// Fold one record into all three subsystems.
    ///
    /// Every step is idempotent (`write_at` positional, `insert_row`
    /// set-semantics, `write_row` overwrite-same, deletes are bit-clears/
    /// markers), so an error mid-fan-out needs no rollback: the record is
    /// durable and replay retries — partial state is legal between crash and
    /// replay. Watermarks advance LAST, only after every write succeeded: a
    /// watermark must never claim an apply that didn't happen, or checkpoint
    /// could persist it and recovery would skip the record forever.
    ///
    /// That rule is necessary but not sufficient on its own, because watermarks
    /// are a single high-water number per store: if this record fails and a
    /// LATER one succeeds, the later `max()` sweeps right over the gap. What
    /// closes it is the caller — an `Err` from here poisons the WAL, so there
    /// is never a later record to do the sweeping. See `wal::Poison`.
    ///
    /// The liveness version is bumped once per liveness-changing record
    /// (`Insert`/`Delete`, not DDL — a new collection has no rows), and
    /// unconditionally: an idempotent replay of a delete bumps too. Over-
    /// invalidation costs one re-materialization; the branch that avoided it
    /// would cost a correctness argument.
    ///
    /// Fan-out ORDER is load-bearing, not incidental: a row becomes visible
    /// when `meta.insert_row` sets its live bit, so every store a reader can
    /// consult must already hold it by then. Inserts publish liveness last,
    /// deletes publish it first. Do not reorder — `concurrent_stores.rs`
    /// asserts the resulting property directly.
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<Outcome> {
        match record {
            Record::Insert {
                collection,
                ordinal,
                vector,
                metadata,
            } => {
                let w = self.writers(*collection)?;
                let ord32 = ordinal32(*ordinal)?;
                // DATA FIRST, LIVENESS LAST. `meta.insert_row` is what makes
                // the ordinal enumerable, so it must not run until every store
                // that a reader will consult for that row already holds it —
                // otherwise a snapshot can admit a row the tuple store has
                // never seen, and a LiveSet is immutable, so that reader is
                // wrong for as long as it holds the snapshot.
                w.flat.write_at(*ordinal, vector).map_err(to_io)?;
                w.tuple.write_row(ord32, metadata).map_err(to_io)?;
                w.meta.insert_row(ord32, metadata).map_err(to_io)?;
                w.live.bump();
                w.flat.advance_applied_lsn(lsn.0);
                w.meta.advance_applied_lsn(lsn);
                w.tuple.advance_applied_lsn(lsn);
            }
            Record::Delete {
                collection,
                ordinal,
            } => {
                let w = self.writers(*collection)?;
                let ord32 = ordinal32(*ordinal)?;
                // Retiring a row clears ONE bit. The vector stays in the flat
                // index and the values stay in the tuple store, so a reader
                // holding an older snapshot can still read the row it was
                // promised; compaction reclaims both later. Liveness has one
                // owner, so a delete has exactly one place to happen.
                w.meta.remove_row(ord32).map_err(to_io)?;
                w.live.bump();
                w.flat.advance_applied_lsn(lsn.0);
                w.meta.advance_applied_lsn(lsn);
                w.tuple.advance_applied_lsn(lsn);
            }
            Record::CreateCollection { config } => {
                self.apply_create(lsn, config)?;
            }
            Record::Mutate {
                collection,
                retire,
                install,
            } => {
                let w = self.writers(*collection)?;
                let rows: Vec<(Ordinal, &Row)> = install
                    .iter()
                    .map(|r| ordinal32(r.ordinal).map(|o| (o, &r.metadata)))
                    .collect::<io::Result<_>>()?;

                // STALENESS, checked before anything is written. A statement
                // built from rows that have since been retired by someone else
                // must not land, or one row becomes two — see
                // `Writer::is_stale`. Deliberately NOT an error: the record is
                // durable and the database is healthy, so halting on it (see
                // `wal::Poison`) would be wrong. The waiter is told, and
                // retries.
                let installed: Vec<Ordinal> = rows.iter().map(|(o, _)| *o).collect();
                if w.meta.is_stale(&retire.0, &installed) {
                    // The watermarks still advance: a skipped record IS fully
                    // reflected (it changed nothing), and the verdict is
                    // deterministic, so replay reaches it again and skips it
                    // again. Stalling them here would wedge truncation behind
                    // a record that will never do anything.
                    w.flat.advance_applied_lsn(lsn.0);
                    w.meta.advance_applied_lsn(lsn);
                    w.tuple.advance_applied_lsn(lsn);
                    self.last_applied = self.last_applied.max(lsn.0);
                    return Ok(Outcome::Conflicted);
                }

                // DATA FIRST, LIVENESS LAST — the same rule as `Insert`, but
                // now spanning a whole statement. Every installed row must be
                // readable in both stores before anything publishes it, or a
                // snapshot could admit a row the tuple store has never seen.
                for row in install {
                    let ord32 = ordinal32(row.ordinal)?;
                    w.flat.write_at(row.ordinal, &row.vector).map_err(to_io)?;
                    w.tuple.write_row(ord32, &row.metadata).map_err(to_io)?;
                }
                // ONE publish for the whole statement. `apply_mutation` holds
                // the metadata lock across every retire and install precisely
                // so no reader can materialize a snapshot mid-statement — see
                // its doc. A loop of insert_row/remove_row here would be
                // correct on disk and wrong in memory.
                w.meta.apply_mutation(&retire.0, &rows).map_err(to_io)?;
                w.live.bump();
                w.flat.advance_applied_lsn(lsn.0);
                w.meta.advance_applied_lsn(lsn);
                w.tuple.advance_applied_lsn(lsn);
            }
        }
        // The catalog watermark advances only on success — a `?` above skips
        // this, so a failed record can never be claimed as reflected.
        self.last_applied = self.last_applied.max(lsn.0);
        Ok(Outcome::Applied)
    }

    /// Checkpoint every store of every collection, returning the WAL
    /// truncation point: the MINIMUM durable watermark across all of them.
    ///
    /// Flat-index per-store order (a → b → c) is the crash-safety argument —
    /// do not reorder:
    ///   a. `sync_data`       — vector + tombstone pages durable
    ///   b. `stage_watermark` — write `(count, last_lsn)` into the spare slot
    ///   c. `sync_header`     — that slot's page durable, then it becomes active
    /// The metadata stores use serialize-and-rename snapshots (their own
    /// crash-safe protocol). The commit thread truncates the WAL strictly
    /// AFTER this returns, so a crash between any two steps stays safe.
    ///
    /// If ANY store's checkpoint fails the whole call errs and the commit
    /// loop skips truncation this round (flusher retries next tick). Don't
    /// get clever with partial truncation: "some snapshots advanced, then
    /// truncate to their min" is only safe because a failed store's OLD
    /// persisted watermark is still in the min — erroring out keeps that
    /// trivially true.
    fn checkpoint(&mut self) -> io::Result<Option<u64>> {
        let mut min_durable = u64::MAX;
        for w in self.writers.values_mut() {
            // Snapshot BEFORE syncing so we never persist a watermark/count that
            // covers bytes the sync didn't flush.
            let (count, flat_lsn) = w.flat.begin_checkpoint();
            w.flat.sync_data().map_err(to_io)?; // a
            w.flat.stage_watermark(count, flat_lsn).map_err(to_io)?; // b
            w.flat.sync_header().map_err(to_io)?; // c

            let meta_lsn = w.meta.applied_lsn();
            w.meta.checkpoint(meta_lsn).map_err(to_io)?;
            let tuple_lsn = w.tuple.applied_lsn();
            w.tuple.checkpoint(tuple_lsn).map_err(to_io)?;

            min_durable = min_durable.min(flat_lsn).min(meta_lsn.0).min(tuple_lsn.0);
        }

        // The catalog is the FOURTH store: persist the registry stamped with
        // its watermark and fold that stamp into the min. This is what stops
        // truncation from ever dropping a CreateCollection record whose
        // registration isn't durably on disk yet.
        let cat_lsn = self.catalog_watermark();
        persist_catalog(&self.dir, &self.registry, cat_lsn).map_err(to_io)?;
        min_durable = min_durable.min(cat_lsn);

        Ok(if min_durable == u64::MAX || min_durable == 0 {
            None
        } else {
            Some(min_durable)
        })
    }
}

impl IndexApplier {
    fn writers(&mut self, id: u32) -> io::Result<&mut CollectionWriters> {
        self.writers.get_mut(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("record references unknown collection {id}"),
            )
        })
    }

    /// The LSN through which the in-memory catalog provably reflects every
    /// record — in particular, every CreateCollection at or below it is
    /// registered. Frozen below the first failed create (see the field doc).
    fn catalog_watermark(&self) -> u64 {
        match self.failed_create {
            Some(l) => self.last_applied.min(l.saturating_sub(1)),
            None => self.last_applied,
        }
    }

    /// Apply one CreateCollection. Idempotent per the spec: if the id or name
    /// already exists it is a NO-OP, never an error — recovery redelivers
    /// this record on every restart until a checkpoint folds it away, and an
    /// error here would make `Db::open` fail forever.
    ///
    /// All-or-nothing visibility: writers/registry/read-catalog are published
    /// only after BOTH the stores are materialized AND catalog.snap is
    /// persisted. On any failure the record stays > `catalog_watermark`, so
    /// it survives truncation and restart replays it.
    fn apply_create(&mut self, lsn: Lsn, config: &CollectionConfig) -> io::Result<()> {
        if self.writers.contains_key(&config.id)
            || self.registry.iter().any(|c| c.name == config.name)
        {
            // Pure replay redelivers the identical config; anything else is a
            // collision that can only come from mixing open()-registration
            // with an unreplayed WAL create. Surface it, but stay a no-op.
            if !self.registry.contains(config) {
                eprintln!(
                    "flats: CreateCollection (lsn {}, id {}, name {:?}) collides with an \
                     existing collection; skipped",
                    lsn.0, config.id, config.name
                );
            }
            return Ok(());
        }

        let materialized = (|| -> Result<(CollectionWriters, Collection)> {
            let (writers, collection, _watermark) = open_collection(&self.dir, config)?;
            // The new collection-{id}/ entry itself must survive a crash: the
            // stores fsync'd their own files, but the parent dir entry is
            // separate. (Replay would heal it anyway; this just makes the
            // common case not need healing.)
            std::fs::File::open(&self.dir)?.sync_all()?;

            // Register durably BEFORE publishing anything in memory.
            let mut registry = self.registry.clone();
            registry.push(config.clone());
            registry.sort_by_key(|c| c.id);
            persist_catalog(&self.dir, &registry, self.catalog_watermark())?;
            self.registry = registry;
            Ok((writers, collection))
        })();

        let (writers, collection) = match materialized {
            Ok(v) => v,
            Err(e) => {
                self.failed_create.get_or_insert(lsn.0);
                return Err(to_io(e));
            }
        };

        self.writers.insert(config.id, writers);
        let mut guard = self.catalog.write().unwrap_or_else(|e| e.into_inner());
        let mut map: CatalogMap = (**guard).clone();
        map.insert(config.id, Arc::new(collection));
        *guard = Arc::new(map);
        Ok(())
    }
}

/// Records carry `ordinal: u64`; the metadata ordinal space is u32 by
/// construction (RoaringBitmap). Capacity bounds make overflow unreachable —
/// check anyway, never truncate.
fn ordinal32(ordinal: u64) -> io::Result<Ordinal> {
    u32::try_from(ordinal).map(Ordinal).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("ordinal {ordinal} overflows the u32 metadata ordinal space"),
        )
    })
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
                //
                // A halted log refuses every checkpoint until reopen, so this
                // then ticks along doing nothing. That is the intended
                // behaviour and the reason the error is dropped rather than
                // logged per tick — the write path already told the caller.
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
// Catalog persistence
// ---------------------------------------------------------------------------
//
// `catalog.snap` (root dir) is the registry: which collection ids exist and
// their full config (name, dim, capacity, schema). Same envelope +
// atomic-rename pattern as the per-store snapshots:
//
//   magic b"CAT0" | version u32 | last_lsn u64 | body_len u32 |
//   body = bincode(Vec<CollectionConfig>) | crc32 over all preceding bytes
//
// `last_lsn` is the catalog's WATERMARK (Phase 6): every WAL record at or
// below it — in particular every CreateCollection — is reflected in this
// file. It enters recovery's skip_through and checkpoint's truncation min
// exactly like a store watermark, which is what makes CREATE COLLECTION
// crash-safe: an unregistered create is always > the stamp, so it always
// survives truncation and always gets replayed.
//
// v1 → v2: CollectionConfig gained `name` (body encoding changed) and
// last_lsn became meaningful. v1 files are refused.
//
// CORRUPTION POLICY — deliberately the opposite of the stores': a corrupt
// catalog.snap is a LOUD error, never an empty fallback. WAL-created
// collections could be replayed back, but open()-registered ones are not
// logged anywhere — an empty fallback would silently hide them. Refuse and
// make the operator decide.

const CATALOG_MAGIC: &[u8; 4] = b"CAT0";
const CATALOG_VERSION: u32 = 2;
const CATALOG_FILE: &str = "catalog.snap";
const CATALOG_TMP: &str = "catalog.snap.tmp";

fn encode_catalog(registry: &[CollectionConfig], last_lsn: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(CATALOG_MAGIC);
    buf.extend_from_slice(&CATALOG_VERSION.to_le_bytes());
    buf.extend_from_slice(&last_lsn.to_le_bytes());

    // Plain data into memory — no fallible step (same reasoning as
    // Record::encode in wal.rs).
    let body = bincode::serialize(&registry)
        .expect("catalog body serialization into memory cannot fail");
    buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
    buf.extend_from_slice(&body);

    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

/// Inverse of `encode_catalog`. Trailing CRC verified FIRST, like the stores.
fn decode_catalog(bytes: &[u8]) -> Result<(Vec<CollectionConfig>, u64)> {
    let corrupt = |why: &str| Error::CorruptSnapshot(format!("catalog: {why}"));

    // magic + version + last_lsn + body_len + trailing crc.
    const MIN_LEN: usize = 4 + 4 + 8 + 4 + 4;
    if bytes.len() < MIN_LEN {
        return Err(corrupt("truncated"));
    }
    let (head, crc_tail) = bytes.split_at(bytes.len() - 4);
    let stored_crc = u32::from_le_bytes(crc_tail.try_into().map_err(|_| corrupt("truncated"))?);
    if crc32(head) != stored_crc {
        return Err(corrupt("crc mismatch"));
    }

    if &head[0..4] != CATALOG_MAGIC {
        return Err(corrupt("bad magic"));
    }
    let version = u32::from_le_bytes(head[4..8].try_into().map_err(|_| corrupt("truncated"))?);
    if version != CATALOG_VERSION {
        return Err(Error::CorruptSnapshot(format!(
            "catalog: unsupported version {version}"
        )));
    }
    let last_lsn =
        u64::from_le_bytes(head[8..16].try_into().map_err(|_| corrupt("truncated"))?);
    let body_len =
        u32::from_le_bytes(head[16..20].try_into().map_err(|_| corrupt("truncated"))?) as usize;
    let body = head
        .get(20..20 + body_len)
        .ok_or_else(|| corrupt("body length overruns file"))?;
    if head.len() != 20 + body_len {
        return Err(corrupt("trailing bytes"));
    }

    let registry = bincode::deserialize(body)
        .map_err(|e| Error::CorruptSnapshot(format!("catalog body: {e}")))?;
    Ok((registry, last_lsn))
}

/// Load the persisted registry + its watermark. Missing file = fresh database
/// (empty at 0); corrupt file = Err — see the corruption-policy note above.
fn load_catalog(dir: &Path) -> Result<(Vec<CollectionConfig>, u64)> {
    match std::fs::read(dir.join(CATALOG_FILE)) {
        Ok(bytes) => decode_catalog(&bytes),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok((Vec::new(), 0)),
        Err(e) => Err(e.into()),
    }
}

/// Persist the registry with the shared tmp → fsync → rename → dir-fsync
/// dance (a leftover garbage .tmp from a crash is overwritten, never read).
fn persist_catalog(dir: &Path, registry: &[CollectionConfig], last_lsn: u64) -> Result<()> {
    write_snapshot_atomic(dir, CATALOG_TMP, CATALOG_FILE, &encode_catalog(registry, last_lsn))
}

// ---------------------------------------------------------------------------
// Db
// ---------------------------------------------------------------------------

/// The embeddable vector database handle.
pub struct Db {
    catalog: SharedCatalog,
    /// Serializes DDL: id assignment + name-uniqueness check in
    /// `create_collection` must not race a concurrent create.
    ddl: Mutex<()>,
    wal: Option<Wal>,
    flusher: Option<Flusher>,
}

impl Db {
    /// Open (creating if missing) the database rooted at `dir`. Runs WAL
    /// recovery — replaying the post-checkpoint tail into each index — before
    /// returning, then starts the background flusher.
    ///
    /// `collections` is the set to ENSURE exists: ids not yet in the persisted
    /// catalog are registered (and `catalog.snap` rewritten); ids already
    /// registered must match their persisted config exactly
    /// (`CollectionConfigMismatch` otherwise). Collections registered by
    /// earlier opens re-emerge without being re-supplied — pass `&[]` to open
    /// an existing database as-is.
    pub fn open(dir: impl AsRef<Path>, collections: &[CollectionConfig], opts: DbOptions) -> Result<Db> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // 0. The registry: persisted catalog ∪ caller-supplied configs.
        //    (Runtime registration goes through the WAL — `create_collection`;
        //    this open-time path exists for embedders that declare their
        //    collections up front.) Persist changes before touching any store,
        //    preserving the loaded watermark: registering a collection doesn't
        //    change which WAL records the file reflects.
        let (mut registry, cat_lsn) = load_catalog(dir)?;
        let mut dirty = !dir.join(CATALOG_FILE).exists();
        for cfg in collections {
            match registry.iter().find(|c| c.id == cfg.id) {
                Some(existing) if existing == cfg => {}
                Some(_) => return Err(Error::CollectionConfigMismatch { id: cfg.id }),
                None => {
                    if registry.iter().any(|c| c.name == cfg.name) {
                        return Err(Error::CollectionExists {
                            name: cfg.name.clone(),
                        });
                    }
                    registry.push(cfg.clone());
                    dirty = true;
                }
            }
        }
        registry.sort_by_key(|c| c.id);
        if dirty {
            persist_catalog(dir, &registry, cat_lsn)?;
        }

        // 1. Open/create every registered collection's three stores (flat
        //    vector index, metadata index, tuple store), splitting each into
        //    read handles (kept in the catalog) and writers (handed to the
        //    applier).
        //
        //    Layout: <db dir>/{catalog.snap, wal.log,
        //    collection-{id}/{vectors.idx, metadata.snap, tuples.snap}}.
        //    Phase 4c was a breaking on-disk change from the flat
        //    `collection-{id}.idx` layout (the WAL format changed too); the
        //    upgrade path is a clean shutdown on the old build.
        let mut map: CatalogMap = HashMap::with_capacity(registry.len());
        let mut writers: HashMap<u32, CollectionWriters> =
            HashMap::with_capacity(registry.len());
        // Conservative global minimum watermark across every store of every
        // collection (see the recovery note below).
        let mut min_watermark = u64::MAX;
        for cfg in &registry {
            let (collection_writers, collection, watermark) = open_collection(dir, cfg)?;
            min_watermark = min_watermark.min(watermark);
            map.insert(cfg.id, Arc::new(collection));
            writers.insert(cfg.id, collection_writers);
        }
        let catalog: SharedCatalog = Arc::new(RwLock::new(Arc::new(map)));

        // 2. Recovery skips everything already folded into the indexes.
        //
        //    CORRECTNESS — skip-through is conceptually PER-COLLECTION (and now
        //    per-store): a frame at LSN L may be dropped only if EVERY store
        //    that might own it is durable past L. With a single shared WAL and
        //    no per-frame routing in recovery yet, we use the conservative
        //    global *minimum* across all stores of all collections: anything
        //    <= the slowest store's watermark is durable everywhere, so
        //    skipping it is always safe; frames above it are replayed and
        //    idempotent apply absorbs any a faster store had already seen.
        //    This only over-replays — never skips a record a store still needs.
        //
        //    DO NOT change this to a global `max`: that would skip frames for a
        //    lagging store that has NOT durably applied them — silent data
        //    loss. The eventual fix is a real per-collection watermark keyed on
        //    each frame's collection id; never collapse it to one global
        //    high-water mark.
        //
        //    The CATALOG's watermark joins the min (Phase 6): a
        //    CreateCollection not yet reflected in catalog.snap is > cat_lsn,
        //    so it can never be skipped — replay re-materializes the
        //    collection, then its data records follow.
        let skip_through = min_watermark.min(cat_lsn);
        let skip_through = if skip_through == u64::MAX { 0 } else { skip_through };

        // 3. Start the WAL; recovery replays the tail into the writers (which the
        //    applier owns). The catalog's readers observe the same inners; a
        //    replayed CreateCollection publishes straight into the shared map.
        let wal_path = Self::wal_path(dir);
        let applier = IndexApplier {
            dir: dir.to_path_buf(),
            writers,
            registry,
            catalog: catalog.clone(),
            last_applied: skip_through,
            failed_create: None,
        };
        let wal = Wal::start(&wal_path, applier, skip_through)?;

        // 4. Seed each ordinal allocator from the post-recovery high-water mark
        //    (read via the reader — recovery has already run on this thread).
        //    Snapshot AFTER recovery so WAL-replayed collections are included.
        for coll in catalog_snapshot(&catalog).values() {
            coll.next_ordinal
                .store(coll.reader.len() as u64, Ordering::Release);
        }

        // 5. Background checkpoints (the flusher just pokes the commit thread).
        let flusher = Flusher::spawn(wal.handle(), opts.checkpoint_interval);

        Ok(Db {
            catalog,
            ddl: Mutex::new(()),
            wal: Some(wal),
            flusher: Some(flusher),
        })
    }

    /// Insert `vector` plus its metadata `row` into `collection` as ONE
    /// durable record. Blocks until durable. Returns the vector's ordinal
    /// (its stable id within the collection, shared by all three indexes).
    /// Vector-only collections (empty schema) pass an empty row.
    pub fn insert(&self, collection: u32, vector: &[f32], row: Row) -> Result<Ordinal> {
        let coll = self.collection(collection)?;
        // Validate dim AND row BEFORE we log: a record that can't apply must
        // never reach the durable WAL, or it would fail apply forever on every
        // replay. (The applier's own validation then never fires except on
        // version-skew/corruption.)
        if vector.len() != coll.config.schema.vector().dim.get() {
            return Err(Error::DimensionMismatch {
                expected: coll.config.schema.vector().dim.get(),
                got: vector.len(),
            });
        }
        coll.config.schema.validate_row(&row)?;
        // Refuse before allocating: a halted log is going to reject this
        // append anyway, and burning an ordinal for a write that never reaches
        // the file costs the collection a slot of its capacity for nothing.
        self.check_poison()?;
        let ordinal = coll.alloc_ordinal()?;
        match self.wal_handle()?.append(Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
            metadata: row,
        }) {
            Ok(_lsn) => Ok(Ordinal(ordinal as u32)),
            Err(e) => {
                // Two different failures arrive here and the caller must not
                // conflate them:
                //
                //   * PRE-FSYNC (frame or fsync failed): the append never
                //     became durable, so there is nothing to undo and nothing
                //     to hide. The allocator already burned `ordinal`, leaving
                //     a zero-filled slot in the flat index below the high-water
                //     mark — but a burned ordinal is in no WAL record, so it is
                //     in `live` on no path, and every read (search, scan,
                //     cursor) filters through the liveness snapshot. The hole
                //     is unreachable by construction rather than by a
                //     tombstone.
                //
                //   * POISONED (`Error::Poisoned`): the log halted. If THIS
                //     record is the one that failed to apply then it is
                //     durable and WILL come back on the next open, ordinal and
                //     all — do not tell the caller their write vanished. Which
                //     of the two it was is in the error, not in this branch.
                //
                // This used to flip the flat index's tombstone bit from the
                // CALLER's thread, the one place outside the WAL thread that
                // mutated index state. It also left a durability hole: the bit
                // was in-memory only, so a transient failure followed by a
                // successful insert and a crash before checkpoint would
                // resurface the phantom. Both went away with the snapshot —
                // `live` is rebuilt from metadata.snap plus WAL replay, and the
                // burned ordinal was never in either.
                //
                // NOTE FOR COMPACTION: on the pre-fsync path `ordinal` is
                // permanently burned — the allocator never reuses it, so every
                // failed append costs the collection one slot of its capacity.
                // Compaction renumbers ordinals and reclaims these gaps for
                // free; nothing else does.
                Err(self.wal_err(e))
            }
        }
    }

    /// Tombstone `ordinal` in `collection`. Blocks until durable. Idempotent.
    pub fn delete(&self, collection: u32, ordinal: u64) -> Result<()> {
        // Validate the collection exists before logging.
        let _ = self.collection(collection)?;
        self.check_poison()?;
        self.wal_handle()?
            .append(Record::Delete {
                collection,
                ordinal,
            })
            .map_err(|e| self.wal_err(e))?;
        Ok(())
    }

    /// Apply ONE statement — retire a set of ordinals, install a set of rows —
    /// as a single durable record. Blocks until durable.
    ///
    /// This is the primitive `DELETE` and `UPDATE` compile to. A `DELETE`
    /// installs nothing; an `UPDATE` is copy-on-write, retiring the old
    /// ordinals and installing their replacements at fresh ones. Both are
    /// all-or-nothing because both are one frame, one CRC, one fsync.
    ///
    /// Retiring an ordinal that is not live is a silent no-op (as for
    /// `delete`), which is what keeps replay idempotent. A statement that
    /// retires nothing and installs nothing succeeds without logging: a
    /// predicate matching zero rows is a successful no-op, not an error.
    ///
    /// Everything that can be checked is checked BEFORE logging — dim, rows,
    /// capacity, and the payload cap — because a record that cannot apply must
    /// never reach the durable WAL, or it would fail apply forever on every
    /// replay and (since Phase 7) halt the database each time.
    pub fn mutate(
        &self,
        collection: u32,
        mut retire: RoaringBitmap,
        install: Vec<Install>,
    ) -> Result<Outcome> {
        let coll = self.collection(collection)?;
        if retire.is_empty() && install.is_empty() {
            return Ok(Outcome::Applied);
        }
        // Collapse contiguous runs into run containers before the bitmap is
        // encoded. A bulk DELETE retires long runs, and this is the difference
        // between ~16 KiB and a few dozen bytes per 100k rows in the frame. We
        // own the bitmap here, so it costs no copy.
        retire.optimize();
        self.check_poison()?;

        let dim = coll.config.schema.vector().dim.get();
        for row in &install {
            if row.vector.len() != dim {
                return Err(Error::DimensionMismatch {
                    expected: dim,
                    got: row.vector.len(),
                });
            }
            coll.config.schema.validate_row(&row.metadata)?;
            if row.ordinal >= coll.config.capacity as u64 {
                return Err(Error::CapacityExceeded {
                    capacity: coll.config.capacity,
                });
            }
        }

        // Measured on the installs alone: the retire bitmap is compressed and
        // bounded by the collection, while installs carry the vectors that
        // actually make a statement big.
        let bytes = bincode::serialized_size(&install)
            .map_err(|e| Error::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?
            as usize;
        if bytes > MAX_INSTALL_BYTES {
            return Err(Error::StatementTooLarge {
                bytes,
                cap: MAX_INSTALL_BYTES,
            });
        }

        // Claim the installed ordinals BEFORE logging: a concurrent `insert`
        // must not be able to allocate one of them in the window between the
        // append and the reservation. Like `insert`, a failed append leaves the
        // reservation burned — the ordinals are simply never reused.
        if let Some(highest) = install.iter().map(|r| r.ordinal).max() {
            coll.reserve_through(highest + 1);
        }

        let committed = self
            .wal_handle()?
            .append(Record::Mutate {
                collection,
                retire: Bitmap(retire),
                install,
            })
            .map_err(|e| self.wal_err(e))?;
        // `Conflicted` is a normal return, not an error: the record committed,
        // the commit thread judged it stale, and the caller's move is to read
        // the rows again and retry. Ordinals reserved above stay burned.
        Ok(committed.outcome)
    }

    /// Rewrite every row in `matched`, copy-on-write: read each row, apply the
    /// `set` overrides to a copy at a FRESH ordinal, and retire the original —
    /// all as one `Record::Mutate`, so the statement lands whole or not at all.
    ///
    /// The vector is copied across untouched. `set` is storage-level
    /// (`ColumnId`), already resolved and type-checked by the binder, and is
    /// applied in order so a column named twice takes its last value.
    ///
    /// Returns `Outcome::Conflicted` when the rows moved on between the read
    /// and the commit — the caller re-resolves its predicate and tries again.
    pub fn update_rows(
        &self,
        collection: u32,
        matched: &RoaringBitmap,
        set: &[(ColumnId, Value)],
    ) -> Result<Outcome> {
        let coll = self.collection(collection)?;
        // A predicate matching no rows is a successful no-op, like a DELETE's:
        // nothing to copy, nothing to log, no fsync spent.
        if matched.is_empty() {
            return Ok(Outcome::Applied);
        }
        self.check_poison()?;

        let tuples = self
            .tuple_reader(collection)
            .ok_or(Error::UnknownCollection { id: collection })?;
        let vectors = self
            .reader(collection)
            .ok_or(Error::UnknownCollection { id: collection })?;
        // Every scalar column: an UPDATE rewrites the WHOLE row, so the copy
        // must carry the columns it does not change as well as the ones it does.
        let scalars: Vec<ColumnId> = (0..coll.config.schema.columns.len() as ColumnId).collect();

        // 1. READ the matched rows and build their edited copies. Nothing is
        //    logged yet, and no ordinal is claimed — a failure here costs
        //    nothing but the work.
        let mut copies: Vec<(Vec<f32>, Row)> = Vec::with_capacity(matched.len() as usize);
        for ordinal in matched {
            let ordinal = Ordinal(ordinal);
            let values = match tuples.get(ordinal, &scalars)? {
                tuples::RowGet::Live(values) => values,
                // The caller's bitmap came from a liveness snapshot, so an
                // ordinal in it that the tuple store never saw means those two
                // disagree — the same consistency failure `SnapshotRowMissing`
                // exists to catch, not something to skip past.
                tuples::RowGet::Missing => {
                    return Err(Error::SnapshotRowMissing {
                        collection,
                        ordinal: ordinal.0,
                    });
                }
            };
            let vector = vectors
                .vector_at(ordinal)
                .ok_or(Error::SnapshotRowMissing {
                    collection,
                    ordinal: ordinal.0,
                })?
                .to_vec();

            let mut row: Row = values
                .into_iter()
                .enumerate()
                .map(|(id, value)| (id as ColumnId, value))
                .collect();
            // Overrides in order: a column assigned twice takes its last value.
            for (column, value) in set {
                row[*column as usize] = (*column, value.clone());
            }
            copies.push((vector, row));
        }

        // 2. CLAIM the ordinals for the copies — one reservation, capacity
        //    checked, before any record exists.
        let start = coll.alloc_ordinals(copies.len() as u64)?;
        let install: Vec<Install> = copies
            .into_iter()
            .enumerate()
            .map(|(i, (vector, metadata))| Install {
                ordinal: start + i as u64,
                vector,
                metadata,
            })
            .collect();

        // 3. ONE record: retire the originals, install the copies. If the rows
        //    moved on while we were reading them, the commit thread says so and
        //    the caller retries — see `Outcome`.
        self.mutate(collection, matched.clone(), install)
    }

    /// Brute-force top-`k` search within `collection`. Tombstoned vectors are
    /// excluded. Results are most-similar (highest dot product) first. Runs
    /// lock-free against the reader; concurrent searches do not serialize.
    pub fn search(&self, collection: u32, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let coll = self.collection(collection)?;
        // The SNAPSHOT is the authority, not the flat index's own view of what
        // exists. `flat.write_at` runs first in the fan-out, so without this a
        // bare SEARCH could rank a vector that a scan issued at the same instant
        // would not show — one collection, two answers.
        let live = coll.live.resolve();
        coll.reader
            .search_filtered(query, k, Some(live.bits()), None)
    }

    /// Brute-force top-`k` search within `collection`, restricted to `allowed`.
    ///
    /// The `WHERE`-clause path for `SEARCH`. `allowed` is a PREFILTER — `k`
    /// counts rows that satisfy it, so a filtered search still returns `k` rows
    /// whenever `k` of them match. See
    /// [`index::Reader::search_filtered`](crate::index::index::Reader::search_filtered).
    pub fn search_where(
        &self,
        collection: u32,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        let coll = self.collection(collection)?;
        let live = coll.live.resolve();
        coll.reader
            .search_filtered(query, k, Some(live.bits()), Some(allowed))
    }

    /// Create a new collection through the WAL (Phase 6): DDL is a mutation,
    /// so it rides the same durable path as inserts. Blocks until the record
    /// is fsync'd AND applied — on `Ok`, the collection is durable, visible,
    /// and immediately usable. Returns the assigned collection id.
    ///
    /// Idempotency: replaying the record is a no-op (apply checks the
    /// catalog). A *user*-repeated create errs here with `CollectionExists` —
    /// validation happens before anything reaches the WAL, like every other
    /// record.
    pub fn create_collection(&self, name: &str, capacity: usize, schema: Schema) -> Result<u32> {
        // Serialize DDL: the id assignment and name check below must not race
        // another create (creates are rare; a mutex is fine).
        let _ddl = self.ddl.lock().unwrap_or_else(|e| e.into_inner());

        if name.is_empty() {
            return Err(Error::InvalidCollectionName);
        }
        if capacity == 0 {
            return Err(Error::InvalidCapacity);
        }
        // The schema carries the vector (name + declaration ordinal + dim) and
        // every scalar, and is invalid-by-construction impossible via
        // `Schema::from_columns`. Re-check in case the caller hand-assembled one
        // and bypassed the constructor.
        schema.validate()?;
        // `id` is the ordinal pseudo-column SEARCH returns — a stored column
        // by that name would collide with it in RETURNING.
        for col in &schema.columns {
            if col.name.eq_ignore_ascii_case("id") {
                return Err(Error::ReservedColumn(col.name.clone()));
            }
        }
        let snapshot = catalog_snapshot(&self.catalog);
        if snapshot.values().any(|c| c.config.name == name) {
            return Err(Error::CollectionExists { name: name.into() });
        }
        // Assign the id on the logging side (like ordinals): max + 1. No DROP
        // means ids are never reused.
        let id = snapshot.keys().max().map_or(0, |m| m + 1);

        let config = CollectionConfig {
            id,
            name: name.to_string(),
            capacity,
            schema,
        };
        self.check_poison()?;
        self.wal_handle()?
            .append(Record::CreateCollection { config })
            .map_err(|e| self.wal_err(e))?;
        Ok(id)
    }

    /// The current liveness snapshot for `collection`, materializing it if no
    /// reader has needed this version yet. `None` for an unknown collection.
    ///
    /// Resolve ONCE and hold the returned `Arc` for the life of a query: a
    /// `LiveSet` is immutable, so the rows it admits never change underneath
    /// the caller. Calling this per row would defeat the point.
    pub fn live_snapshot(&self, collection: u32) -> Option<Arc<meta::LiveSet>> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.live.resolve())
    }

    /// A cloneable read handle for `collection`, for issuing searches from other
    /// threads in parallel. Returns `None` for an unknown collection.
    pub fn reader(&self, collection: u32) -> Option<Reader> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.reader.clone())
    }

    /// Read handle for `collection`'s metadata index (lookup_eq/lookup_range
    /// bitmaps). Primitive read-side plumbing — the real SEARCH..WHERE
    /// executor is a later phase. `None` for an unknown collection.
    pub fn metadata_reader(&self, collection: u32) -> Option<meta::Reader> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.meta.clone())
    }

    /// Read handle for `collection`'s tuple store (RETURNING values by
    /// ordinal). `None` for an unknown collection.
    pub fn tuple_reader(&self, collection: u32) -> Option<tuples::Reader> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.tuple.clone())
    }

    /// A full-scan [`Cursor`] over `collection`'s live rows, projecting every
    /// scalar column in `ColumnId` order.
    ///
    /// The cursor's ordinal source is this collection's `live()` bitmap,
    /// snapshotted here — so the row SET is fixed for the scan's lifetime, and
    /// rows inserted after this call are not visited. Rows deleted after it are
    /// skipped when reached (see [`Cursor`]'s skip policy).
    ///
    /// Full scan is only the first customer of that seam: a `WHERE`-filtered
    /// bitmap or a ranked KNN result builds the same cursor through
    /// [`Cursor::over`].
    pub fn scan(&self, collection: u32) -> Result<Cursor<'static>> {
        let coll = self.collection(collection)?;
        // The SNAPSHOT, not a fresh read of `live`. Two readers that resolve at
        // the same version get the same set, which is what lets a scan and a
        // filtered search agree about what is live; `meta.live()` is a read of
        // mutable state and two calls a microsecond apart can differ.
        //
        // The bitmap is cloned out because `Cursor<'static>` needs an owning
        // iterator and `LiveSet::iter` borrows. That clone is the same cost as
        // the `live()` call it replaces — and next to a brute-force scan over
        // the vectors themselves, it is noise.
        let live = coll.live.resolve().bitmap().clone();
        self.scan_over(collection, live.into_iter().map(Ordinal))
    }

    /// A [`Cursor`] over an ARBITRARY sequence of ordinals, projecting every
    /// scalar column in `ColumnId` order.
    ///
    /// The general form of [`scan`](Self::scan), and the seam the cursor was
    /// built around: the ordinal source is a plain iterator, so a full scan, a
    /// ranked KNN result, and a `WHERE`-filtered bitmap all produce the SAME
    /// cursor type with no new machinery.
    ///
    /// **`ordinals` must come from this collection's liveness snapshot** — a
    /// full snapshot, or anything derived from one by intersection or ranking.
    /// An ordinal the tuple store cannot produce is reported as
    /// [`Error::SnapshotRowMissing`] rather than skipped, because from a
    /// snapshot it can only mean the stores disagree. To iterate ordinals from
    /// somewhere else, build a [`Cursor::over`](crate::Cursor::over) directly;
    /// that one skips.
    ///
    /// **Order is preserved.** The cursor visits ordinals in the order given and
    /// imposes none of its own — which is what lets a KNN result stay in SCORE
    /// order rather than collapsing to ascending. An ordinal with no
    /// materialized row is skipped, per the cursor's documented policy.
    ///
    /// The iterator is `'static` so the returned cursor borrows nothing; pass an
    /// owned collection's `into_iter()`.
    pub fn scan_over(
        &self,
        collection: u32,
        ordinals: impl Iterator<Item = Ordinal> + 'static,
    ) -> Result<Cursor<'static>> {
        let coll = self.collection(collection)?;
        // Dense and ascending by construction (`Schema::columns` IS the
        // ColumnId space), which is what makes `Cursor::column`'s position
        // coincide with the ColumnId for this projection.
        let columns = coll.config.schema.columns.iter().map(|c| c.id).collect();
        // STRICT: every ordinal source the engine hands a cursor is derived
        // from a liveness snapshot, so a row the tuple store cannot produce is
        // a consistency failure rather than something to skip past.
        Ok(Cursor::over_snapshot(
            collection,
            ordinals,
            coll.tuple.clone(),
            columns,
        ))
    }

    /// Resolve a collection NAME to the id every other method here is keyed by.
    ///
    /// The one public name→id path. Compiled bytecode carries
    /// `collection: String` (a program is portable — it outlives any particular
    /// catalog state and must not bake in an id), while `insert`, `search`,
    /// `scan` and the WAL records all speak [`CollectionId`]. The VM crosses
    /// that gap exactly once, here, when it opens a cursor.
    ///
    /// Matching is EXACT, deliberately: `create_collection` enforces uniqueness
    /// with the same exact comparison, so a case-insensitive lookup here could
    /// resolve two names that the catalog considers distinct. It reads the same
    /// live snapshot as [`Catalog::schema_of`](crate::sql::bind::Catalog) — so a
    /// name the binder resolved
    /// against resolves to a real id here, including one created earlier in the
    /// same session.
    pub fn collection_id(&self, name: &str) -> Result<CollectionId> {
        catalog_snapshot(&self.catalog)
            .values()
            .find(|c| c.config.name == name)
            .map(|c| c.config.id)
            .ok_or_else(|| Error::UnknownCollectionName { name: name.into() })
    }

    /// Every registered collection's config, sorted by id — exactly what the
    /// persisted catalog holds.
    pub fn collections(&self) -> Vec<CollectionConfig> {
        let mut v: Vec<CollectionConfig> = catalog_snapshot(&self.catalog)
            .values()
            .map(|c| c.config.clone())
            .collect();
        v.sort_by_key(|c| c.id);
        v
    }

    /// Force a checkpoint now (index durable + WAL truncated). Runs on the WAL
    /// commit thread. Mostly for tests and graceful shutdown; the flusher does
    /// this on a timer otherwise.
    pub fn checkpoint(&self) -> Result<()> {
        // A halted log refuses this: checkpointing would stamp a durable
        // watermark over the record the index never absorbed and then truncate
        // that record away. See `Error::Poisoned`.
        self.wal_handle()?
            .checkpoint()
            .map_err(|e| self.wal_err(e))?;
        Ok(())
    }

    /// Graceful shutdown: stop the flusher, take one final checkpoint so reopen
    /// is fast and the WAL is trimmed, then drain and join the WAL thread.
    pub fn close(mut self) -> Result<()> {
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        let wal = self.wal.take().expect("wal present until close");
        // Final checkpoint on the commit thread — but shut the thread down
        // whichever way it goes. Returning early here used to drop `Wal`
        // without joining, leaking the commit thread; a halted log makes that
        // path routine rather than exotic, since it refuses the checkpoint by
        // design. The WAL is durable either way, so a skipped checkpoint only
        // means the next open replays a longer tail.
        let checkpointed = wal.handle().checkpoint();
        let poison = wal.poison();
        wal.shutdown();
        if let Err(e) = checkpointed {
            return Err(match poison {
                Some(p) => Error::Poisoned {
                    lsn: p.lsn.0,
                    cause: p.cause,
                },
                None => Error::from(e),
            });
        }
        Ok(())
    }

    fn collection(&self, id: u32) -> Result<Arc<Collection>> {
        catalog_snapshot(&self.catalog)
            .get(&id)
            .cloned()
            .ok_or(Error::UnknownCollection { id })
    }

    /// Whether the database has halted on an apply failure. `true` means every
    /// write and every checkpoint is being refused and the fix is to reopen;
    /// reads still work. See `Error::Poisoned`.
    pub fn is_poisoned(&self) -> bool {
        self.wal.as_ref().is_some_and(|w| w.poison().is_some())
    }

    /// Refuse early if the log has halted, with the typed error rather than the
    /// WAL's io-shaped one.
    fn check_poison(&self) -> Result<()> {
        match self.wal.as_ref().and_then(|w| w.poison()) {
            Some(p) => Err(Error::Poisoned {
                lsn: p.lsn.0,
                cause: p.cause,
            }),
            None => Ok(()),
        }
    }

    /// Map a WAL-layer failure into the engine's error type, promoting it to
    /// `Poisoned` when the log has halted.
    ///
    /// A pre-fsync failure on one thread that races an apply failure on another
    /// is reported as `Poisoned` rather than as itself. That is a fair trade:
    /// both mean "this write did not take effect", and the halt is the more
    /// important half of the news.
    fn wal_err(&self, e: io::Error) -> Error {
        match self.wal.as_ref().and_then(|w| w.poison()) {
            Some(p) => Error::Poisoned {
                lsn: p.lsn.0,
                cause: p.cause,
            },
            None => Error::from(e),
        }
    }

    fn wal_handle(&self) -> Result<WalHandle> {
        self.wal
            .as_ref()
            .map(|w| w.handle())
            .ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::BrokenPipe, "db is closing")))
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
// Query frontend integration: the binder's catalog read path
// ---------------------------------------------------------------------------

/// Reconstruct the binder-shaped, VECTOR-INCLUSIVE schema from a collection's
/// persisted metadata schema. The binder needs every column — the vector plus
/// the scalars — in one ordinal space; this maps the storage schema (scalar
/// `ColumnId` space + the separate vector) into it. The binder is unchanged;
/// this adapter does all the conversion.
pub(crate) fn bind_schema(schema: &Schema) -> crate::sql::bind::Schema {
    use crate::metadata::common::ColumnType;
    use crate::sql::ast::ColumnType as AstType;
    use crate::sql::bind::ColumnSchema;

    let v = schema.vector();
    let mut columns = Vec::with_capacity(schema.columns.len() + 1);
    columns.push(ColumnSchema {
        name: v.name.clone(),
        ty: AstType::Vector(v.dim.get()),
        ordinal: v.ordinal.get(),
        is_vector: true,
    });
    for def in &schema.columns {
        columns.push(ColumnSchema {
            name: def.name.clone(),
            ty: match def.ty {
                ColumnType::Int => AstType::Int,
                ColumnType::Float => AstType::Float,
                ColumnType::Text => AstType::Text,
            },
            ordinal: def.ordinal.get(),
            is_vector: false,
        });
    }
    // Declaration order — the binder indexes projections by these ordinals.
    columns.sort_by_key(|c| c.ordinal);
    crate::sql::bind::Schema { columns }
}

/// The engine IS the binder's catalog: resolve a collection name to its
/// binder-shaped [`Schema`](crate::sql::bind::Schema). This is the real read
/// path that replaces the test-only fixture — the binder consumes it unchanged.
impl crate::sql::bind::Catalog for Db {
    fn get_collection(&self, name: &str) -> Option<crate::sql::bind::Schema> {
        catalog_snapshot(&self.catalog)
            .values()
            .find(|c| c.config.name == name)
            .map(|c| bind_schema(&c.config.schema))
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
    use crate::metadata::common::{
        ColumnDef, ColumnSpec, ColumnType, DeclarationOrdinal, RangeOp, Value,
    };
    use crate::metadata::tuples::RowGet;
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    /// Vector-only collection: empty schema, inserts pass an empty row.
    fn cfg(id: u32, dim: usize, capacity: usize) -> CollectionConfig {
        CollectionConfig {
            id,
            name: format!("c{id}"),
            capacity,
            schema: Schema::from_columns(vec![ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(dim).unwrap(),
            }])
            .unwrap(),
        }
    }

    /// Collection with a metadata schema: (a INT, c TEXT) alongside the vector.
    fn cfg_meta(id: u32, dim: usize, capacity: usize) -> CollectionConfig {
        CollectionConfig {
            id,
            name: format!("c{id}"),
            capacity,
            schema: Schema::from_columns(vec![
                ColumnSpec::Vector {
                    name: "vector".into(),
                    dim: NonZeroUsize::new(dim).unwrap(),
                },
                ColumnSpec::Scalar {
                    name: "a".into(),
                    ty: ColumnType::Int,
                },
                ColumnSpec::Scalar {
                    name: "c".into(),
                    ty: ColumnType::Text,
                },
            ])
            .unwrap(),
        }
    }

    /// A vector-only schema (no scalar columns), embedding dim `dim`.
    fn vec_only(dim: usize) -> Schema {
        Schema::from_columns(vec![ColumnSpec::Vector {
            name: "vector".into(),
            dim: NonZeroUsize::new(dim).unwrap(),
        }])
        .unwrap()
    }

    fn meta_row(a: i64, c: &str) -> Row {
        vec![(0, Value::Int(a)), (1, Value::Text(c.into()))]
    }

    /// Rows written by `writes_materialize_nothing`. Each insert is its own
    /// group-commit batch (one writer, one record) and therefore its own fsync,
    /// so this trades coverage against wall clock. Override to push it harder:
    /// `FLATS_WRITE_ONLY_ROWS=100000 cargo test writes_materialize_nothing`.
    fn write_only_rows() -> usize {
        std::env::var("FLATS_WRITE_ONLY_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_WRITE_ONLY_ROWS)
    }

    const DEFAULT_WRITE_ONLY_ROWS: usize = 2_000;

    // Large interval => the background flusher never fires; tests drive
    // checkpoints explicitly for determinism.
    fn manual_opts() -> DbOptions {
        DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
        }
    }

    /// Poison a live `Db` the way production would: hand the WAL a record that
    /// is durable but that the applier cannot absorb. `collection` is not
    /// registered, so `IndexApplier::writers` errors AFTER the fsync — the same
    /// shape as an ENOSPC while materializing a collection, but deterministic.
    ///
    /// Goes straight to the WAL handle on purpose: `Db::insert` validates the
    /// collection first, which is exactly the guard that makes this
    /// unreachable through the public API.
    fn poison_db(db: &Db, collection: u32, ordinal: u64, vector: &[f32]) {
        db.wal
            .as_ref()
            .expect("wal present")
            .handle()
            .append(Record::Insert {
                collection,
                ordinal,
                vector: vector.to_vec(),
                metadata: vec![],
            })
            .expect_err("apply must fail for an unregistered collection");
    }

    #[test]
    fn poisoned_db_refuses_writes() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();

        poison_db(&db, 9, 0, &[1.0, 0.0]);
        assert!(db.is_poisoned());

        // Both write doors report the same typed error, naming the record the
        // index is missing. Acking anything else would be a lie: the record is
        // committed and WILL come back on the next open.
        match db.insert(0, &[2.0, 0.0], vec![]) {
            Err(Error::Poisoned { lsn, .. }) => assert_eq!(lsn, 2),
            other => panic!("expected Poisoned, got {other:?}"),
        }
        match db.delete(0, 0) {
            Err(Error::Poisoned { lsn, .. }) => assert_eq!(lsn, 2),
            other => panic!("expected Poisoned, got {other:?}"),
        }
    }

    #[test]
    fn poisoned_db_still_serves_reads() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();

        poison_db(&db, 9, 0, &[1.0, 0.0]);

        // The index is STALE, not inconsistent: it holds a consistent prefix of
        // the log. Readers keep their guarantees, so a poisoned db degrades to
        // read-only rather than going dark.
        let hits = db.search(0, &[1.0, 0.0], 4).unwrap();
        assert_eq!(hits.len(), 1);
        let mut cursor = db.scan(0).unwrap();
        assert!(cursor.seek_first().unwrap(), "the scan cursor still sees the row");
        assert_eq!(cursor.ordinal().unwrap(), Ordinal(0));
    }

    #[test]
    fn reopen_after_poison_replays_the_unapplied_record() {
        let dir = tempfile::tempdir().unwrap();

        {
            let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
            db.insert(0, &[1.0, 0.0], vec![]).unwrap();
            // Collection 1 is not registered in THIS session, so the record is
            // durable but unappliable.
            poison_db(&db, 1, 0, &[0.0, 1.0]);

            // The halt's whole job: refuse to checkpoint, so the frame stays in
            // the log for the recovery below. Without this the durable
            // watermark would advance over the hole and truncation would erase
            // the record permanently.
            assert!(matches!(db.checkpoint(), Err(Error::Poisoned { .. })));
            db.close().expect_err("close reports the poison");
        }

        // Reopen DECLARING collection 1: `open` registers and materializes it
        // (step 0/1) before the WAL replays (step 3), so the record that
        // poisoned the last session now applies cleanly.
        let db = Db::open(dir.path(), &[cfg(0, 2, 64), cfg(1, 2, 64)], manual_opts()).unwrap();
        assert!(!db.is_poisoned(), "recovery clears the condition");

        let hits = db.search(1, &[0.0, 1.0], 4).unwrap();
        assert_eq!(hits.len(), 1, "the unapplied record survived and replayed");

        // And the db is fully writable again.
        db.insert(1, &[0.0, 2.0], vec![]).unwrap();
        db.checkpoint().unwrap();
    }

    #[test]
    fn close_after_poison_reports_it_and_still_joins() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        poison_db(&db, 9, 0, &[1.0, 0.0]);

        // close() takes a final checkpoint, which a poisoned db refuses. The
        // caller must hear about it — but the commit thread still has to be
        // joined, or an early return leaks it.
        match db.close() {
            Err(Error::Poisoned { .. }) => {}
            other => panic!("expected Poisoned, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Record::Mutate — one statement, one record
    // -----------------------------------------------------------------------

    fn install(ordinal: u64, x: f32) -> Install {
        Install {
            ordinal,
            vector: vec![x, 0.0],
            metadata: vec![],
        }
    }

    /// An installed ordinal is OCCUPIED. The allocator hands out the next
    /// free one, so a statement that places a row without telling it would let
    /// the very next `insert` allocate the same ordinal and overwrite the row
    /// it just installed — silently, and only until the next reopen reseeds
    /// the allocator from the flat index's high-water mark.
    #[test]
    fn installing_a_row_reserves_its_ordinal_against_the_allocator() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        assert_eq!(db.insert(0, &[1.0, 0.0], vec![]).unwrap(), Ordinal(0));

        // Install past the allocator's high-water mark, leaving a gap.
        db.mutate(0, RoaringBitmap::new(), vec![install(4, 7.0)])
            .unwrap();

        assert_eq!(
            db.insert(0, &[3.0, 0.0], vec![]).unwrap(),
            Ordinal(5),
            "the allocator must resume above every installed ordinal"
        );
        // And the installed row is still intact, not overwritten.
        let hits = db.search(0, &[7.0, 0.0], 8).unwrap();
        assert!(hits.iter().any(|h| h.id == Ordinal(4)));
    }

    #[test]
    fn mutate_end_to_end_then_reopen_replays() {
        let dir = tempfile::tempdir().unwrap();
        let cfgs = [cfg(0, 2, 64)];

        {
            let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
            db.insert(0, &[1.0, 0.0], vec![]).unwrap();
            db.insert(0, &[2.0, 0.0], vec![]).unwrap();

            // The UPDATE shape: retire ordinal 0, install its copy-on-write
            // replacement at a fresh ordinal. One record, one fsync.
            let retire: RoaringBitmap = [0u32].into_iter().collect();
            db.mutate(0, retire, vec![install(2, 9.0)]).unwrap();

            let hits = db.search(0, &[9.0, 0.0], 8).unwrap();
            assert_eq!(hits.len(), 2, "one retired, one installed => count holds");
            assert!(
                hits.iter().any(|h| h.id == Ordinal(2)),
                "the installed row is searchable"
            );
            assert!(
                !hits.iter().any(|h| h.id == Ordinal(0)),
                "the retired row is gone"
            );
            db.close().unwrap();
        }

        // The whole statement must come back from the log as one unit.
        let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
        let hits = db.search(0, &[9.0, 0.0], 8).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| h.id == Ordinal(2)));
        assert!(!hits.iter().any(|h| h.id == Ordinal(0)));
    }

    /// The fan-out rule from `Insert`, extended to a statement: any ordinal the
    /// liveness snapshot admits must ALREADY be readable in the tuple store and
    /// the flat index. A snapshot is immutable, so a reader that catches an
    /// installed ordinal before its data landed stays wrong for its whole life.
    #[test]
    fn mutate_writes_data_before_publishing_liveness() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();

        let retire: RoaringBitmap = [0u32].into_iter().collect();
        db.mutate(0, retire, vec![install(1, 5.0), install(2, 6.0)])
            .unwrap();

        let snap = db.live_snapshot(0).expect("collection is open");
        let reader = db.reader(0).expect("collection is open");
        for ordinal in snap.iter() {
            assert!(
                reader.vector_at(Ordinal(ordinal)).is_some(),
                "ordinal {ordinal} is live but has no vector"
            );
        }
        assert_eq!(snap.iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    /// A statement matching nothing is a successful no-op, not an error and not
    /// a log record. `DELETE FROM docs WHERE <matches nothing>` must succeed,
    /// and burning an fsync on a record that changes nothing is pure cost.
    #[test]
    fn empty_mutate_is_a_no_op_and_logs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();

        let before = std::fs::metadata(dir.path().join("wal.log")).unwrap().len();
        db.mutate(0, RoaringBitmap::new(), vec![])
            .expect("an empty statement succeeds");
        assert_eq!(
            std::fs::metadata(dir.path().join("wal.log")).unwrap().len(),
            before,
            "an empty statement must not reach the log"
        );
    }

    /// One record means one buffer, on the commit thread now and on every
    /// replay forever. The cap turns "this UPDATE rewrites a million vectors"
    /// into a named error at the door instead of an allocation failure inside
    /// the commit loop — and it is what bounds the metadata lock hold, since
    /// that is proportional to statement size.
    #[test]
    fn oversized_install_is_refused_before_logging() {
        let dir = tempfile::tempdir().unwrap();
        // 4096 floats = 16 KiB of vector per row; enough rows to clear 256 MiB.
        let dim = 4096;
        let per_row = dim * std::mem::size_of::<f32>();
        let rows = MAX_INSTALL_BYTES / per_row + 2;
        let db = Db::open(dir.path(), &[cfg(0, dim, rows + 8)], manual_opts()).unwrap();

        let installs: Vec<Install> = (0..rows as u64)
            .map(|ordinal| Install {
                ordinal,
                vector: vec![0.5; dim],
                metadata: vec![],
            })
            .collect();

        let before = std::fs::metadata(dir.path().join("wal.log")).unwrap().len();
        match db.mutate(0, RoaringBitmap::new(), installs) {
            Err(Error::StatementTooLarge { bytes, cap }) => {
                assert!(bytes > cap);
                assert_eq!(cap, MAX_INSTALL_BYTES);
            }
            other => panic!("expected StatementTooLarge, got {other:?}"),
        }
        assert_eq!(
            std::fs::metadata(dir.path().join("wal.log")).unwrap().len(),
            before,
            "a refused statement must never reach the durable log"
        );
    }

    #[test]
    fn install_just_under_the_cap_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 4096;
        // bincode spends ~24 bytes of framing per Install on top of the vector
        // (ordinal, plus a length prefix for each of the two vecs). Budget for
        // it, or "just under" silently becomes "just over".
        let per_row = dim * std::mem::size_of::<f32>() + 32;
        let rows = MAX_INSTALL_BYTES / per_row - 1;
        let db = Db::open(dir.path(), &[cfg(0, dim, rows + 8)], manual_opts()).unwrap();

        let installs: Vec<Install> = (0..rows as u64)
            .map(|ordinal| Install {
                ordinal,
                vector: vec![0.5; dim],
                metadata: vec![],
            })
            .collect();
        // Pin that this really is a boundary case rather than a small
        // statement that trivially fits.
        let bytes = bincode::serialized_size(&installs).unwrap() as usize;
        assert!(
            bytes <= MAX_INSTALL_BYTES && bytes > MAX_INSTALL_BYTES - 1024 * 1024,
            "expected a statement just under the cap, got {bytes}"
        );

        db.mutate(0, RoaringBitmap::new(), installs)
            .expect("just under the cap is a legal statement");
        assert_eq!(db.live_snapshot(0).unwrap().len(), rows as u64);
    }

    // -----------------------------------------------------------------------
    // DELETE — a predicate resolved to a bitmap, logged as one Mutate
    // -----------------------------------------------------------------------

    /// Resolve `c = value` against the CURRENT snapshot and delete the matches,
    /// exactly as the VM's `Op::Mutate` will: the predicate becomes a bitmap on
    /// the logging side, and the bitmap becomes one `Record::Mutate` with
    /// nothing installed. Returns what was matched.
    fn delete_where_c(db: &Db, value: &str) -> RoaringBitmap {
        let snapshot = db.live_snapshot(0).unwrap();
        let matched = db
            .metadata_reader(0)
            .unwrap()
            .lookup_eq(1, &Value::Text(value.into()), &snapshot)
            .unwrap();
        db.mutate(0, matched.clone(), vec![]).unwrap();
        matched
    }

    fn seed_alice_bob_alice(db: &Db) {
        db.insert(0, &[1.0, 0.0], meta_row(1, "alice")).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], meta_row(2, "bob")).unwrap(); // ord 1
        db.insert(0, &[3.0, 0.0], meta_row(3, "alice")).unwrap(); // ord 2
    }

    /// The core contract, through every read path: a deleted row is gone from
    /// search, from scans, and from metadata lookups — and nothing else is.
    #[test]
    fn delete_by_bitmap_retires_exactly_the_matched_rows() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_alice_bob_alice(&db);

        let matched = delete_where_c(&db, "alice");
        assert_eq!(matched.iter().collect::<Vec<_>>(), vec![0, 2]);

        let hits = db.search(0, &[1.0, 0.0], 8).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![Ordinal(1)]
        );

        let mut cursor = db.scan(0).unwrap();
        assert!(cursor.seek_first().unwrap());
        assert_eq!(cursor.ordinal().unwrap(), Ordinal(1));
        assert!(!cursor.next().unwrap(), "the scan sees nothing but bob");

        let snapshot = db.live_snapshot(0).unwrap();
        let meta = db.metadata_reader(0).unwrap();
        assert!(
            meta.lookup_eq(1, &Value::Text("alice".into()), &snapshot)
                .unwrap()
                .is_empty(),
            "a deleted row satisfies no predicate"
        );
    }

    /// Drop skips the closing checkpoint, so the delete survives only if the
    /// WAL replays it on open.
    #[test]
    fn delete_by_bitmap_replays_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let cfgs = [cfg_meta(0, 2, 64)];
        {
            let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
            seed_alice_bob_alice(&db);
            delete_where_c(&db, "alice");
            drop(db);
        }
        let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// Two statements racing to delete the same rows, or one replayed after a
    /// crash, must both succeed: retiring a retired row is a no-op, not an error.
    #[test]
    fn deleting_already_deleted_rows_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_alice_bob_alice(&db);

        let stale: RoaringBitmap = [0u32, 2].into_iter().collect();
        db.mutate(0, stale.clone(), vec![]).unwrap();
        db.mutate(0, stale, vec![])
            .expect("retiring retired rows succeeds");
        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// `DELETE … WHERE <matches nothing>` is a successful no-op that costs no
    /// fsync.
    #[test]
    fn zero_match_delete_logs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_alice_bob_alice(&db);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).unwrap().len();
        assert!(delete_where_c(&db, "carol").is_empty());
        assert_eq!(std::fs::metadata(&wal).unwrap().len(), before);
        assert_eq!(db.live_snapshot(0).unwrap().len(), 3);
    }

    /// A DELETE deletes what matched in the snapshot it read. A row inserted
    /// after that snapshot survives even if it matches — and it can never be
    /// hit by accident, because ordinals are never reused, so a stale bitmap
    /// cannot name a newer row.
    #[test]
    fn delete_resolved_on_an_old_snapshot_spares_later_inserts() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], meta_row(1, "alice")).unwrap(); // ord 0

        let snapshot = db.live_snapshot(0).unwrap();
        let matched = db
            .metadata_reader(0)
            .unwrap()
            .lookup_eq(1, &Value::Text("alice".into()), &snapshot)
            .unwrap();

        // A matching row lands between resolving and logging.
        let late = db.insert(0, &[2.0, 0.0], meta_row(2, "alice")).unwrap();
        db.mutate(0, matched, vec![]).unwrap();

        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![late.0],
            "the late insert was not in the statement's snapshot"
        );
    }

    /// "The record is kilobytes": a bulk DELETE of a contiguous range must log
    /// a few hundred bytes, not one entry per row. Roaring only reaches that
    /// size in RUN containers, which it builds on `optimize()` — left alone, the
    /// same 100k ordinals sit in two 8 KiB bitmap containers.
    #[test]
    fn delete_of_100k_contiguous_rows_logs_kilobytes() {
        const ROWS: u64 = 100_000;
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, ROWS as usize)], manual_opts()).unwrap();

        // Seed in ONE record: 100k single-row inserts would be 100k fsyncs.
        let seed: Vec<Install> = (0..ROWS).map(|o| install(o, 1.0)).collect();
        db.mutate(0, RoaringBitmap::new(), seed).unwrap();
        assert_eq!(db.live_snapshot(0).unwrap().len(), ROWS);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).unwrap().len();
        let everything: RoaringBitmap = (0..ROWS as u32).collect();
        db.mutate(0, everything, vec![]).unwrap();
        let grew = std::fs::metadata(&wal).unwrap().len() - before;

        assert!(db.live_snapshot(0).unwrap().is_empty());
        assert!(
            grew < 1024,
            "deleting {ROWS} contiguous rows logged {grew} bytes; runs must be compressed"
        );
    }

    // -----------------------------------------------------------------------
    // Staleness — a copy-on-write statement whose rows moved on
    // -----------------------------------------------------------------------
    //
    // An UPDATE is built on the caller's thread: find the rows, read them,
    // then log "retire the old, install the copy". Two callers can do that
    // concurrently and both read the same row, so both log a copy of it — and
    // the WAL, doing exactly its job, applies both and the row exists twice.
    //
    // The fix is a precondition checked where everything is already
    // serialized, on the commit thread: a record that INSTALLS rows is stale
    // if the rows it retires are already retired AND none of the rows it
    // installs are live. Stale records are skipped whole.

    /// The loser of a race: it retires a row someone else already retired, and
    /// its own copy never landed. Skipping it is what stops the duplicate.
    #[test]
    fn a_stale_mutate_is_skipped_whole() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], vec![]).unwrap(); // ord 1

        // The winner retires ordinal 0 and installs its copy at 2.
        db.mutate(0, [0u32].into_iter().collect(), vec![install(2, 9.0)])
            .unwrap();

        // The loser was built from the same read: same retire set, its own
        // fresh ordinal. By now ordinal 0 is gone and 3 was never installed.
        db.mutate(0, [0u32].into_iter().collect(), vec![install(3, 9.0)])
            .unwrap();

        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1, 2],
            "the loser's copy must not be installed — that is the duplicate"
        );
        assert!(
            db.reader(0).unwrap().vector_at(Ordinal(3)).is_none()
                || !db.live_snapshot(0).unwrap().contains(3),
            "ordinal 3 is not live"
        );
    }

    /// The winner, and the ordinary case: the rows it retires are still live,
    /// so it applies.
    #[test]
    fn a_fresh_mutate_still_applies() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0

        db.mutate(0, [0u32].into_iter().collect(), vec![install(1, 9.0)])
            .unwrap();

        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1]
        );
        let hits = db.search(0, &[9.0, 0.0], 8).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![Ordinal(1)]
        );
    }

    /// Retiring nothing cannot be stale. This is the bulk-load shape the WAL
    /// and delete tests seed with, and the precondition must leave it alone.
    #[test]
    fn a_bulk_install_with_no_retire_is_never_stale() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();

        let rows: Vec<Install> = (0..8u64).map(|o| install(o, o as f32)).collect();
        db.mutate(0, RoaringBitmap::new(), rows).unwrap();

        assert_eq!(db.live_snapshot(0).unwrap().len(), 8);
    }

    /// A DELETE retires rows and installs nothing, so it is idempotent and must
    /// never be judged stale — replay redelivers it and it must stay a no-op
    /// rather than becoming a skipped record.
    #[test]
    fn a_retire_only_mutate_is_never_stale() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], vec![]).unwrap(); // ord 1

        let retire: RoaringBitmap = [0u32].into_iter().collect();
        db.mutate(0, retire.clone(), vec![]).unwrap();
        db.mutate(0, retire, vec![]).unwrap();

        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// The loser of the race has to LEARN it lost, or it silently drops the
    /// user's update. Applied vs conflicted is the difference between "done"
    /// and "read the row again and try once more".
    #[test]
    fn mutate_tells_its_caller_it_conflicted() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0

        let retire: RoaringBitmap = [0u32].into_iter().collect();
        assert_eq!(
            db.mutate(0, retire.clone(), vec![install(1, 9.0)]).unwrap(),
            Outcome::Applied,
            "the winner applies"
        );
        assert_eq!(
            db.mutate(0, retire, vec![install(2, 9.0)]).unwrap(),
            Outcome::Conflicted,
            "the loser is told, rather than silently dropped"
        );
    }

    /// The second staleness clause, seen from outside. A record that already
    /// applied gets redelivered by replay; its retires are long done, so the
    /// first clause alone would call it stale. Its installs are live, so it
    /// must be re-applied idempotently instead — otherwise a store that had
    /// not yet absorbed it would never get the row.
    #[test]
    fn a_redelivered_mutate_reports_applied_not_conflicted() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0

        let record = Record::Mutate {
            collection: 0,
            retire: Bitmap([0u32].into_iter().collect()),
            install: vec![install(1, 9.0)],
        };
        let handle = db.wal.as_ref().expect("wal present").handle();
        assert_eq!(
            handle.append(record.clone()).unwrap().outcome,
            Outcome::Applied
        );
        // The same record again — exactly what recovery delivers.
        assert_eq!(
            handle.append(record).unwrap().outcome,
            Outcome::Applied,
            "a record whose installs are live is redone, not skipped"
        );
        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// THE test this whole mechanism exists for.
    ///
    /// Two threads run the copy-on-write shape of an UPDATE against the same
    /// single row: read the live row, then log "retire it, install my copy at
    /// my own fresh ordinal". Barriers make both threads read BEFORE either
    /// logs, which is exactly the interleaving that produces the bug — both
    /// records are built from the same row, and applying both leaves two live
    /// copies of one row.
    ///
    /// Two invariants, checked every round: the collection still holds exactly
    /// one row, and exactly one of the two writers was told it conflicted. The
    /// loser simply re-reads on the next round, which is what the UPDATE
    /// executor will do in a retry loop.
    #[test]
    fn concurrent_copy_on_write_updates_cannot_duplicate_a_row() {
        use std::sync::Barrier;

        const ROUNDS: usize = 16;
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Db::open(dir.path(), &[cfg(0, 2, 256)], manual_opts()).unwrap());
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0: the row

        // Each thread draws its own fresh ordinals, as the executor will from
        // the collection's allocator.
        let next_ordinal = Arc::new(AtomicU64::new(1));
        let barrier = Arc::new(Barrier::new(2));

        let workers: Vec<_> = (0..2)
            .map(|tag| {
                let db = Arc::clone(&db);
                let next_ordinal = Arc::clone(&next_ordinal);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut conflicts = 0usize;
                    for _ in 0..ROUNDS {
                        // PHASE 1 — read. The barrier below guarantees the
                        // other thread has also read before either writes.
                        let live = db.live_snapshot(0).expect("collection open");
                        let current = live.iter().next().expect("exactly one row");
                        let old = db
                            .reader(0)
                            .expect("collection open")
                            .vector_at(Ordinal(current))
                            .expect("the live row has a vector")
                            .to_vec();
                        barrier.wait();

                        // PHASE 2 — copy-on-write: retire what we read,
                        // install our edit of it at a fresh ordinal.
                        let ordinal = next_ordinal.fetch_add(1, Ordering::SeqCst);
                        let edited = vec![old[0] + 1.0, tag as f32];
                        let outcome = db
                            .mutate(
                                0,
                                [current].into_iter().collect::<RoaringBitmap>(),
                                vec![Install {
                                    ordinal,
                                    vector: edited,
                                    metadata: vec![],
                                }],
                            )
                            .expect("a conflict is not an error");
                        if outcome == Outcome::Conflicted {
                            conflicts += 1;
                        }
                        barrier.wait();
                    }
                    conflicts
                })
            })
            .collect();

        let conflicts: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();

        assert_eq!(
            db.live_snapshot(0).unwrap().len(),
            1,
            "one row must stay one row, however the two writers interleaved"
        );
        assert_eq!(
            conflicts, ROUNDS,
            "each round has exactly one winner and one loser"
        );
    }

    // -----------------------------------------------------------------------
    // UPDATE — copy-on-write, one statement, one record
    // -----------------------------------------------------------------------

    fn seed_three(db: &Db) {
        db.insert(0, &[1.0, 0.0], meta_row(1, "draft")).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], meta_row(2, "draft")).unwrap(); // ord 1
        db.insert(0, &[3.0, 0.0], meta_row(3, "draft")).unwrap(); // ord 2
    }

    /// Read a live row's (a, c) values plus its vector, by ordinal.
    fn row_at(db: &Db, ordinal: u32) -> (Vec<Value>, Vec<f32>) {
        let values = match db.tuple_reader(0).unwrap().get(Ordinal(ordinal), &[0, 1]).unwrap() {
            tuples::RowGet::Live(values) => values,
            other => panic!("ordinal {ordinal} is not readable: {other:?}"),
        };
        let vector = db
            .reader(0)
            .unwrap()
            .vector_at(Ordinal(ordinal))
            .expect("live row has a vector")
            .to_vec();
        (values, vector)
    }

    /// The whole contract of a metadata-only UPDATE: the SET columns change,
    /// everything else survives — including the embedding, byte for byte —
    /// the originals retire, and rows nobody matched are untouched.
    #[test]
    fn update_rows_copies_each_row_with_the_set_applied() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let matched: RoaringBitmap = [0u32, 2].into_iter().collect();
        let outcome = db
            .update_rows(0, &matched, &[(1, Value::Text("archived".into()))])
            .unwrap();
        assert_eq!(outcome, Outcome::Applied);

        let live: Vec<u32> = db.live_snapshot(0).unwrap().iter().collect();
        assert_eq!(live, vec![1, 3, 4], "originals retired, copies at fresh ordinals");

        // The untouched row is exactly as it was.
        assert_eq!(
            row_at(&db, 1).0,
            vec![Value::Int(2), Value::Text("draft".into())]
        );

        // Each copy: SET column changed, other column kept, vector identical.
        let (values, vector) = row_at(&db, 3);
        assert_eq!(values, vec![Value::Int(1), Value::Text("archived".into())]);
        assert_eq!(vector, vec![1.0, 0.0], "the embedding is copied unchanged");

        let (values, vector) = row_at(&db, 4);
        assert_eq!(values, vec![Value::Int(3), Value::Text("archived".into())]);
        assert_eq!(vector, vec![3.0, 0.0]);
    }

    /// One statement is one publish, however many rows it rewrites. The
    /// liveness version is what readers watch, so a multi-row UPDATE that
    /// bumped it per row would invalidate every cached snapshot N times and,
    /// worse, expose the half-rewritten states in between.
    #[test]
    fn a_multi_row_update_publishes_once() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let before = db.live_snapshot(0).unwrap().version();
        let matched: RoaringBitmap = [0u32, 1, 2].into_iter().collect();
        db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))])
            .unwrap();

        assert_eq!(
            db.live_snapshot(0).unwrap().version(),
            before + 1,
            "three rows rewritten, one publish"
        );
    }

    /// Later assignments win, so `SET c = 'a', c = 'b'` means 'b'.
    #[test]
    fn update_applies_assignments_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let matched: RoaringBitmap = [0u32].into_iter().collect();
        db.update_rows(
            0,
            &matched,
            &[
                (1, Value::Text("first".into())),
                (1, Value::Text("last".into())),
            ],
        )
        .unwrap();

        assert_eq!(
            row_at(&db, 3).0,
            vec![Value::Int(1), Value::Text("last".into())]
        );
    }

    #[test]
    fn an_update_matching_nothing_logs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).unwrap().len();
        let outcome = db
            .update_rows(0, &RoaringBitmap::new(), &[(1, Value::Text("x".into()))])
            .unwrap();

        assert_eq!(outcome, Outcome::Applied);
        assert_eq!(std::fs::metadata(&wal).unwrap().len(), before);
        assert_eq!(db.live_snapshot(0).unwrap().len(), 3);
    }

    #[test]
    fn an_update_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let cfgs = [cfg_meta(0, 2, 64)];
        {
            let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
            seed_three(&db);
            let matched: RoaringBitmap = [1u32].into_iter().collect();
            db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))])
                .unwrap();
            drop(db); // no closing checkpoint: the WAL must carry it
        }
        let db = Db::open(dir.path(), &cfgs, manual_opts()).unwrap();
        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        assert_eq!(
            row_at(&db, 3).0,
            vec![Value::Int(2), Value::Text("archived".into())]
        );
    }

    #[test]
    fn insert_search_delete_e2e() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();

        let a = db.insert(0, &[1.0, 0.0], vec![]).unwrap();
        let b = db.insert(0, &[2.0, 0.0], vec![]).unwrap();
        let _c = db.insert(0, &[0.0, 1.0], vec![]).unwrap();
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
            db.insert(0, &[1.0, 0.0], vec![]).unwrap();
            db.insert(0, &[2.0, 0.0], vec![]).unwrap();
            db.checkpoint().unwrap(); // durable + WAL truncated
            db.insert(0, &[3.0, 0.0], vec![]).unwrap(); // lives only in the WAL tail
            db.close().unwrap();
        }
        // Reopen: checkpointed prefix comes from the index, the tail from the WAL.
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        let hits = db.search(0, &[1.0, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, Ordinal(2)); // [3,0]
        // The allocator resumed past the recovered high-water mark.
        let next = db.insert(0, &[4.0, 0.0], vec![]).unwrap();
        assert_eq!(next, Ordinal(3));
        db.close().unwrap();
    }

    #[test]
    fn capacity_is_enforced_before_logging() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 1, 2)], manual_opts()).unwrap();
        db.insert(0, &[1.0], vec![]).unwrap();
        db.insert(0, &[2.0], vec![]).unwrap();
        assert!(matches!(
            db.insert(0, &[3.0], vec![]),
            Err(Error::CapacityExceeded { .. })
        ));
        db.close().unwrap();
    }

    #[test]
    fn unknown_collection_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 8)], manual_opts()).unwrap();
        assert!(matches!(
            db.insert(7, &[1.0, 2.0], vec![]),
            Err(Error::UnknownCollection { id: 7 })
        ));
        db.close().unwrap();
    }

    #[test]
    fn failed_insert_does_not_leak_phantom_ordinal() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 1

        // Force the next durability to fail; its ordinal (2) is burned by the
        // allocator but its slot is never written.
        db.wal.as_ref().unwrap().fail_next_append();
        let failed = db.insert(0, &[9.0, 9.0], vec![]);
        assert!(failed.is_err(), "durability failure must surface to caller");

        // A subsequent successful insert takes ordinal 3 and pushes the
        // high-water mark to 4, pulling the burned slot 2 (zero-filled) into
        // search range. It must NOT appear — the error path tombstoned it.
        let later = db.insert(0, &[1.0, 0.0], vec![]).unwrap();
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

    /// The burned slot is hidden by the SNAPSHOT, not by a tombstone — and the
    /// flat index is left physically untouched to prove it.
    ///
    /// `Db::insert`'s error path used to flip the flat index's tombstone bit
    /// from the CALLER's thread: the one place outside the WAL thread that
    /// mutated index state. It is unnecessary, because a burned ordinal is in
    /// no WAL record and therefore in `live` on no path, and every read filters
    /// through the liveness snapshot.
    #[test]
    fn burned_ordinal_is_hidden_without_touching_the_index() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 0
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 1

        db.wal.as_ref().unwrap().fail_next_append();
        assert!(db.insert(0, &[9.0, 9.0], vec![]).is_err(), "must surface");

        // A later success pulls the burned slot into search range.
        assert_eq!(db.insert(0, &[1.0, 0.0], vec![]).unwrap(), Ordinal(3));

        // THE POINT: the flat index still holds ordinal 2, zero-filled and
        // UNTOMBSTONED. Nothing mutated it — `vector_at` would return None if
        // anything had.
        let flat = db.reader(0).expect("reader");
        assert_eq!(
            flat.vector_at(Ordinal(2)),
            Some(&[0.0, 0.0][..]),
            "the burned slot must be left exactly as the allocator left it"
        );

        // ...and it is invisible anyway, because the snapshot never admitted it.
        assert!(!db.live_snapshot(0).unwrap().contains(2));
        let ids: BTreeSet<u32> = db
            .search(0, &[1.0, 1.0], 64)
            .unwrap()
            .iter()
            .map(|h| h.id.0)
            .collect();
        assert_eq!(ids, [0u32, 1, 3].into_iter().collect());

        // The scan path agrees — one authority, one answer.
        let mut cursor = db.scan(0).unwrap();
        let mut scanned = Vec::new();
        let mut has = cursor.seek_first().unwrap();
        while has {
            scanned.push(cursor.ordinal().unwrap().0);
            has = cursor.next().unwrap();
        }
        assert_eq!(scanned, vec![0, 1, 3]);

        db.close().unwrap();
    }

    /// The durability hole the old TODO deferred, now closed.
    ///
    /// The in-memory tombstone was lost on reopen, so a TRANSIENT append
    /// failure, then a successful insert (pushing the high-water mark past the
    /// gap), then a crash before any checkpoint would rebuild the mark over an
    /// untombstoned zero slot and resurface the phantom. `live` is rebuilt from
    /// metadata.snap plus WAL replay instead, and the burned ordinal is in
    /// neither — so there is nothing to lose.
    #[test]
    fn burned_ordinal_stays_hidden_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();
        db.wal.as_ref().unwrap().fail_next_append();
        assert!(db.insert(0, &[9.0, 9.0], vec![]).is_err());
        db.insert(0, &[1.0, 0.0], vec![]).unwrap(); // ord 2, mark -> 3

        // NO checkpoint: reopen replays the WAL from the last durable snapshot,
        // which is exactly the window the old mechanism could not survive.
        db.close().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();

        assert!(!db.live_snapshot(0).unwrap().contains(1));
        let ids: BTreeSet<u32> = db
            .search(0, &[1.0, 1.0], 64)
            .unwrap()
            .iter()
            .map(|h| h.id.0)
            .collect();
        assert_eq!(
            ids,
            [0u32, 2].into_iter().collect(),
            "the burned ordinal came back after recovery"
        );
        db.close().unwrap();
    }

    // -----------------------------------------------------------------------
    // Phase 4c: metadata through the WAL apply path
    // -----------------------------------------------------------------------

    /// The full read-side loop by hand (the SEARCH..WHERE executor is a later
    /// phase): filter bitmap from the metadata index, intersect with search
    /// candidates, fetch RETURNING values from the tuple store.
    #[test]
    fn metadata_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();

        db.insert(0, &[1.0, 0.0], meta_row(1, "alice")).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], meta_row(2, "bob")).unwrap(); // ord 1
        db.insert(0, &[3.0, 0.0], meta_row(3, "alice")).unwrap(); // ord 2

        let meta = db.metadata_reader(0).unwrap();
        let tuples = db.tuple_reader(0).unwrap();

        // WHERE a < 3 → {0, 1}.
        let filter = meta.lookup_range(0, RangeOp::Lt, &Value::Int(3), &db.live_snapshot(0).unwrap()).unwrap();
        assert_eq!(filter.iter().collect::<Vec<u32>>(), vec![0, 1]);

        // Intersect with search candidates by hand.
        let hits = db.search(0, &[1.0, 0.0], 10).unwrap();
        let filtered: Vec<u32> = hits
            .iter()
            .map(|h| h.id.0)
            .filter(|id| filter.contains(*id))
            .collect();
        assert_eq!(filtered, vec![1, 0], "best-first among a<3 is ord 1 then 0");

        // RETURNING c, a for the winner.
        assert_eq!(
            tuples.get(Ordinal(1), &[1, 0]).unwrap(),
            RowGet::Live(vec![Value::Text("bob".into()), Value::Int(2)])
        );

        // Delete and re-check exclusion everywhere.
        db.delete(0, 0).unwrap();
        let filter = meta.lookup_eq(1, &Value::Text("alice".into()), &db.live_snapshot(0).unwrap()).unwrap();
        assert_eq!(filter.iter().collect::<Vec<u32>>(), vec![2]);
        // Retiring a row clears its live bit and touches nothing else — the
        // values are still there for anyone holding an older snapshot.
        assert!(matches!(tuples.get(Ordinal(0), &[0]).unwrap(), RowGet::Live(_)));
        assert_eq!(meta.live_count(), 2);

        db.close().unwrap();
    }

    /// Crash (drop without close = no final checkpoint) → reopen must bring
    /// all three stores to the same state purely from WAL replay.
    #[test]
    fn recovery_without_checkpoint_keeps_three_stores_consistent() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
            for i in 0..5i64 {
                db.insert(0, &[i as f32, 1.0], meta_row(i, if i % 2 == 0 { "even" } else { "odd" }))
                    .unwrap();
            }
            db.delete(0, 3).unwrap();
            // Drop, not close: skips the final checkpoint, everything lives
            // only in the WAL tail.
        }

        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        let meta = db.metadata_reader(0).unwrap();
        let tuples = db.tuple_reader(0).unwrap();

        // flat live count == metadata live count == live tuple slots.
        let hits = db.search(0, &[0.0, 1.0], 64).unwrap();
        assert_eq!(hits.len(), 4);
        assert_eq!(meta.live_count(), 4);
        for ord in [0u32, 1, 2, 4] {
            assert!(matches!(tuples.get(Ordinal(ord), &[0]).unwrap(), RowGet::Live(_)));
        }
        // Retired, so absent from `live` — but its values survive.
        assert!(matches!(tuples.get(Ordinal(3), &[0]).unwrap(), RowGet::Live(_)));
        assert!(!meta.live().contains(3));

        // Spot-check lookups and values.
        let evens = meta.lookup_eq(1, &Value::Text("even".into()), &db.live_snapshot(0).unwrap()).unwrap();
        assert_eq!(evens.iter().collect::<Vec<u32>>(), vec![0, 2, 4]);
        assert_eq!(
            tuples.get(Ordinal(4), &[0, 1]).unwrap(),
            RowGet::Live(vec![Value::Int(4), Value::Text("even".into())])
        );
        db.close().unwrap();
    }

    /// The serialize-and-rename crash mode: a checkpoint dies BEFORE the
    /// rename, leaving a garbage .tmp while the real snapshot (and the WAL
    /// tail beyond it) stay intact. Reopen must converge. Each metadata store
    /// takes a turn as the victim.
    #[test]
    fn mixed_crash_torn_checkpoint_converges() {
        for victim in ["metadata.snap.tmp", "tuples.snap.tmp"] {
            let dir = tempfile::tempdir().unwrap();
            {
                let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
                db.insert(0, &[1.0, 0.0], meta_row(1, "x")).unwrap();
                db.insert(0, &[2.0, 0.0], meta_row(2, "y")).unwrap();
                db.checkpoint().unwrap(); // snapshots at LSN 2, WAL truncated
                db.insert(0, &[3.0, 0.0], meta_row(3, "z")).unwrap(); // WAL tail
                drop(db); // crash: no close checkpoint
            }
            // The crash artifact of a torn SECOND checkpoint.
            std::fs::write(dir.path().join("collection-0").join(victim), b"garbage").unwrap();

            let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
            let meta = db.metadata_reader(0).unwrap();
            let tuples = db.tuple_reader(0).unwrap();
            assert_eq!(db.search(0, &[0.0, 1.0], 64).unwrap().len(), 3, "victim {victim}");
            assert_eq!(meta.live_count(), 3, "victim {victim}");
            assert_eq!(
                tuples.get(Ordinal(2), &[0]).unwrap(),
                RowGet::Live(vec![Value::Int(3)]),
                "victim {victim}"
            );
            db.close().unwrap();
        }
    }

    /// A corrupt snapshot while the WAL still holds every record: the victim
    /// opens empty at Lsn(0), its 0 drags the min watermark down, the full
    /// replay heals it while the other stores absorb the replay idempotently.
    /// (Corruption AFTER the WAL truncated past the data is outside the crash
    /// model — serialize-and-rename makes torn checkpoints leave the OLD
    /// complete snapshot, which bounds truncation; see Apply::checkpoint.)
    #[test]
    fn corrupt_snapshot_with_full_wal_self_heals() {
        for victim in ["metadata.snap", "tuples.snap"] {
            let dir = tempfile::tempdir().unwrap();
            {
                let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
                db.insert(0, &[1.0, 0.0], meta_row(1, "x")).unwrap();
                db.insert(0, &[2.0, 0.0], meta_row(2, "y")).unwrap();
                db.delete(0, 0).unwrap();
                drop(db); // no checkpoint: the WAL holds everything
            }
            // A torn/garbage snapshot file where none (or a partial one) was.
            std::fs::write(dir.path().join("collection-0").join(victim), b"garbage").unwrap();

            let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
            let meta = db.metadata_reader(0).unwrap();
            let tuples = db.tuple_reader(0).unwrap();
            assert_eq!(meta.live_count(), 1, "victim {victim}");
            assert_eq!(
                meta.lookup_eq(0, &Value::Int(2), &db.live_snapshot(0).unwrap()).unwrap().iter().collect::<Vec<u32>>(),
                vec![1],
                "victim {victim}"
            );
            assert!(matches!(tuples.get(Ordinal(0), &[1]).unwrap(), RowGet::Live(_)));
            assert_eq!(
                tuples.get(Ordinal(1), &[1]).unwrap(),
                RowGet::Live(vec![Value::Text("y".into())]),
                "victim {victim}"
            );
            db.close().unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // Phase 5: persistent catalog
    // -----------------------------------------------------------------------

    #[test]
    fn catalog_round_trips_collections_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(
                dir.path(),
                &[cfg_meta(0, 2, 64), cfg(1, 3, 32)],
                manual_opts(),
            )
            .unwrap();
            db.insert(0, &[1.0, 0.0], meta_row(7, "kept")).unwrap();
            db.insert(1, &[1.0, 0.0, 0.0], vec![]).unwrap();
            db.close().unwrap();
        }

        // Reopen with NO configs: both collections re-emerge from catalog.snap
        // with their exact configs, and their data is intact.
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        assert_eq!(db.collections(), vec![cfg_meta(0, 2, 64), cfg(1, 3, 32)]);
        assert_eq!(db.search(0, &[1.0, 0.0], 10).unwrap().len(), 1);
        assert_eq!(db.search(1, &[1.0, 0.0, 0.0], 10).unwrap().len(), 1);
        let meta = db.metadata_reader(0).unwrap();
        assert_eq!(
            meta.lookup_eq(0, &Value::Int(7), &db.live_snapshot(0).unwrap()).unwrap().iter().collect::<Vec<u32>>(),
            vec![0]
        );
        // The re-emerged schema still validates inserts.
        assert!(matches!(
            db.insert(0, &[2.0, 0.0], vec![]),
            Err(Error::IncompleteRow)
        ));
        db.close().unwrap();
    }

    #[test]
    fn catalog_crash_mid_persist_keeps_old_catalog() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg(0, 2, 8)], manual_opts()).unwrap();
            db.close().unwrap();
        }
        // The crash artifact: a torn catalog rewrite that never got renamed.
        std::fs::write(dir.path().join("catalog.snap.tmp"), b"garbage").unwrap();

        // The good snap loads; the tmp is ignored.
        {
            let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
            assert_eq!(db.collections(), vec![cfg(0, 2, 8)]);
            db.close().unwrap();
        }
        // And the next registration overwrites the stale tmp and persists.
        {
            let db = Db::open(dir.path(), &[cfg(1, 4, 8)], manual_opts()).unwrap();
            db.close().unwrap();
        }
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        assert_eq!(db.collections(), vec![cfg(0, 2, 8), cfg(1, 4, 8)]);
        db.close().unwrap();
    }

    #[test]
    fn conflicting_config_for_registered_collection_errs() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
            db.close().unwrap();
        }
        // Same id, different schema (cfg's is empty) → refuse before any
        // store is touched.
        assert!(matches!(
            Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()),
            Err(Error::CollectionConfigMismatch { id: 0 })
        ));
        // Same id, same schema shape but different capacity → also refuse.
        assert!(matches!(
            Db::open(dir.path(), &[cfg_meta(0, 2, 128)], manual_opts()),
            Err(Error::CollectionConfigMismatch { id: 0 })
        ));
        // The exact registered config still opens fine.
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        db.close().unwrap();
    }

    #[test]
    fn corrupt_catalog_is_a_loud_error_not_a_fallback() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg(0, 2, 8)], manual_opts()).unwrap();
            db.close().unwrap();
        }
        let snap = dir.path().join(CATALOG_FILE);
        let mut bytes = std::fs::read(&snap).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&snap, &bytes).unwrap();

        // Collection registration is not WAL-rebuildable, so this must NOT
        // silently open an empty database.
        assert!(matches!(
            Db::open(dir.path(), &[], manual_opts()),
            Err(Error::CorruptSnapshot(_))
        ));
    }

    /// WAL records for different collections dispatch to the right writers,
    /// and recovery restores each collection independently — including their
    /// separate ordinal spaces and allocators.
    #[test]
    fn multi_collection_apply_dispatches_and_recovers_independently() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(
                dir.path(),
                &[cfg_meta(0, 2, 64), cfg_meta(1, 4, 64)],
                manual_opts(),
            )
            .unwrap();
            // Interleave records across the two collections in one WAL.
            assert_eq!(db.insert(0, &[1.0, 0.0], meta_row(1, "a")).unwrap(), Ordinal(0));
            assert_eq!(db.insert(1, &[1.0, 0.0, 0.0, 0.0], meta_row(10, "b")).unwrap(), Ordinal(0));
            assert_eq!(db.insert(0, &[2.0, 0.0], meta_row(2, "a")).unwrap(), Ordinal(1));
            assert_eq!(db.insert(1, &[2.0, 0.0, 0.0, 0.0], meta_row(20, "b")).unwrap(), Ordinal(1));
            db.delete(0, 0).unwrap(); // only collection 0's ordinal 0 dies
            // Drop without close: everything recovers from the WAL tail alone.
        }

        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();

        // Collection 0: one live row (ordinal 1), the tombstone held.
        assert_eq!(db.search(0, &[1.0, 0.0], 10).unwrap().len(), 1);
        let meta0 = db.metadata_reader(0).unwrap();
        assert_eq!(meta0.live_count(), 1);
        assert!(matches!(
            db.tuple_reader(0).unwrap().get(Ordinal(0), &[0]).unwrap(),
            RowGet::Live(_)
        ));
        assert!(!meta0.live().contains(0), "retired, so out of `live`");

        // Collection 1: untouched by collection 0's delete; both rows live.
        assert_eq!(db.search(1, &[1.0, 0.0, 0.0, 0.0], 10).unwrap().len(), 2);
        let meta1 = db.metadata_reader(1).unwrap();
        assert_eq!(meta1.live_count(), 2);
        assert_eq!(
            meta1.lookup_eq(0, &Value::Int(20), &db.live_snapshot(0).unwrap()).unwrap().iter().collect::<Vec<u32>>(),
            vec![1]
        );
        assert_eq!(
            db.tuple_reader(1).unwrap().get(Ordinal(0), &[1]).unwrap(),
            RowGet::Live(vec![Value::Text("b".into())])
        );

        // Allocators resumed independently per collection.
        assert_eq!(db.insert(0, &[3.0, 0.0], meta_row(3, "a")).unwrap(), Ordinal(2));
        assert_eq!(
            db.insert(1, &[3.0, 0.0, 0.0, 0.0], meta_row(30, "b")).unwrap(),
            Ordinal(2)
        );
        db.close().unwrap();
    }

    // -----------------------------------------------------------------------
    // Phase 6: CREATE COLLECTION through the WAL
    // -----------------------------------------------------------------------

    /// Logs a record but never applies it — used to model "the WAL frame's
    /// fsync landed, then the process died before apply ran at all." Mirrors
    /// `tests/reconciliation.rs`'s `LogOnly`.
    struct LogOnly;
    impl Apply for LogOnly {
        fn apply(&mut self, _lsn: Lsn, _record: &Record) -> io::Result<Outcome> {
            Ok(Outcome::Applied)
        }
        fn checkpoint(&mut self) -> io::Result<Option<u64>> {
            Ok(None)
        }
    }

    #[test]
    fn create_collection_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();

        let schema = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(3).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap();
        let id = db.create_collection("docs", 16, schema).unwrap();
        assert_eq!(id, 0, "first collection gets id 0");

        db.insert(id, &[1.0, 0.0, 0.0], vec![(0, Value::Int(7))]).unwrap();
        db.insert(id, &[0.0, 1.0, 0.0], vec![(0, Value::Int(8))]).unwrap();

        let hits = db.search(id, &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, Ordinal(0));
        assert_eq!(
            db.metadata_reader(id)
                .unwrap()
                .lookup_eq(0, &Value::Int(8), &db.live_snapshot(0).unwrap())
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![1]
        );
        db.close().unwrap();
    }

    /// Models a true crash: the CreateCollection frame is fsync'd to the WAL
    /// but NOTHING ever applies it — no collection dir, no catalog.snap.
    /// `Db::open`'s recovery (skip_through = 0 on a fresh db) must replay it
    /// through the real applier and fully materialize the collection.
    #[test]
    fn create_collection_crash_before_materialization_replays_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");
        let config = CollectionConfig {
            id: 0,
            name: "docs".into(),
            capacity: 8,
            schema: Schema::from_columns(vec![
                ColumnSpec::Vector {
                    name: "vector".into(),
                    dim: NonZeroUsize::new(2).unwrap(),
                },
                ColumnSpec::Scalar {
                    name: "a".into(),
                    ty: ColumnType::Int,
                },
            ])
            .unwrap(),
        };
        {
            let wal = Wal::start(&wal_path, LogOnly, 0).unwrap();
            wal.handle()
                .append(Record::CreateCollection {
                    config: config.clone(),
                })
                .unwrap();
            wal.shutdown();
        }
        assert!(!dir.path().join("catalog.snap").exists(), "nothing applied yet");
        assert!(!dir.path().join("collection-0").exists());

        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        assert_eq!(db.collections(), vec![config]);
        db.insert(0, &[1.0, 0.0], vec![(0, Value::Int(1))]).unwrap();
        assert_eq!(db.search(0, &[1.0, 0.0], 10).unwrap().len(), 1);
        db.close().unwrap();
    }

    #[test]
    fn duplicate_create_is_a_noop_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        let id = db.create_collection("docs", 8, vec_only(2)).unwrap();
        db.insert(id, &[1.0, 0.0], vec![]).unwrap();

        // User-level repeat: caught before it ever reaches the WAL.
        assert!(matches!(
            db.create_collection("docs", 8, vec_only(2)),
            Err(Error::CollectionExists { .. })
        ));

        // Replay-level repeat — what recovery actually does: redeliver the
        // exact record already durable in the log. Apply's own existence
        // check must make this silent, not an error or a duplicate.
        let config = db.collections().into_iter().find(|c| c.id == id).unwrap();
        db.wal_handle()
            .unwrap()
            .append(Record::CreateCollection { config })
            .unwrap();

        assert_eq!(db.collections().len(), 1);
        assert_eq!(
            db.search(id, &[1.0, 0.0], 10).unwrap().len(),
            1,
            "data untouched by the replayed create"
        );
        db.close().unwrap();
    }

    #[test]
    fn invalid_create_is_rejected_before_the_wal_and_burns_no_id() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();

        assert!(matches!(
            db.create_collection("", 8, vec_only(2)),
            Err(Error::InvalidCollectionName)
        ));
        assert!(matches!(
            db.create_collection("x", 0, vec_only(2)),
            Err(Error::InvalidCapacity)
        ));
        // The `id` pseudo-column collides with SEARCH's ordinal column.
        assert!(matches!(
            db.create_collection(
                "x",
                8,
                Schema::from_columns(vec![
                    ColumnSpec::Vector {
                        name: "vector".into(),
                        dim: NonZeroUsize::new(2).unwrap(),
                    },
                    ColumnSpec::Scalar {
                        name: "id".into(),
                        ty: ColumnType::Int,
                    },
                ])
                .unwrap(),
            ),
            Err(Error::ReservedColumn(_))
        ));
        // A hand-built Schema bypassing the `Schema::from_columns` constructor
        // still gets caught — create_collection re-validates it.
        let mut sneaky = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(2).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap();
        sneaky.columns.push(ColumnDef {
            id: 1,
            name: "a".into(),
            ty: ColumnType::Text,
            ordinal: DeclarationOrdinal::new(2),
        });
        assert!(matches!(
            db.create_collection("x", 8, sneaky),
            Err(Error::DuplicateColumn(_))
        ));

        assert!(db.collections().is_empty(), "nothing registered by any failed attempt");

        // No id was burned by the failed attempts: the first real create
        // still gets id 0.
        let id = db.create_collection("real", 8, vec_only(2)).unwrap();
        assert_eq!(id, 0);
        db.close().unwrap();
    }

    // -----------------------------------------------------------------------
    // Phase 7g.1: full schema on disk + the binder's real catalog read path
    // -----------------------------------------------------------------------

    /// The full schema — vector name/ordinal/dim + every scalar
    /// name/id/ordinal/type — survives persist + reopen, reconstructed from
    /// disk alone. The scalar ColumnId space stays dense 0..N (vector excluded);
    /// declaration ordinals are vector-inclusive.
    #[test]
    fn full_schema_round_trips_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        // Vector deliberately NOT first: [author@0, embedding@1, year@2].
        let schema = Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Vector {
                name: "embedding".into(),
                dim: NonZeroUsize::new(16).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "year".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap();
        let cfg0 = CollectionConfig {
            id: 0,
            name: "docs".into(),
            capacity: 8,
            schema,
        };
        {
            let db = Db::open(dir.path(), std::slice::from_ref(&cfg0), manual_opts()).unwrap();
            db.close().unwrap();
        }
        // Reopen with NO configs: the collection re-emerges from catalog.snap.
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        assert_eq!(db.collections(), vec![cfg0]);
        let s = &db.collections()[0].schema;
        // Vector: name + declaration ordinal + dim.
        assert_eq!(s.vector().name, "embedding");
        assert_eq!(s.vector().ordinal.get(), 1);
        assert_eq!(s.vector().dim.get(), 16);
        // Scalars: ColumnId dense 0..N (vector excluded); ordinals vector-inclusive.
        assert_eq!(s.columns.len(), 2);
        assert_eq!(
            (s.columns[0].id, s.columns[0].name.as_str(), s.columns[0].ordinal.get()),
            (0, "author", 0)
        );
        assert_eq!(
            (s.columns[1].id, s.columns[1].name.as_str(), s.columns[1].ordinal.get()),
            (1, "year", 2)
        );
        db.close().unwrap();
    }

    /// The engine satisfies the binder's `Catalog` trait, and the binder
    /// consumes the REAL schema with zero binder changes.
    #[test]
    fn engine_is_a_binder_catalog_and_binder_consumes_it() {
        use crate::sql::bind::{BoundStatement, Catalog, analyze};
        use crate::sql::parse;

        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        // docs: vector VECTOR(768) @0, author TEXT @1, title TEXT @2.
        let schema = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(768).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "title".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap();
        db.create_collection("docs", 1_000_000, schema).unwrap();

        // Read the binder-shaped schema back through the trait.
        let bound = Catalog::get_collection(&db, "docs").expect("collection exists");
        assert_eq!(bound.columns.len(), 3);
        assert!(bound.columns[0].is_vector);
        assert_eq!(
            bound
                .columns
                .iter()
                .map(|c| (c.name.as_str(), c.ordinal, c.is_vector))
                .collect::<Vec<_>>(),
            vec![
                ("vector", 0, true),
                ("author", 1, false),
                ("title", 2, false),
            ]
        );
        assert!(Catalog::get_collection(&db, "ghost").is_none());

        // The binder resolves queries against the REAL engine catalog.
        match analyze(parse("SELECT author, title FROM docs;").unwrap(), &db).unwrap() {
            BoundStatement::Select(sel) => {
                assert_eq!(sel.from, "docs");
                assert_eq!(
                    sel.projection
                        .iter()
                        .map(|c| (c.name.as_str(), c.ordinal))
                        .collect::<Vec<_>>(),
                    vec![("author", 1), ("title", 2)]
                );
                assert!(!sel.include_vector);
            }
            other => panic!("expected Select, got {other:?}"),
        }
        // SELECT * excludes the embedding; naming the vector includes it.
        match analyze(parse("SELECT * FROM docs;").unwrap(), &db).unwrap() {
            BoundStatement::Select(sel) => {
                assert_eq!(
                    sel.projection.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                    vec!["author".to_string(), "title".to_string()]
                );
                assert!(!sel.include_vector);
            }
            other => panic!("expected Select, got {other:?}"),
        }
        match analyze(parse("SELECT vector FROM docs;").unwrap(), &db).unwrap() {
            BoundStatement::Select(sel) => assert!(sel.include_vector),
            other => panic!("expected Select, got {other:?}"),
        }
        db.close().unwrap();
    }

    // -- name → id resolution ------------------------------------------------
    //
    // Bytecode carries `collection: String`; every `Db` method is keyed by id.
    // These pin the one public bridge between the two.

    #[test]
    fn resolve_known_collection() {
        let dir = tempfile::tempdir().unwrap();
        // Two collections, so a lookup that returned "some id" rather than
        // "this name's id" cannot pass.
        let db = Db::open(
            dir.path(),
            &[cfg(0, 2, 16), cfg_meta(1, 2, 16)],
            manual_opts(),
        )
        .unwrap();

        assert_eq!(db.collection_id("c0").unwrap(), 0);
        assert_eq!(db.collection_id("c1").unwrap(), 1);

        // A collection created at RUNTIME resolves too — the VM's own path is
        // CREATE then INSERT in one session, so the id must come back from the
        // live catalog, not from whatever `open` was handed.
        let id = db.create_collection("made_later", 16, vec_only(2)).unwrap();
        assert_eq!(db.collection_id("made_later").unwrap(), id);
        // ...and it agrees with the id every other `Db` method wants.
        db.insert(db.collection_id("made_later").unwrap(), &[1.0, 0.0], vec![])
            .unwrap();

        db.close().unwrap();
    }

    #[test]
    fn resolve_unknown_collection_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 16)], manual_opts()).unwrap();

        // A typed error naming the offender — not a panic, not `UnknownCollection
        // { id }` (there is no id to report; that is the whole failure).
        match db.collection_id("nope") {
            Err(Error::UnknownCollectionName { name }) => assert_eq!(name, "nope"),
            other => panic!("expected UnknownCollectionName, got {other:?}"),
        }
        assert!(db.collection_id("").is_err());
        // Names are case-SENSITIVE, matching the uniqueness check in
        // `create_collection`: if "C0" resolved to "c0" here, two names that
        // `create_collection` considers distinct would collide at execution.
        assert!(db.collection_id("C0").is_err());

        db.close().unwrap();
    }

    // -----------------------------------------------------------------------
    // Liveness versioning
    // -----------------------------------------------------------------------

    /// The collection's shared liveness handle. Tests live in this module, so
    /// they reach it directly rather than widening the public API.
    fn live_handle(db: &Db, id: u32) -> Arc<meta::LiveHandle> {
        Arc::clone(
            &catalog_snapshot(&db.catalog)
                .get(&id)
                .expect("collection exists")
                .live,
        )
    }

    /// One bump per liveness-changing record, no more and no less. The counter
    /// is what invalidates every cached snapshot, so a missed bump makes writes
    /// permanently invisible and a doubled bump throws away a good snapshot.
    #[test]
    fn insert_and_delete_bump_the_version_once_each() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        let live = live_handle(&db, 0);

        let before = live.version();
        db.insert(0, &[1.0, 0.0], vec![]).unwrap();
        assert_eq!(live.version(), before + 1, "insert bumps exactly once");

        db.insert(0, &[2.0, 0.0], vec![]).unwrap();
        db.insert(0, &[3.0, 0.0], vec![]).unwrap();
        assert_eq!(live.version(), before + 3);

        db.delete(0, 1).unwrap();
        assert_eq!(live.version(), before + 4, "delete bumps exactly once");

        // Deleting an already-dead ordinal still bumps: the applier does not
        // branch on whether liveness actually moved. Over-invalidation is a
        // wasted re-materialization, never a correctness problem.
        db.delete(0, 1).unwrap();
        assert_eq!(live.version(), before + 5);

        db.close().unwrap();
    }

    /// DDL is a mutation, but not a LIVENESS mutation — a fresh collection has
    /// no rows. Bumping here would invalidate every other collection's readers
    /// for nothing.
    #[test]
    fn create_collection_does_not_bump() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        let live = live_handle(&db, 0);

        db.insert(0, &[1.0, 0.0], vec![]).unwrap();
        let after_insert = live.version();

        let id = db.create_collection("fresh", 64, vec_only(2)).unwrap();
        assert_eq!(live.version(), after_insert, "DDL must not bump");
        assert_eq!(
            live_handle(&db, id).version(),
            0,
            "a new collection starts at version 0"
        );

        db.close().unwrap();
    }

    /// A snapshot resolved over a real collection agrees with the authority it
    /// was copied from, and reading it costs exactly one materialization no
    /// matter how many times it is resolved.
    #[test]
    fn resolve_matches_the_live_bitmap() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, 64)], manual_opts()).unwrap();
        let live = live_handle(&db, 0);
        let meta = db.metadata_reader(0).unwrap();

        for i in 0..8 {
            db.insert(0, &[i as f32, 0.0], vec![]).unwrap();
        }
        db.delete(0, 3).unwrap();

        let set = live.resolve();
        assert_eq!(set.bitmap(), &meta.live(), "snapshot must match the authority");
        assert_eq!(set.len(), 7);
        assert!(!set.contains(3));
        assert_eq!(set.version(), 9, "8 inserts + 1 delete");

        // Re-resolving at an unchanged version is a pointer clone.
        assert!(Arc::ptr_eq(&set, &live.resolve()));
        assert_eq!(live.materializations(), 1);

        // A write invalidates it, and the OLD snapshot is unmoved — the row it
        // could see when it was taken, it can still see.
        db.delete(0, 4).unwrap();
        let newer = live.resolve();
        assert_eq!(newer.len(), 6);
        assert_eq!(set.len(), 7, "an outstanding snapshot never mutates");
        assert!(set.contains(4));
        assert_eq!(live.materializations(), 2);

        db.close().unwrap();
    }

    /// A lookup answers as of the SNAPSHOT it is handed, not as of now.
    ///
    /// This is what makes a query correspond to one instant. Before the
    /// snapshot became a parameter, every lookup masked against the metadata
    /// index's current liveness, so two predicates in one `WHERE` answered as
    /// of two different moments and their union could describe a row set that
    /// never existed.
    #[test]
    fn a_lookup_answers_as_of_its_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        db.insert(0, &[1.0, 0.0], meta_row(1, "alice")).unwrap(); // ord 0
        db.insert(0, &[2.0, 0.0], meta_row(2, "alice")).unwrap(); // ord 1
        db.insert(0, &[3.0, 0.0], meta_row(3, "bob")).unwrap(); // ord 2

        let meta = db.metadata_reader(0).unwrap();
        let before = db.live_snapshot(0).unwrap();

        db.delete(0, 1).unwrap();
        let after = db.live_snapshot(0).unwrap();
        assert!(!Arc::ptr_eq(&before, &after), "the delete moved the version");

        // The OLD snapshot still sees ordinal 1 — a query that opened before
        // the delete must keep seeing the row it was promised, predicate or no
        // predicate.
        let alice = Value::Text("alice".into());
        assert_eq!(
            meta.lookup_eq(1, &alice, &before)
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![0, 1],
            "a lookup against the old snapshot lost a row it should still see"
        );
        // The new one does not.
        assert_eq!(
            meta.lookup_eq(1, &alice, &after)
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![0]
        );

        // Same for ranges.
        assert_eq!(
            meta.lookup_range(0, RangeOp::Lt, &Value::Int(3), &before)
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![0, 1]
        );
        assert_eq!(
            meta.lookup_range(0, RangeOp::Lt, &Value::Int(3), &after)
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![0]
        );

        db.close().unwrap();
    }

    /// THE write-path gate: writing must never build a snapshot. Rebuilding a
    /// LiveSet per statement would cost a ~125KB bitmap copy per million rows,
    /// on the write path — the whole reason the version counter exists.
    ///
    /// This cannot fail today: nothing in the crate materializes yet. It is a
    /// tripwire planted before the code that could trip it, and it becomes
    /// load-bearing the moment lazy resolution lands.
    #[test]
    fn writes_materialize_nothing() {
        let rows = write_only_rows();
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg(0, 2, rows + 8)], manual_opts()).unwrap();
        let live = live_handle(&db, 0);

        for i in 0..rows {
            db.insert(0, &[i as f32, 0.0], vec![]).unwrap();
        }
        for o in 0..(rows as u64 / 4) {
            db.delete(0, o).unwrap();
        }

        assert_eq!(
            live.version(),
            (rows + rows / 4) as u64,
            "one bump per write"
        );
        assert_eq!(
            live.materializations(),
            0,
            "a write-only workload must not build a single snapshot"
        );
        assert!(live.cached().is_none(), "nothing should be cached");

        db.close().unwrap();
    }
}
