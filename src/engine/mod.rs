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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

mod compact;
pub mod cursor;

pub use crate::engine::compact::CompactionOptions;
use crate::engine::compact::{Compactor, Epoch, GateGuard, Shared, WriteGate};
use crate::engine::cursor::Cursor;
use crate::error::{Error, Result};
use crate::index::index::{FlatIndex, Ordinal, Reader, SearchResult, Writer};
use roaring::RoaringBitmap;

use crate::metadata::common::{ColumnId, Row, Schema, Value};
use crate::metadata::index::MetadataIndex;
use crate::metadata::tuples::TupleStore;
use crate::metadata::{crc32, index as meta, tuples, write_snapshot_atomic};
use crate::wal::wal::{Apply, Bitmap, Install, Lsn, Outcome, Record, RejectReason, Wal, WalHandle};

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

/// The largest vector dimension a collection may declare.
pub const MAX_DIM: usize = 65_536;

/// The most vector storage one collection may reserve: `capacity × dim × 4`
/// bytes. Checked, with [`MAX_DIM`] and the ordinal space, BEFORE a CREATE is
/// logged — a create that could not be materialized would otherwise be durable,
/// fail to apply, and poison the database on every reopen.
pub const MAX_COLLECTION_BYTES: u64 = 4 << 40; // 4 TiB

/// Runtime limits, changeable on an open database with [`Db::set_limits`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// The longest statement text accepted, in bytes. Default 256 MiB —
    /// room for a 10,000-row multi-row INSERT of 768-dim vectors as text.
    pub max_statement_bytes: usize,
    /// How long one statement may run before it is stopped with
    /// [`Error::Interrupted`]. Checked while searching and between rows of a
    /// scan; a write, once logged, always completes. Default: none.
    pub statement_timeout: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_statement_bytes: 256 * 1024 * 1024,
            statement_timeout: None,
        }
    }
}

/// Engine tunables.
#[derive(Debug, Clone, Copy)]
pub struct DbOptions {
    /// How often the background flusher runs a checkpoint. Set very large to
    /// effectively disable it and drive checkpoints manually via `checkpoint`.
    pub checkpoint_interval: Duration,
    /// When the background compactor rebuilds a collection. It wakes on
    /// `checkpoint_interval` too. See [`CompactionOptions`].
    pub compaction: CompactionOptions,
}

impl Default for DbOptions {
    fn default() -> Self {
        DbOptions {
            checkpoint_interval: Duration::from_secs(5),
            compaction: CompactionOptions::default(),
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
fn open_collection(
    dir: &Path,
    cfg: &CollectionConfig,
    salvage: bool,
) -> Result<(CollectionWriters, Collection, u64)> {
    // An interrupted compaction can leave the directory mid-swap; resolve that
    // first. A no-op otherwise.
    compact::settle(dir, cfg.id)?;
    open_collection_with(&collection_dir(dir, cfg.id), cfg, salvage)
}

/// `open_collection` against an explicit directory — the canonical one, or a
/// compaction's staging copy. Never salvages: both of its other callers open
/// directories they just created.
fn open_collection_in(
    cdir: &Path,
    cfg: &CollectionConfig,
) -> Result<(CollectionWriters, Collection, u64)> {
    open_collection_with(cdir, cfg, false)
}

/// The store file names, for `open_collection_with`'s refusals.
const VECTORS_FILE: &str = "vectors.idx";
const METADATA_FILE: &str = "metadata.snap";
const TUPLES_FILE: &str = "tuples.snap";

/// LOST-STORE POLICY. Opening never trades rows away silently — the posture
/// of SQLite (`SQLITE_CORRUPT`), Postgres (checksum failure), RocksDB
/// (`paranoid_checks`) and InnoDB alike.
///
/// A store whose file is missing or fails its checks comes up empty. That is
/// only correct when the WAL still holds every record the store ever applied,
/// and the proof is the collection's OTHER stores: WAL truncation never goes
/// past the lowest watermark of any store (`IndexApplier::checkpoint`), so if
/// every surviving store is still at watermark 0 — this generation was never
/// checkpointed — nothing it needs can have been truncated, and replay refills
/// the empty store exactly. That is the crash-before-first-checkpoint case the
/// old "start empty, replay rebuilds" fallback was built for.
///
/// Any surviving watermark above 0 means a checkpoint happened and the log may
/// have been cut past rows only the lost file held. A compaction flip counts:
/// it stamps the new generation with the flip's LSN, and records from before
/// it name ordinals that no longer exist. Then:
///
/// * by default, refuse with `CorruptSnapshot` naming the collection and the
///   file. Nothing on disk is touched, so restoring a backup stays possible.
/// * through `Db::open_salvage`, reset BOTH metadata stores (they hold liveness
///   and values together — keeping one would leave live rows with no values),
///   keep a damaged file as `<name>.damaged`, and set the collection's floor
///   to the vector index's watermark: what replays is exactly what the log
///   proves, never a pre-flip record onto renumbered rows, and the LSN counter
///   resumes above every checkpoint, so later writes can't be mistaken for
///   checkpointed ones. A lost vector index is not salvageable: without the
///   vectors there's nothing to salvage onto.
///
/// All three files missing is a collection being materialized for the first
/// time (open()-registration persists the catalog before the stores exist),
/// and opens empty as it always has.
fn open_collection_with(
    cdir: &Path,
    cfg: &CollectionConfig,
    salvage: bool,
) -> Result<(CollectionWriters, Collection, u64)> {
    use crate::metadata::Loaded;

    let cdir = cdir.to_path_buf();
    std::fs::create_dir_all(&cdir)?;

    let idx_path = cdir.join(VECTORS_FILE);
    let idx_existed = idx_path.exists();
    let (meta_w, meta_r, meta_loaded) = MetadataIndex::open_reporting(&cdir, cfg.schema.clone())?;
    let (tuple_w, tuple_r, tuple_loaded) = TupleStore::open_reporting(&cdir, cfg.schema.clone())?;

    let fresh = !idx_existed && meta_loaded == Loaded::Missing && tuple_loaded == Loaded::Missing;
    let lost = |l: &Loaded| !matches!(l, Loaded::Snapshot(_));
    // The highest watermark among the stores that survived: the evidence of
    // whether a checkpoint (and so a truncation) ever happened.
    let mut evidence = 0u64;
    if !lost(&meta_loaded) {
        evidence = evidence.max(meta_loaded.lsn().0);
    }
    if !lost(&tuple_loaded) {
        evidence = evidence.max(tuple_loaded.lsn().0);
    }

    let (flat_w, flat_r) = if idx_existed {
        FlatIndex::open(&idx_path)?
    } else {
        if !fresh && evidence > 0 {
            return Err(lost_store(cfg, VECTORS_FILE, &Loaded::Missing, false));
        }
        FlatIndex::create(&idx_path, cfg.schema.vector().dim, cfg.capacity)?
    };
    let flat_lsn = flat_w.checkpoint_lsn();
    evidence = evidence.max(flat_lsn);

    let any_lost = lost(&meta_loaded) || lost(&tuple_loaded);
    let (meta_w, meta_r, tuple_w, tuple_r, watermark) = if !any_lost {
        let watermark = flat_lsn.min(meta_loaded.lsn().0).min(tuple_loaded.lsn().0);
        (meta_w, meta_r, tuple_w, tuple_r, watermark)
    } else if fresh || evidence == 0 {
        // Rebuildable: the log holds everything. Write the empty snapshot now
        // (as `create` always has) so the next open finds a file, and let the
        // 0 drag the floor down: the full-tail replay refills this store while
        // the healthy ones absorb it idempotently.
        let (meta_w, meta_r) = if lost(&meta_loaded) {
            MetadataIndex::create(&cdir, cfg.schema.clone())?
        } else {
            (meta_w, meta_r)
        };
        let (tuple_w, tuple_r) = if lost(&tuple_loaded) {
            TupleStore::create(&cdir, cfg.schema.clone())?
        } else {
            (tuple_w, tuple_r)
        };
        (meta_w, meta_r, tuple_w, tuple_r, 0)
    } else if salvage {
        for (file, loaded) in [(METADATA_FILE, &meta_loaded), (TUPLES_FILE, &tuple_loaded)] {
            if let Loaded::Corrupt(_) = loaded {
                std::fs::rename(cdir.join(file), cdir.join(format!("{file}.damaged")))?;
            }
            if lost(loaded) {
                eprintln!("flats: salvage: {}", lost_store(cfg, file, loaded, true));
            }
        }
        let (meta_w, meta_r) = MetadataIndex::create(&cdir, cfg.schema.clone())?;
        let (tuple_w, tuple_r) = TupleStore::create(&cdir, cfg.schema.clone())?;
        (meta_w, meta_r, tuple_w, tuple_r, flat_lsn)
    } else {
        let (file, loaded) = if lost(&meta_loaded) {
            (METADATA_FILE, &meta_loaded)
        } else {
            (TUPLES_FILE, &tuple_loaded)
        };
        return Err(lost_store(cfg, file, loaded, false));
    };

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

/// The refusal for a store the log can't rebuild — or, when `salvaged`, the
/// notice that salvage gave up its rows.
fn lost_store(
    cfg: &CollectionConfig,
    file: &str,
    loaded: &crate::metadata::Loaded,
    salvaged: bool,
) -> Error {
    use crate::metadata::Loaded;
    let what = match loaded {
        Loaded::Corrupt(why) => format!("is damaged ({why})"),
        Loaded::Missing | Loaded::Snapshot(_) => "is missing".to_string(),
    };
    let head = format!("collection '{}' (id {}): {file} {what}", cfg.name, cfg.id);
    Error::CorruptSnapshot(if salvaged {
        format!(
            "{head}; opened with only the rows written since its last checkpoint, \
             the rest are gone"
        )
    } else {
        format!(
            "{head}, and the write-ahead log may no longer hold everything it did \
             (a checkpoint has run since), so opening could lose rows. Nothing was \
             changed. Restore a backup, or open it \
             once in salvage mode to keep only the rows written since the last \
             checkpoint"
        )
    })
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

impl CollectionWriters {
    /// Make all three stores durable, returning the lowest watermark among
    /// them.
    ///
    /// Flat-index order (a → b → c) is the crash-safety argument — do not
    /// reorder:
    ///   a. `sync_data`       — vector + tombstone pages durable
    ///   b. `stage_watermark` — write `(count, last_lsn)` into the spare slot
    ///   c. `sync_header`     — that slot's page durable, then it becomes active
    /// The metadata stores use serialize-and-rename snapshots (their own
    /// crash-safe protocol).
    fn checkpoint(&mut self) -> io::Result<u64> {
        // Snapshot BEFORE syncing so we never persist a watermark/count that
        // covers bytes the sync didn't flush.
        let (count, flat_lsn) = self.flat.begin_checkpoint();
        self.flat.sync_data().map_err(to_io)?; // a
        fail_point!("checkpoint.before_flat_header");
        self.flat.stage_watermark(count, flat_lsn).map_err(to_io)?; // b
        self.flat.sync_header().map_err(to_io)?; // c

        // THE METADATA INDEX GOES LAST. Its watermark must never be ahead of
        // the other stores': replay starts at the collection's slowest store
        // (`floors`), and the apply-time verdicts — is this key taken, has
        // this row been retired — are read off the metadata index. With it
        // the slowest, a crash mid-checkpoint replays from exactly the state
        // it holds, so every verdict comes out as it did the first time,
        // while the stores already ahead just absorb idempotent re-applies.
        // The other order let a crash leave the index one statement ahead:
        // replay then judged an INSERT against a later UPDATE's copy of the
        // same key, refused it as a duplicate, and the tuple store never got
        // the row (`tests/failpoints.rs`).
        fail_point!("checkpoint.before_tuples");
        let tuple_lsn = self.tuple.applied_lsn();
        self.tuple.checkpoint(tuple_lsn).map_err(to_io)?;
        fail_point!("checkpoint.before_metadata");
        let meta_lsn = self.meta.applied_lsn();
        self.meta.checkpoint(meta_lsn).map_err(to_io)?;

        Ok(flat_lsn.min(meta_lsn.0).min(tuple_lsn.0))
    }
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
    /// Per collection, the lowest durable watermark among its stores at open
    /// (and after a compaction, the flip's LSN). A data record at or below it
    /// is already reflected in every store of that collection, so `apply`
    /// skips it.
    ///
    /// For a collection that was never compacted this only saves work — the
    /// apply would have been an idempotent no-op. After a compaction it is
    /// load-bearing: a record from before the flip names an OLD ordinal, which
    /// in the new generation is a different row, and the WAL may still hold it
    /// (truncation is the global minimum). See `compact`.
    floors: HashMap<u32, u64>,
    /// Bumped around every compaction publish; see `compact::Epoch`.
    epoch: Arc<Epoch>,
    /// Set when a compaction left memory and disk disagreeing about which
    /// generation a collection is. Every later apply and checkpoint refuses,
    /// and the log halts (`Apply::halted`) before writing another frame.
    halted: Option<String>,
    /// One past the highest collection id EVER registered — shared with the
    /// `Db`, which assigns ids from it, and persisted in `catalog.snap`.
    ///
    /// Ids are never reused, so this is also how the applier knows a
    /// collection was DROPPED rather than never existing: an id below it that
    /// is not registered. That distinction is what lets replay pass over the
    /// records a dropped collection left in the log instead of halting on
    /// them, and what stops a replayed `CreateCollection` from resurrecting it.
    next_id: Arc<AtomicU32>,
}

/// The collection a data record writes to; `None` for DDL.
fn data_collection(record: &Record) -> Option<u32> {
    match record {
        Record::Insert { collection, .. }
        | Record::Delete { collection, .. }
        | Record::Mutate { collection, .. } => Some(*collection),
        Record::CreateCollection { .. } | Record::DropCollection { .. } => None,
    }
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
        if let Some(cause) = &self.halted {
            return Err(io::Error::other(cause.clone()));
        }
        // A data record for a DROPPED collection: a write that resolved it
        // before the drop committed and was logged after, or a record of it
        // that replay meets again. Either way there is nothing to apply it
        // to, and the verdict is the same on every replay — a refusal, not an
        // apply failure, so it must not halt the log.
        if let Some(collection) = data_collection(record)
            && self.is_dropped(collection)
        {
            self.last_applied = self.last_applied.max(lsn.0);
            return Ok(Outcome::Rejected(RejectReason::CollectionDropped));
        }
        // Already reflected in every store of its collection — see `floors`.
        if let Some(collection) = data_collection(record)
            && lsn.0 <= self.floors.get(&collection).copied().unwrap_or(0)
        {
            self.last_applied = self.last_applied.max(lsn.0);
            return Ok(Outcome::Applied);
        }
        match record {
            Record::Insert {
                collection,
                ordinal,
                vector,
                metadata,
            } => {
                // THE UNIQUENESS DECISION, and it has to be made here.
                //
                // The caller already checked, but a check on a caller's thread
                // settles nothing: two of them can read "absent" and both
                // append. This thread is the only serialization point in the
                // system, so this is the only place where "no live row holds
                // this key" is still true by the time the row goes in.
                //
                // A refusal is NOT an error. Returning `Err` here would poison
                // the log (an apply failure means the index could not absorb a
                // committed record), and a repeated key is an ordinary caller
                // mistake, not a broken database. It is also not a conflict:
                // every replay of this frame reaches the same verdict, because
                // the winner's row is live by then — which is exactly the
                // determinism replay needs.
                if let Some(reason) = self.duplicate_key(*collection, metadata, *ordinal)? {
                    // The watermarks still advance, for the same reason a
                    // conflicted Mutate's do: a refused record IS fully
                    // reflected (it changed nothing), and the verdict is
                    // deterministic, so replay reaches it again and refuses it
                    // again. Stalling them here would wedge truncation behind
                    // a frame that will never do anything — one duplicate key
                    // would stop the log from ever shedding bytes.
                    let w = self.writers(*collection)?;
                    w.flat.advance_applied_lsn(lsn.0);
                    w.meta.advance_applied_lsn(lsn);
                    w.tuple.advance_applied_lsn(lsn);
                    self.last_applied = self.last_applied.max(lsn.0);
                    return Ok(Outcome::Rejected(reason));
                }
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
            Record::DropCollection { id } => {
                self.apply_drop(*id)?;
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

                // UNIQUENESS, decided here for the same reason an `Insert`'s
                // is: this thread is the only place where "no live row holds
                // this key" cannot change between asking and acting. Two bulk
                // inserts can both pass their callers' checks; only one gets
                // past this. A refusal is a verdict, not an apply failure, and
                // it is deterministic — replay reaches it again — so the
                // watermarks advance exactly as for a conflict.
                if let Some(reason) = self.install_conflict(*collection, &retire.0, install)? {
                    let w = self.writers(*collection)?;
                    w.flat.advance_applied_lsn(lsn.0);
                    w.meta.advance_applied_lsn(lsn);
                    w.tuple.advance_applied_lsn(lsn);
                    self.last_applied = self.last_applied.max(lsn.0);
                    return Ok(Outcome::Rejected(reason));
                }
                let w = self.writers(*collection)?;

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
        if let Some(cause) = &self.halted {
            return Err(io::Error::other(cause.clone()));
        }
        // Every record at or below `last_applied` has been fully processed,
        // for every collection: applied, or refused/conflicted (which also
        // leaves the stores fully reflecting it). A store whose collection
        // saw none of the records since its own last write is therefore just
        // as caught up as the busiest one, so it is durable through
        // `last_applied` too. Without this, a store's watermark only moves
        // when its own collection is written: an idle collection sat at 0
        // forever, and since truncation stops at the lowest watermark, it
        // held the whole log back — the file grew without bound and every
        // open replayed all of it. (Compaction stamps its new generation with
        // `last_applied` on the same argument.) A failed apply halts the log
        // before this can run, so there is never a hole below it.
        let through = self.last_applied;
        let mut min_durable = u64::MAX;
        for w in self.writers.values_mut() {
            w.flat.advance_applied_lsn(through);
            w.meta.advance_applied_lsn(Lsn(through));
            w.tuple.advance_applied_lsn(Lsn(through));
            min_durable = min_durable.min(w.checkpoint()?);
        }

        // The catalog is the FOURTH store: persist the registry stamped with
        // its watermark and fold that stamp into the min. This is what stops
        // truncation from ever dropping a CreateCollection record whose
        // registration isn't durably on disk yet.
        let cat_lsn = self.catalog_watermark();
        fail_point!("checkpoint.before_catalog");
        persist_catalog(&self.dir, &self.registry, cat_lsn, self.next_id()).map_err(to_io)?;
        min_durable = min_durable.min(cat_lsn);

        Ok(if min_durable == u64::MAX || min_durable == 0 {
            None
        } else {
            Some(min_durable)
        })
    }

    fn control(&mut self, job: Box<dyn std::any::Any + Send>) -> io::Result<Option<u64>> {
        if let Some(cause) = &self.halted {
            return Err(io::Error::other(cause.clone()));
        }
        self.run_control(job)
    }

    fn halted(&self) -> Option<String> {
        self.halted.clone()
    }
}

impl IndexApplier {
    /// `Some(DuplicateKey)` when `row`'s primary key is already held by a live
    /// row of `collection`; `None` when the collection declares no key, or the
    /// key is free.
    ///
    /// Reads the key column off the REGISTERED schema rather than the record,
    /// so a frame cannot nominate its own key column.
    ///
    /// `own` is the ordinal this record installs. A key held by THAT row is
    /// the record itself, already applied by a store that was checkpointed
    /// further than the others: a crash between a checkpoint's metadata and
    /// tuple snapshots leaves exactly that, and recovery replays the record
    /// to catch the lagging store up. Counting its own row as a rival would
    /// refuse the replay and leave the tuple store without the row. The
    /// verdict is still deterministic: a real duplicate is held by a
    /// DIFFERENT ordinal, which every replay sees the same way.
    fn duplicate_key(
        &mut self,
        collection: u32,
        row: &Row,
        own: u64,
    ) -> io::Result<Option<RejectReason>> {
        let Some(key) = self
            .registry
            .iter()
            .find(|c| c.id == collection)
            .and_then(|c| c.schema.primary_key)
        else {
            return Ok(None);
        };
        let Some(value) = row
            .iter()
            .find(|(id, _)| *id == key)
            .map(|(_, v)| v.clone())
        else {
            // Every column is present by the time a record is logged
            // (`validate_row`), so a missing key column would be a
            // contradiction rather than a caller error.
            return Ok(None);
        };
        let w = self.writers(collection)?;
        let mut own_rows = RoaringBitmap::new();
        own_rows.insert(ordinal32(own)?.0);
        if w.meta
            .any_key_held_outside(key, &[&value], &own_rows)
            .map_err(to_io)?
        {
            return Ok(Some(RejectReason::DuplicateKey));
        }
        Ok(None)
    }

    /// The primary-key verdict for a `Record::Mutate`: a key repeated among
    /// its installs, or held by a live row the record does not retire.
    ///
    /// Records that predate this check could only reach it with a key that
    /// was ALREADY duplicated — UPDATE cannot assign a key, and its copies'
    /// keys are freed by the rows it retires — so replaying an older log
    /// reaches the verdict it always effectively had.
    fn install_conflict(
        &mut self,
        collection: u32,
        retire: &RoaringBitmap,
        install: &[Install],
    ) -> io::Result<Option<RejectReason>> {
        let Some(key) = self
            .registry
            .iter()
            .find(|c| c.id == collection)
            .and_then(|c| c.schema.primary_key)
        else {
            return Ok(None);
        };
        let values: Vec<&Value> = install
            .iter()
            .filter_map(|r| r.metadata.iter().find(|(id, _)| *id == key).map(|(_, v)| v))
            .collect();
        let mut seen = std::collections::HashSet::with_capacity(values.len());
        if values.iter().any(|v| !seen.insert(render_key(v))) {
            return Ok(Some(RejectReason::DuplicateKey));
        }
        // Keys held by rows this record retires are freed by it; keys held
        // by rows it INSTALLS are the record itself, already applied by a
        // store checkpointed ahead of the others (see `duplicate_key`).
        let mut own_rows = retire.clone();
        for r in install {
            own_rows.insert(ordinal32(r.ordinal)?.0);
        }
        let w = self.writers(collection)?;
        if w.meta
            .any_key_held_outside(key, &values, &own_rows)
            .map_err(to_io)?
        {
            return Ok(Some(RejectReason::DuplicateKey));
        }
        Ok(None)
    }

    /// Whether `id` names a collection that existed and was dropped: issued
    /// (below `next_id`, since ids are never reused) but no longer registered.
    fn is_dropped(&self, id: u32) -> bool {
        id < self.next_id() && !self.writers.contains_key(&id)
    }

    fn next_id(&self) -> u32 {
        self.next_id.load(Ordering::Acquire)
    }

    /// Apply a `Record::DropCollection`.
    ///
    /// ORDER, and why:
    ///   1. persist `catalog.snap` without the collection — the COMMIT POINT.
    ///      From here a restart never opens it again: `next_id` still covers
    ///      its id, so it reads as dropped, and replay passes over every
    ///      record of it.
    ///   2. unpublish it from the read path, so new statements cannot find it.
    ///      A read already holding it keeps its own handles; unlinked files
    ///      stay mapped until the last one lets go.
    ///   3. release its writers.
    ///   4. delete its files. A crash before this leaves them behind, and
    ///      `Db::open` removes leftovers of dropped ids before anything opens.
    ///
    /// A drop of a collection that is not registered — replay of a drop the
    /// catalog already reflects — only makes sure no files of it remain.
    fn apply_drop(&mut self, id: u32) -> io::Result<()> {
        if self.registry.iter().any(|c| c.id == id) {
            let registry: Vec<CollectionConfig> =
                self.registry.iter().filter(|c| c.id != id).cloned().collect();
            persist_catalog(&self.dir, &registry, self.catalog_watermark(), self.next_id())
                .map_err(to_io)?;
            self.registry = registry;

            let mut guard = self.catalog.write().unwrap_or_else(|e| e.into_inner());
            let mut map: CatalogMap = (**guard).clone();
            map.remove(&id);
            *guard = Arc::new(map);
            drop(guard);

            self.writers.remove(&id);
            self.floors.remove(&id);
        }
        // Best effort: the catalog no longer names it, so a file that will not
        // go now is a leftover the next open clears — not a reason to halt a
        // healthy database.
        if let Err(e) = remove_collection_files(&self.dir, id) {
            eprintln!("flats: dropped collection {id}: files left for the next open to clear: {e}");
        }
        Ok(())
    }

    /// The backup control job: checkpoint, then copy every file a `Db::open`
    /// needs into `dest`. Runs on the commit thread, between records, so no
    /// apply, checkpoint or compaction flip can touch a file mid-copy.
    ///
    /// The copy holds the log as well as the checkpointed stores, so it is the
    /// database as of THIS point in the log even if a store is behind it. The
    /// returned truncation point is applied to the source log after the copy.
    pub(super) fn backup_into(&mut self, dest: &Path) -> io::Result<Option<u64>> {
        let up_to = self.checkpoint()?;
        copy_file_synced(&self.dir.join(CATALOG_FILE), &dest.join(CATALOG_FILE))?;
        for cfg in &self.registry {
            let from = collection_dir(&self.dir, cfg.id);
            let to = collection_dir(dest, cfg.id);
            std::fs::create_dir_all(&to)?;
            for entry in std::fs::read_dir(&from)? {
                let entry = entry?;
                let name = entry.file_name();
                // A `.tmp` is a snapshot write that never renamed into place.
                if !entry.file_type()?.is_file() || name.to_string_lossy().ends_with(".tmp") {
                    continue;
                }
                copy_file_synced(&entry.path(), &to.join(&name))?;
            }
            std::fs::File::open(&to)?.sync_all()?;
        }
        let wal = Db::wal_path(&self.dir);
        if wal.exists() {
            copy_file_synced(&wal, &Db::wal_path(dest))?;
        }
        std::fs::File::open(dest)?.sync_all()?;
        Ok(up_to)
    }

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
        // Replay meeting the create of a collection that was later dropped.
        // Materializing it only for the drop further down the log to delete
        // it again would be correct but wasteful — and briefly re-register
        // its name, which a newcomer may hold by now.
        if self.is_dropped(config.id) {
            return Ok(());
        }
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
            let (writers, collection, watermark) = open_collection(&self.dir, config, false)?;
            self.floors.insert(config.id, watermark);
            // The new collection-{id}/ entry itself must survive a crash: the
            // stores fsync'd their own files, but the parent dir entry is
            // separate. (Replay would heal it anyway; this just makes the
            // common case not need healing.)
            std::fs::File::open(&self.dir)?.sync_all()?;

            // Register durably BEFORE publishing anything in memory.
            let mut registry = self.registry.clone();
            registry.push(config.clone());
            registry.sort_by_key(|c| c.id);
            self.next_id.fetch_max(config.id.saturating_add(1), Ordering::AcqRel);
            persist_catalog(
                &self.dir,
                &registry,
                self.catalog_watermark(),
                self.next_id(),
            )?;
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
    fn spawn(wal: WalHandle, interval: Duration) -> io::Result<Flusher> {
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
            })?;
        Ok(Flusher {
            stop: stop_tx,
            join,
        })
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
/// 4 = + `next_id` (DROP COLLECTION: ids are never reused, so the catalog must
/// remember the highest it ever issued even after that collection is gone).
/// Version 3 files still read: nothing was ever dropped under them, so their
/// `next_id` is one past the highest registered id.
const CATALOG_VERSION: u32 = 4;
const CATALOG_V3: u32 = 3;
const CATALOG_FILE: &str = "catalog.snap";
const CATALOG_TMP: &str = "catalog.snap.tmp";

fn encode_catalog(registry: &[CollectionConfig], last_lsn: u64, next_id: u32) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(CATALOG_MAGIC);
    buf.extend_from_slice(&CATALOG_VERSION.to_le_bytes());
    buf.extend_from_slice(&last_lsn.to_le_bytes());
    buf.extend_from_slice(&next_id.to_le_bytes());

    // Plain data into memory — no fallible step today, but surfaced rather
    // than unwrapped (same reasoning as Record::encode in wal.rs).
    let body = bincode::serialize(&registry).map_err(io::Error::other)?;
    buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
    buf.extend_from_slice(&body);

    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

/// What `catalog.snap` holds: the registry, its LSN stamp, and one past the
/// highest collection id ever issued.
struct LoadedCatalog {
    registry: Vec<CollectionConfig>,
    last_lsn: u64,
    next_id: u32,
}

/// One past the highest id in `registry`: the `next_id` of a catalog that
/// never dropped anything.
fn next_id_after(registry: &[CollectionConfig]) -> u32 {
    registry
        .iter()
        .map(|c| c.id.saturating_add(1))
        .max()
        .unwrap_or(0)
}

/// Inverse of `encode_catalog`, for versions 3 and 4. Trailing CRC verified
/// FIRST, like the stores.
fn decode_catalog(bytes: &[u8]) -> Result<LoadedCatalog> {
    let corrupt = |why: &str| Error::CorruptSnapshot(format!("catalog: {why}"));
    let u32_at = |at: usize| -> Result<u32> {
        bytes
            .get(at..at + 4)
            .and_then(|b| b.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| corrupt("truncated"))
    };

    // magic + version + last_lsn + body_len + trailing crc (v3's minimum).
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
    let version = u32_at(4)?;
    if version != CATALOG_VERSION && version != CATALOG_V3 {
        return Err(Error::CorruptSnapshot(format!(
            "catalog: unsupported version {version}"
        )));
    }
    let last_lsn = u64::from_le_bytes(head[8..16].try_into().map_err(|_| corrupt("truncated"))?);
    // v4 carries next_id between the stamp and the body length.
    let (stored_next_id, body_at) = if version == CATALOG_VERSION {
        (Some(u32_at(16)?), 20)
    } else {
        (None, 16)
    };
    let body_len = u32_at(body_at)? as usize;
    let body_start = body_at + 4;
    let body = head
        .get(body_start..body_start + body_len)
        .ok_or_else(|| corrupt("body length overruns file"))?;
    if head.len() != body_start + body_len {
        return Err(corrupt("trailing bytes"));
    }

    let registry: Vec<CollectionConfig> = bincode::deserialize(body)
        .map_err(|e| Error::CorruptSnapshot(format!("catalog body: {e}")))?;
    // Never below what the registry itself proves was issued.
    let next_id = stored_next_id
        .unwrap_or(0)
        .max(next_id_after(&registry));
    Ok(LoadedCatalog {
        registry,
        last_lsn,
        next_id,
    })
}

/// Load the persisted registry + its watermark. Missing file = fresh database
/// (empty at 0); corrupt file = Err — see the corruption-policy note above.
fn load_catalog(dir: &Path) -> Result<LoadedCatalog> {
    match std::fs::read(dir.join(CATALOG_FILE)) {
        Ok(bytes) => decode_catalog(&bytes),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(LoadedCatalog {
            registry: Vec::new(),
            last_lsn: 0,
            next_id: 0,
        }),
        Err(e) => Err(e.into()),
    }
}

/// Persist the registry with the shared tmp → fsync → rename → dir-fsync
/// dance (a leftover garbage .tmp from a crash is overwritten, never read).
/// `create_dir_all`, then fsync the parent of every directory it created,
/// deepest first. A new directory entry is durable only once its parent is
/// synced: without this, a power cut after the first acknowledged write could
/// take the whole database directory with it (`tests/sync_order.rs`).
fn create_dir_durably(dir: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut at = Some(dir);
    while let Some(d) = at {
        if d.as_os_str().is_empty() || d.exists() {
            break;
        }
        missing.push(d.to_path_buf());
        at = d.parent();
    }
    std::fs::create_dir_all(dir)?;
    for d in &missing {
        if let Some(parent) = d.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn persist_catalog(
    dir: &Path,
    registry: &[CollectionConfig],
    last_lsn: u64,
    next_id: u32,
) -> Result<()> {
    fail_point!("catalog.before_write");
    write_snapshot_atomic(
        dir,
        CATALOG_TMP,
        CATALOG_FILE,
        &encode_catalog(registry, last_lsn, next_id)?,
    )
}

/// Delete everything of collection `id` from `dir`: its canonical directory
/// and any compaction staging or retired copy. Missing ones are fine.
fn remove_collection_files(dir: &Path, id: u32) -> io::Result<()> {
    for path in [
        collection_dir(dir, id),
        compact::staging_dir(dir, id),
        compact::retired_dir(dir, id),
    ] {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    std::fs::File::open(dir)?.sync_all()
}

/// The collection id a `collection-{id}` directory entry (or its `.compact` /
/// `.old` sibling) belongs to; `None` for anything else.
fn collection_id_of_entry(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("collection-")?;
    let digits = rest
        .strip_suffix(".compact")
        .or_else(|| rest.strip_suffix(".old"))
        .unwrap_or(rest);
    digits.parse().ok()
}

/// Copy `from` to `to` and make the copy durable before returning.
fn copy_file_synced(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::copy(from, to)?;
    std::fs::File::open(to)?.sync_all()
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
    /// Held shared by every write while it is in flight, so a compaction's
    /// flip can wait them out. See `compact::WriteGate`.
    gate: Arc<WriteGate>,
    /// Bumped around every compaction publish, so a query can tell it straddled
    /// one. See `compact::Epoch`.
    epoch: Arc<Epoch>,
    /// What a compaction needs; shared with the background compactor.
    compaction: Arc<Shared>,
    /// The background compactor, unless `CompactionOptions::enabled` is off.
    compactor: Option<Compactor>,
    /// One past the highest collection id ever issued; shared with the
    /// applier, which advances it as creates land. Ids are never reused.
    next_id: Arc<AtomicU32>,
    /// See [`Limits`]; read at the start of every statement.
    limits: RwLock<Limits>,
    /// Bumped by [`Db::interrupt`]; a statement started in an older
    /// generation stops at its next check. See `crate::cancel`.
    interrupts: Arc<AtomicU64>,
    /// The exclusive lock on `<dir>/LOCK`. LAST on purpose: fields drop in
    /// declaration order, after `Drop::drop` has shut everything down, so the
    /// directory is released only once nothing can touch it any more.
    _lock: std::fs::File,
}

/// Refuse a collection that could never be materialized — checked before a
/// CREATE is logged, so it can never become a durable record that fails to
/// apply on every replay.
fn check_collection_size(dim: usize, capacity: usize) -> Result<()> {
    if dim > MAX_DIM {
        return Err(Error::InvalidDimension);
    }
    // Ordinals are u32 (the metadata bitmaps' address space).
    if capacity > u32::MAX as usize {
        return Err(Error::InvalidCapacity);
    }
    let bytes = dim as u128 * capacity as u128 * 4;
    if bytes > u128::from(MAX_COLLECTION_BYTES) {
        return Err(Error::CollectionTooLarge {
            bytes,
            cap: MAX_COLLECTION_BYTES,
        });
    }
    Ok(())
}

/// Name of the lock file in a database directory.
const LOCK_FILE: &str = "LOCK";

/// Take the exclusive lock that makes this the only handle on `dir`.
///
/// An OS advisory lock (`flock` on Linux) on `<dir>/LOCK`, held by the
/// returned file for the life of the `Db`. It belongs to the open file, not
/// to the process, so a second `Db::open` of the same directory in the SAME
/// process is refused too; and the OS drops it when the process dies, so a
/// crash never leaves a database locked. The file's contents are unused.
fn lock_directory(dir: &Path) -> Result<std::fs::File> {
    let path = dir.join(LOCK_FILE);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(Error::DatabaseLocked {
            path: path.display().to_string(),
        }),
        Err(std::fs::TryLockError::Error(e)) => Err(e.into()),
    }
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
    pub fn open(
        dir: impl AsRef<Path>,
        collections: &[CollectionConfig],
        opts: DbOptions,
    ) -> Result<Db> {
        Self::open_inner(dir.as_ref(), collections, opts, false)
    }

    /// [`open`](Db::open), but get past a collection whose store file is
    /// damaged or missing in a way the write-ahead log can't repair, giving up
    /// the rows only that file held.
    ///
    /// Without this, such a database refuses to open with
    /// [`Error::CorruptSnapshot`] (naming the collection and the file) and
    /// nothing on disk is touched, so restoring a backup is still possible.
    /// Prefer that. Salvage is for when there is no backup.
    ///
    /// Each damaged collection opens holding only the rows written since its
    /// last checkpoint (the ones the log still has) and is fully usable; the
    /// damaged file is kept beside it as `<name>.damaged`, and each salvage is
    /// reported on stderr. Healthy collections open as usual. It does not get
    /// past a damaged `catalog.snap` (the list of collections, which no log
    /// can rebuild) or a damaged vector index. Use it for one open, then go
    /// back to [`open`](Db::open): once salvaged, the database is healthy.
    pub fn open_salvage(
        dir: impl AsRef<Path>,
        collections: &[CollectionConfig],
        opts: DbOptions,
    ) -> Result<Db> {
        Self::open_inner(dir.as_ref(), collections, opts, true)
    }

    fn open_inner(
        dir: &Path,
        collections: &[CollectionConfig],
        opts: DbOptions,
        salvage: bool,
    ) -> Result<Db> {
        create_dir_durably(dir)?;
        // FIRST, before reading or writing anything: one handle per directory.
        let lock = lock_directory(dir)?;

        // 0. The registry: persisted catalog ∪ caller-supplied configs.
        //    (Runtime registration goes through the WAL — `create_collection`;
        //    this open-time path exists for embedders that declare their
        //    collections up front.) Persist changes before touching any store,
        //    preserving the loaded watermark: registering a collection doesn't
        //    change which WAL records the file reflects.
        let LoadedCatalog {
            mut registry,
            last_lsn: cat_lsn,
            mut next_id,
        } = load_catalog(dir)?;
        let mut dirty = !dir.join(CATALOG_FILE).exists();
        for cfg in collections {
            match registry.iter().find(|c| c.id == cfg.id) {
                Some(existing) if existing == cfg => {}
                Some(_) => return Err(Error::CollectionConfigMismatch { id: cfg.id }),
                // An id below `next_id` that is not registered was DROPPED.
                // Reviving it would hand the newcomer every record the old
                // collection left in the log.
                None if cfg.id < next_id => {
                    return Err(Error::CollectionConfigMismatch { id: cfg.id });
                }
                None => {
                    if registry.iter().any(|c| c.name == cfg.name) {
                        return Err(Error::CollectionExists {
                            name: cfg.name.clone(),
                        });
                    }
                    registry.push(cfg.clone());
                    next_id = next_id.max(cfg.id.saturating_add(1));
                    dirty = true;
                }
            }
        }
        registry.sort_by_key(|c| c.id);
        if dirty {
            persist_catalog(dir, &registry, cat_lsn, next_id)?;
        }

        // 0b. Clear what an interrupted DROP left behind: directories of ids
        //     that were issued but are no longer registered. The catalog
        //     stopped naming them at the drop's commit point, so nothing will
        //     ever open them again. Ids at or above `next_id` are left alone —
        //     that is a CREATE whose registration did not land, and replay
        //     materializes it over whatever is there.
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if let Some(id) = collection_id_of_entry(&entry.file_name().to_string_lossy())
                && id < next_id
                && !registry.iter().any(|c| c.id == id)
                && entry.file_type()?.is_dir()
            {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        let next_id = Arc::new(AtomicU32::new(next_id));

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
        let mut writers: HashMap<u32, CollectionWriters> = HashMap::with_capacity(registry.len());
        // Conservative global minimum watermark across every store of every
        // collection (see the recovery note below).
        let mut min_watermark = u64::MAX;
        let mut floors: HashMap<u32, u64> = HashMap::with_capacity(registry.len());
        for cfg in &registry {
            let (collection_writers, collection, watermark) =
                open_collection(dir, cfg, salvage)?;
            min_watermark = min_watermark.min(watermark);
            floors.insert(cfg.id, watermark);
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
        let skip_through = if skip_through == u64::MAX {
            0
        } else {
            skip_through
        };

        // 3. Start the WAL; recovery replays the tail into the writers (which the
        //    applier owns). The catalog's readers observe the same inners; a
        //    replayed CreateCollection publishes straight into the shared map.
        let wal_path = Self::wal_path(dir);
        let epoch = Arc::new(Epoch::new());
        let applier = IndexApplier {
            dir: dir.to_path_buf(),
            writers,
            registry,
            catalog: catalog.clone(),
            last_applied: skip_through,
            failed_create: None,
            floors,
            epoch: Arc::clone(&epoch),
            halted: None,
            next_id: Arc::clone(&next_id),
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
        //    A failed spawn must not leave the commit thread running detached
        //    (`Wal` has no Drop), so tear down what has started, like `Drop`.
        let flusher = match Flusher::spawn(wal.handle(), opts.checkpoint_interval) {
            Ok(flusher) => flusher,
            Err(e) => {
                wal.shutdown();
                return Err(e.into());
            }
        };

        // 6. Background compaction, on the same cadence.
        let gate = Arc::new(WriteGate::new());
        let compaction = Arc::new(Shared {
            dir: dir.to_path_buf(),
            catalog: catalog.clone(),
            gate: Arc::clone(&gate),
            opts: opts.compaction,
            running: Mutex::new(()),
            stop: std::sync::atomic::AtomicBool::new(false),
        });
        let compactor = if opts.compaction.enabled {
            match Compactor::spawn(
                Arc::clone(&compaction),
                wal.handle(),
                opts.checkpoint_interval,
            ) {
                Ok(compactor) => Some(compactor),
                Err(e) => {
                    flusher.stop();
                    wal.shutdown();
                    return Err(e.into());
                }
            }
        } else {
            None
        };

        Ok(Db {
            catalog,
            ddl: Mutex::new(()),
            wal: Some(wal),
            flusher: Some(flusher),
            gate,
            epoch,
            compaction,
            compactor,
            next_id,
            limits: RwLock::new(Limits::default()),
            interrupts: Arc::new(AtomicU64::new(0)),
            _lock: lock,
        })
    }

    /// Insert `vector` plus its metadata `row` into `collection` as ONE
    /// durable record. Blocks until durable. Returns the vector's ordinal
    /// (its position within the collection, shared by all three indexes).
    /// Vector-only collections (empty schema) pass an empty row.
    ///
    /// The ordinal is stable only until the collection is next compacted,
    /// which renumbers every row; a row's lasting identity is its primary key.
    pub fn insert(&self, collection: u32, vector: &[f32], row: Row) -> Result<Ordinal> {
        let _gate = self.gate.enter();
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
        // A key some live row already holds is refused HERE, before the frame
        // is written, so the common case costs the caller nothing and the
        // error can name the column and the value.
        //
        // This check alone is NOT the guarantee: two callers can both read
        // "absent" and both append. The authoritative check runs at apply
        // time, on the single commit thread, where the answer cannot change
        // underneath it — see the `Record::Insert` arm. This one exists for
        // the diagnostic, not the invariant.
        if let Some(key) = coll.config.schema.primary_key {
            self.check_duplicate_key(&coll, key, &row)?;
        }
        // Refuse before allocating: a halted log is going to reject this
        // append anyway, and burning an ordinal for a write that never reaches
        // the file costs the collection a slot of its capacity for nothing.
        self.check_poison()?;
        let ordinal = coll.alloc_ordinal()?;
        self.allocated(&coll, ordinal, ordinal + 1);
        // The row is moved into the record below, so capture the two strings
        // an error would need WHILE the row is still here. A collection with
        // no key cannot be refused, hence the `Option`.
        let key_described = coll.config.schema.primary_key.map(|key| {
            let value = row
                .iter()
                .find(|(id, _)| *id == key)
                .map(|(_, v)| render_key(v))
                // `validate_row` above proved every column is present, so this
                // is unreachable; described rather than unwrapped, because
                // library code does not panic on an invariant it can state.
                .unwrap_or_else(|| "<missing>".to_string());
            (Self::key_name(&coll.config.schema, key), value)
        });
        match self.wal_handle()?.append(Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
            metadata: row,
        }) {
            // REFUSED on the commit thread: the frame is durable but the row
            // was not folded in, because a live row already holds this key.
            // The pre-WAL check above missed it, which is exactly the race it
            // cannot see — so the error is reported from here instead.
            //
            // An insert has exactly one way to be refused, so the reason needs
            // no inspection; a keyless collection has no way at all, which is
            // why the `None` arm cannot be reached from a rejection.
            // The collection was dropped between our lookup and the commit.
            Ok(committed)
                if committed.outcome == Outcome::Rejected(RejectReason::CollectionDropped) =>
            {
                Err(Error::UnknownCollection { id: collection })
            }
            Ok(committed) if !committed.applied() => match key_described {
                Some((column, value)) => Err(Error::DuplicateKey { column, value }),
                None => Err(Error::DuplicateKey {
                    column: "<none>".to_string(),
                    value: "<none>".to_string(),
                }),
            },
            Ok(_committed) => Ok(Ordinal(ordinal as u32)),
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

    /// Insert many rows as ONE durable record: all of them, or none.
    ///
    /// Each element is `(vector, row)`, exactly what [`insert`](Self::insert)
    /// takes. The batch is logged as a single `Record::Mutate` that retires
    /// nothing — one frame, one CRC, one fsync — so it costs one commit instead
    /// of one per row, and a crash either keeps every row or none.
    ///
    /// Returns each row's ordinal, in input order. Like `insert`'s, they are
    /// stable only until the collection is next compacted.
    ///
    /// Everything that can be refused is refused BEFORE an ordinal is claimed,
    /// so a refused batch costs the collection no capacity: a wrong dimension
    /// or malformed row anywhere, a primary key repeated inside the batch or
    /// already held by a live row (both [`Error::DuplicateKey`], naming the
    /// first offender), a payload over [`MAX_INSTALL_BYTES`]
    /// ([`Error::StatementTooLarge`]), or more rows than the collection has
    /// room for ([`Error::CapacityExceeded`]).
    ///
    /// The key checks here are the diagnostic, not the guarantee — two
    /// batches can both pass them. The commit thread re-checks every installed
    /// key before folding the record in, and refuses the whole record if one
    /// is taken; that refusal surfaces here as the same `DuplicateKey`.
    ///
    /// An empty batch succeeds without logging anything.
    pub fn insert_many(&self, collection: u32, rows: Vec<(Vec<f32>, Row)>) -> Result<Vec<Ordinal>> {
        let _gate = self.gate.enter();
        let coll = self.collection(collection)?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // 1. VALIDATE every row, in input order, so the error names the first
        //    offender a caller would find by reading their own batch.
        let dim = coll.config.schema.vector().dim.get();
        for (vector, row) in &rows {
            if vector.len() != dim {
                return Err(Error::DimensionMismatch {
                    expected: dim,
                    got: vector.len(),
                });
            }
            coll.config.schema.validate_row(row)?;
        }
        if let Some(key) = coll.config.schema.primary_key {
            self.check_batch_keys(&coll, key, rows.iter().map(|(_, row)| row))?;
        }
        self.check_poison()?;

        // 2. SIZE, before any ordinal exists. `mutate` checks it too, but only
        //    after the caller has reserved — too late to keep the capacity.
        //    The ordinal is a fixed-width field, so a placeholder measures the
        //    same as the real one.
        let mut install: Vec<Install> = rows
            .into_iter()
            .map(|(vector, metadata)| Install {
                ordinal: 0,
                vector,
                metadata,
            })
            .collect();
        let bytes = bincode::serialized_size(&install)
            .map_err(|e| Error::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?
            as usize;
        if bytes > MAX_INSTALL_BYTES {
            return Err(Error::StatementTooLarge {
                bytes,
                cap: MAX_INSTALL_BYTES,
            });
        }

        // 3. CLAIM one consecutive run — all of it, or a refusal that burns
        //    nothing.
        let count = install.len() as u64;
        let start = coll.alloc_ordinals(count)?;
        self.allocated(&coll, start, start + count);
        for (i, row) in install.iter_mut().enumerate() {
            row.ordinal = start + i as u64;
        }
        let ordinals: Vec<Ordinal> = (start..start + count).map(|o| Ordinal(o as u32)).collect();

        // 4. ONE record. A batch retires nothing, so it can never be stale —
        //    `Conflicted` is unreachable — and the only refusal left is the
        //    commit thread's key check.
        let keys: Vec<Row> = match coll.config.schema.primary_key {
            Some(_) => install.iter().map(|r| r.metadata.clone()).collect(),
            None => Vec::new(),
        };
        match self.mutate(collection, RoaringBitmap::new(), install)? {
            Outcome::Applied => Ok(ordinals),
            Outcome::Rejected(RejectReason::CollectionDropped) => {
                Err(Error::UnknownCollection { id: collection })
            }
            Outcome::Rejected(RejectReason::DuplicateKey) | Outcome::Conflicted => {
                // Another writer took one of these keys between our check and
                // the commit. Name it: by now the winner is live, so the same
                // check that missed it before finds it.
                if let Some(key) = coll.config.schema.primary_key {
                    self.check_batch_keys(&coll, key, keys.iter())?;
                }
                Err(Error::DuplicateKey {
                    column: coll.config.schema.primary_key.map_or_else(
                        || "<none>".to_string(),
                        |k| Self::key_name(&coll.config.schema, k),
                    ),
                    value: "<taken concurrently>".to_string(),
                })
            }
        }
    }

    /// Refuse a batch in which a primary key repeats, or is already held by a
    /// live row — whichever comes first in input order.
    fn check_batch_keys<'r>(
        &self,
        coll: &Collection,
        key: ColumnId,
        rows: impl Iterator<Item = &'r Row>,
    ) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for row in rows {
            if let Some((_, value)) = row.iter().find(|(id, _)| *id == key)
                && !seen.insert(render_key(value))
            {
                return Err(Error::DuplicateKey {
                    column: Self::key_name(&coll.config.schema, key),
                    value: render_key(value),
                });
            }
            self.check_duplicate_key(coll, key, row)?;
        }
        Ok(())
    }

    /// Tombstone `ordinal` in `collection`. Blocks until durable. Idempotent.
    ///
    /// Ordinals are renumbered by compaction, so `ordinal` must come from the
    /// collection's current generation.
    pub fn delete(&self, collection: u32, ordinal: u64) -> Result<()> {
        let _gate = self.gate.enter();
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
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn mutate(
        &self,
        collection: u32,
        mut retire: RoaringBitmap,
        install: Vec<Install>,
    ) -> Result<Outcome> {
        let _gate = self.gate.enter();
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
    /// `vector` REPLACES each copy's embedding; `None` copies the old row's
    /// across untouched. The binder only allows a replacement when the
    /// statement names one row by ordinal, so in practice `Some` arrives with
    /// exactly one matched row. `set` is storage-level (`ColumnId`), already
    /// resolved and type-checked by the binder, and is applied in order so a
    /// column named twice takes its last value.
    ///
    /// Returns `Outcome::Conflicted` when the rows moved on between the read
    /// and the commit — the caller re-resolves its predicate and tries again.
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn update_rows(
        &self,
        collection: u32,
        matched: &RoaringBitmap,
        set: &[(ColumnId, Value)],
        vector: Option<&[f32]>,
    ) -> Result<Outcome> {
        let _gate = self.gate.enter();
        let coll = self.collection(collection)?;
        // A predicate matching no rows is a successful no-op, like a DELETE's:
        // nothing to copy, nothing to log, no fsync spent.
        if matched.is_empty() {
            return Ok(Outcome::Applied);
        }
        self.check_poison()?;

        // Checked before anything is read or logged, exactly as `insert` does.
        // The binder already enforces it, so reaching this means a caller built
        // the statement by hand — a record that cannot apply must still never
        // reach the WAL.
        let dim = coll.config.schema.vector().dim.get();
        if let Some(vector) = vector
            && vector.len() != dim
        {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }

        let tuples = self
            .tuple_reader(collection)
            .ok_or(Error::UnknownCollection { id: collection })?;
        let vectors = self
            .reader(collection)
            .ok_or(Error::UnknownCollection { id: collection })?;
        // Every scalar column: an UPDATE rewrites the WHOLE row, so the copy
        // must carry the columns it does not change as well as the ones it does.
        let scalars: Vec<ColumnId> = (0..coll.config.schema.columns.len() as ColumnId).collect();

        // A replacement embedding may reach EXACTLY ONE row. The binder has
        // already insisted the predicate be equality on the primary key, which
        // the applier keeps unique — so more than one match means those two
        // guarantees disagree, and silently handing one vector to several rows
        // is the specific mistake the key exists to prevent. Checked here
        // because only the engine sees how many rows actually matched.
        if vector.is_some() && matched.len() > 1 {
            return Err(Error::KeyMatchedManyRows {
                matched: matched.len(),
            });
        }

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
            // The statement's embedding replaces the old one when it carries
            // one; otherwise the row's own vector rides along unchanged.
            let vector = match vector {
                Some(replacement) => replacement.to_vec(),
                None => vectors
                    .vector_at(ordinal)
                    .ok_or(Error::SnapshotRowMissing {
                        collection,
                        ordinal: ordinal.0,
                    })?
                    .to_vec(),
            };

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
        self.allocated(&coll, start, start + copies.len() as u64);
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
        crate::cancel::under_current_or(
            || self.statement_watch(),
            || self.search_unwatched(collection, query, k),
        )
    }

    fn search_unwatched(&self, collection: u32, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
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
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
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
        check_collection_size(schema.vector().dim.get(), capacity)?;
        // The schema carries the vector (name + declaration ordinal + dim) and
        // every scalar, and is invalid-by-construction impossible via
        // `Schema::from_columns`. Re-check in case the caller hand-assembled one
        // and bypassed the constructor.
        schema.validate()?;
        // `id` used to be reserved here, because it named the ordinal
        // pseudo-column SEARCH returned. That pseudo-column is gone — identity
        // is a declared PRIMARY KEY now — so `id` is an ordinary name, and in
        // fact the conventional one for a key. Reserving it would forbid the
        // most natural schema a caller can write.
        let snapshot = catalog_snapshot(&self.catalog);
        if snapshot.values().any(|c| c.config.name == name) {
            return Err(Error::CollectionExists { name: name.into() });
        }
        // Assign the id on the logging side (like ordinals). Never reused: a
        // dropped collection's id stays retired, because the log may still
        // hold its records and they must never apply to a newcomer.
        let id = self
            .next_id
            .load(Ordering::Acquire)
            .max(snapshot.keys().max().map_or(0, |m| m + 1));

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

    /// Replace the runtime [`Limits`]. Applies to statements started after
    /// the call; a statement already running keeps the limits it started with.
    pub fn set_limits(&self, limits: Limits) {
        *self.limits.write().unwrap_or_else(|e| e.into_inner()) = limits;
    }

    /// The current runtime [`Limits`].
    pub fn limits(&self) -> Limits {
        *self.limits.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Stop every statement running on this database right now — searches,
    /// scans and streams — with [`Error::Interrupted`] at its next check
    /// (within a fraction of a millisecond for a search). Statements started
    /// after this call are unaffected. Safe from any thread. A write already
    /// logged completes; nothing is half-written.
    pub fn interrupt(&self) {
        self.interrupts.fetch_add(1, Ordering::AcqRel);
    }

    /// The watch a statement starting now runs under: this moment's
    /// interrupt generation, and its deadline under the current limits.
    pub(crate) fn statement_watch(&self) -> crate::cancel::Watch {
        let deadline = self
            .limits()
            .statement_timeout
            .map(|t| std::time::Instant::now() + t);
        crate::cancel::Watch::start(&self.interrupts, deadline)
    }

    /// Drop the collection named `name`: its rows, its stores and its files.
    /// Blocks until durable. The name is free again as soon as this returns;
    /// the id is never reused.
    ///
    /// Logged as one `Record::DropCollection`, so it survives a crash and
    /// replay never brings the collection back. Reads already running against
    /// it finish against what they started with; a write that loses the race
    /// is refused with [`Error::UnknownCollection`].
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        // Serialized with CREATE: a drop and a create of the same name must
        // not interleave their checks.
        let _ddl = self.ddl.lock().unwrap_or_else(|e| e.into_inner());
        let id = self.collection_id(name)?;
        self.check_poison()?;
        self.wal_handle()?
            .append(Record::DropCollection { id })
            .map_err(|e| self.wal_err(e))?;
        Ok(())
    }

    /// Write a consistent copy of the whole database into `dest`, which must
    /// not exist yet or be an empty directory. Restore by passing `dest` to
    /// [`Db::open`].
    ///
    /// The copy is taken on the commit thread, between records: it is the
    /// database as of one point in the log, holding every statement
    /// acknowledged before that point whole and none after it. Reads carry on
    /// throughout; writes wait for the copy, which is a full file copy of
    /// every live collection — plan for it on large databases.
    ///
    /// On failure, whatever was written into `dest` is removed again.
    pub fn backup(&self, dest: impl AsRef<Path>) -> Result<()> {
        let dest = dest.as_ref();
        let created = match std::fs::read_dir(dest) {
            Ok(mut entries) => {
                if entries.next().is_some() {
                    return Err(Error::Io(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!("backup destination {} is not empty", dest.display()),
                    )));
                }
                false
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                std::fs::create_dir_all(dest)?;
                true
            }
            Err(e) => return Err(e.into()),
        };
        self.check_poison()?;
        let res = self
            .wal_handle()
            .and_then(|wal| {
                wal.control(Box::new(compact::Backup {
                    dest: dest.to_path_buf(),
                }))
                .map_err(|e| self.wal_err(e))
            });
        if res.is_err() {
            // Leave the destination as we found it.
            if created {
                let _ = std::fs::remove_dir_all(dest);
            } else if let Ok(entries) = std::fs::read_dir(dest) {
                for entry in entries.flatten() {
                    let _ = std::fs::remove_dir_all(entry.path())
                        .or_else(|_| std::fs::remove_file(entry.path()));
                }
            }
        }
        res
    }

    /// The current liveness snapshot for `collection`, materializing it if no
    /// reader has needed this version yet. `None` for an unknown collection.
    ///
    /// Resolve ONCE and hold the returned `Arc` for the life of a query: a
    /// `LiveSet` is immutable, so the rows it admits never change underneath
    /// the caller. Calling this per row would defeat the point.
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn live_snapshot(&self, collection: u32) -> Option<Arc<meta::LiveSet>> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.live.resolve())
    }

    /// A cloneable read handle for `collection`, for issuing searches from other
    /// threads in parallel. Returns `None` for an unknown collection.
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn reader(&self, collection: u32) -> Option<Reader> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.reader.clone())
    }

    /// Read handle for `collection`'s metadata index (lookup_eq/lookup_range
    /// bitmaps). Primitive read-side plumbing — the real SEARCH..WHERE
    /// executor is a later phase. `None` for an unknown collection.
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn metadata_reader(&self, collection: u32) -> Option<meta::Reader> {
        catalog_snapshot(&self.catalog)
            .get(&collection)
            .map(|c| c.meta.clone())
    }

    /// Read handle for `collection`'s tuple store (RETURNING values by
    /// ordinal). `None` for an unknown collection.
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
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
    ///
    /// *Internal: hidden from the documented API — its signature speaks
    /// engine-internal types. Exempt from semver; reachable for this crate's
    /// tests. Use [`Db::execute`] instead.*
    #[doc(hidden)]
    pub fn scan_over(
        &self,
        collection: u32,
        ordinals: impl Iterator<Item = Ordinal> + Send + 'static,
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
            // The flat reader rides along so the row's embedding is reachable
            // without the `Db` — see `Cursor::vector`.
            coll.reader.clone(),
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
    /// live snapshot as `Catalog::schema_of` — so a
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
        self.stop_compactor();
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        // Only `close` (which consumes self) and `Drop` take the WAL, so it is
        // always here; if not, there is nothing left to shut down.
        let Some(wal) = self.wal.take() else {
            return Ok(());
        };
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

    /// The key column's declared name, for an error message. Falls back to the
    /// `ColumnId` only if the schema and the key disagree, which construction
    /// rules out.
    fn key_name(schema: &Schema, key: ColumnId) -> String {
        schema
            .columns
            .iter()
            .find(|c| c.id == key)
            .map_or_else(|| format!("#{key}"), |c| c.name.clone())
    }

    /// Refuse a row whose primary key a LIVE row already holds.
    ///
    /// One equality lookup against the metadata index, masked by the current
    /// liveness snapshot — so a key freed by a DELETE is available again,
    /// which is the behaviour a caller expects after removing a row.
    fn check_duplicate_key(&self, coll: &Collection, key: ColumnId, row: &Row) -> Result<()> {
        let Some(value) = row.iter().find(|(id, _)| *id == key).map(|(_, v)| v) else {
            // `validate_row` already insisted every column is present, so a
            // missing key column would be a contradiction rather than a
            // caller error.
            return Ok(());
        };
        let live = coll.live.resolve();
        if !coll.meta.lookup_eq(key, value, &live)?.is_empty() {
            return Err(Error::DuplicateKey {
                column: Self::key_name(&coll.config.schema, key),
                value: render_key(value),
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

    /// Compact `collection` now: rebuild its live rows densely and flip to
    /// them, reclaiming every dead ordinal. Blocks until done. `Ok(false)` when
    /// nothing was dead, so nothing happened.
    ///
    /// The background compactor does this on its own when a trigger fires;
    /// this is the same operation on demand. Concurrent reads are never
    /// blocked; writes pause only for the flip itself.
    pub fn compact_now(&self, collection: u32) -> Result<bool> {
        self.check_poison()?;
        let wal = self.wal_handle()?;
        self.compaction
            .compact(&wal, collection)
            .map_err(|e| match e {
                Error::Io(io) => self.wal_err(io),
                other => other,
            })
    }

    /// Enter the write gate for a whole statement (see `compact::WriteGate`).
    /// The `Db` write methods take it themselves; this is for a caller whose
    /// statement resolves ordinals BEFORE reaching them — `execute`.
    pub(crate) fn write_gate(&self) -> GateGuard<'_> {
        self.gate.enter()
    }

    /// The compaction epoch, for a reader that resolves several handles and
    /// must know they came from one generation (see `compact::Epoch`).
    pub(crate) fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    /// A write just allocated ordinals `[from, to)`: let the compactor know if
    /// that crossed its headroom line.
    fn allocated(&self, coll: &Collection, from: u64, to: u64) {
        if let Some(compactor) = &self.compactor {
            compactor.allocated(coll.config.id, coll.config.capacity, from, to);
        }
    }

    /// Abandon any build in progress and join the compactor. Before the
    /// flusher and the WAL, because a flip needs the commit thread.
    fn stop_compactor(&mut self) {
        self.compaction.stop.store(true, Ordering::Relaxed);
        if let Some(compactor) = self.compactor.take() {
            compactor.stop();
        }
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        // If the caller didn't `close()`, still tear down cleanly: stop the
        // flusher and join the WAL thread. We skip the final checkpoint here —
        // the WAL is durable, so the next open just replays a longer tail.
        self.stop_compactor();
        if let Some(flusher) = self.flusher.take() {
            flusher.stop();
        }
        if let Some(wal) = self.wal.take() {
            wal.shutdown();
        }
    }
}

/// A key value as it should read back to the person who typed it — `'alice'`,
/// not `Text("alice")`. Only for error messages.
fn render_key(value: &Value) -> String {
    match value {
        Value::Text(s) => format!("'{s}'"),
        Value::Int(n) => n.to_string(),
        // A FLOAT key is refused at declaration, so this is unreachable from a
        // declared key; rendered anyway rather than panicking in an error path.
        Value::Float(f) => f.to_string(),
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
        // The embedding can never be the key — `from_columns` refuses it.
        primary_key: false,
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
            // Read off the PERSISTED schema, so a reopened collection keys on
            // the column it was created with rather than on a name convention.
            primary_key: schema.primary_key == Some(def.id),
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

/// Fuzzing hook: decode `data` as a `catalog.snap` body that passed every
/// framing check. Must return for any input, never panic.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub fn fuzz_decode_catalog(data: &[u8]) -> bool {
    decode_catalog(&crate::metadata::frame_for_fuzz(CATALOG_MAGIC, CATALOG_VERSION, data)).is_ok()
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
            ..DbOptions::default()
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
        assert!(
            cursor.seek_first().unwrap(),
            "the scan cursor still sees the row"
        );
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

    /// A keyed collection: (k INT PRIMARY KEY) alongside the vector.
    fn cfg_keyed(id: u32, dim: usize, capacity: usize) -> CollectionConfig {
        CollectionConfig {
            id,
            name: format!("k{id}"),
            capacity,
            schema: Schema::from_columns(vec![
                ColumnSpec::Vector {
                    name: "vector".into(),
                    dim: NonZeroUsize::new(dim).unwrap(),
                },
                ColumnSpec::Key {
                    name: "k".into(),
                    ty: ColumnType::Int,
                },
            ])
            .unwrap(),
        }
    }

    /// Log a duplicate-key insert the way a LOST RACE does: straight to the
    /// WAL handle, past `Db::insert`'s pre-WAL lookup.
    ///
    /// That lookup is what makes the refusal unreachable through the public
    /// API single-threaded — which is the point of it — so a deterministic
    /// test of the apply-time verdict has to go around it, exactly as a second
    /// thread does by reading "absent" before the winner commits.
    fn log_duplicate(db: &Db, collection: u32, ordinal: u64, key: i64) -> Outcome {
        db.wal
            .as_ref()
            .expect("wal present")
            .handle()
            .append(Record::Insert {
                collection,
                ordinal,
                vector: vec![0.0, 1.0],
                metadata: vec![(0, Value::Int(key))],
            })
            .expect("a refusal is success, not an error")
            .outcome
    }

    /// A REFUSED record is durable and will never do anything, so the
    /// watermarks must advance past it exactly as they do past a conflicted
    /// one. Otherwise one duplicate key wedges truncation forever: the log can
    /// never shed the refused frame, and every reopen replays it again.
    #[test]
    fn a_refused_insert_does_not_wedge_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_keyed(0, 2, 64)], manual_opts()).unwrap();

        db.insert(0, &[1.0, 0.0], vec![(0, Value::Int(1))]).unwrap();
        // Same key, logged past the pre-WAL check: durable, and refused on the
        // commit thread. Deliberately the LAST record — a later applied one
        // would advance the high-water mark past it and hide the stall.
        assert_eq!(
            log_duplicate(&db, 0, 1, 1),
            Outcome::Rejected(RejectReason::DuplicateKey)
        );

        db.checkpoint().expect("checkpoint runs");
        let after = std::fs::metadata(dir.path().join("wal.log")).unwrap().len();
        assert_eq!(
            after, 8,
            // A fully truncated log is just its magic+version header.
            "truncation must get past the refused frame, not stall on it"
        );
        db.close().unwrap();
    }

    /// And the verdict survives a reopen: the refused frame is replayed, and
    /// must be refused AGAIN rather than installing a second row with the same
    /// key. Determinism is what makes a refusal safe to reach on the apply
    /// path at all.
    #[test]
    fn a_refusal_is_reached_again_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = Db::open(dir.path(), &[cfg_keyed(0, 2, 64)], manual_opts()).unwrap();
            db.insert(0, &[1.0, 0.0], vec![(0, Value::Int(1))]).unwrap();
            assert_eq!(
                log_duplicate(&db, 0, 1, 1),
                Outcome::Rejected(RejectReason::DuplicateKey)
            );
            // No checkpoint and no close: both frames stay in the log, so
            // reopening replays them.
            drop(db);
        }

        let db = Db::open(dir.path(), &[cfg_keyed(0, 2, 64)], manual_opts()).unwrap();
        assert!(
            !db.is_poisoned(),
            "replaying a refusal must not halt the db"
        );
        let live = db.live_snapshot(0).expect("collection open");
        assert_eq!(
            live.bitmap().len(),
            1,
            "replay must refuse the duplicate again, not install it"
        );
        db.close().unwrap();
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
        let values = match db
            .tuple_reader(0)
            .unwrap()
            .get(Ordinal(ordinal), &[0, 1])
            .unwrap()
        {
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
            .update_rows(0, &matched, &[(1, Value::Text("archived".into()))], None)
            .unwrap();
        assert_eq!(outcome, Outcome::Applied);

        let live: Vec<u32> = db.live_snapshot(0).unwrap().iter().collect();
        assert_eq!(
            live,
            vec![1, 3, 4],
            "originals retired, copies at fresh ordinals"
        );

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
        db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))], None)
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
            None,
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
            .update_rows(
                0,
                &RoaringBitmap::new(),
                &[(1, Value::Text("x".into()))],
                None,
            )
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
            db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))], None)
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

    /// Roadmap step 8, as it applies to UPDATE: the room for every copy is
    /// checked in ONE reservation, before the record is built. Half a statement
    /// is not a thing that can happen — a per-row allocator could pass its
    /// check, claim some ordinals, then fail, leaving the statement abandoned
    /// with capacity burned.
    #[test]
    fn an_update_that_would_exceed_capacity_is_refused_before_logging() {
        let dir = tempfile::tempdir().unwrap();
        // Capacity 8, five rows used: four copies need 5..9, one past the end.
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 8)], manual_opts()).unwrap();
        for n in 0..5i64 {
            db.insert(0, &[n as f32, 0.0], meta_row(n, "draft"))
                .unwrap();
        }

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).unwrap().len();
        let matched: RoaringBitmap = [0u32, 1, 2, 3].into_iter().collect();
        match db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))], None) {
            Err(Error::CapacityExceeded { capacity }) => assert_eq!(capacity, 8),
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }

        assert_eq!(
            std::fs::metadata(&wal).unwrap().len(),
            before,
            "a refused update must never reach the log"
        );
        assert_eq!(db.live_snapshot(0).unwrap().len(), 5, "no row changed");
        // The allocator did not move either: the next insert takes ordinal 5.
        assert_eq!(
            db.insert(0, &[9.0, 0.0], meta_row(9, "draft")).unwrap(),
            Ordinal(5)
        );
    }

    #[test]
    fn an_update_that_exactly_fills_capacity_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        // Capacity 9, five rows used, four copies land at 5..9 exactly.
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 9)], manual_opts()).unwrap();
        for n in 0..5i64 {
            db.insert(0, &[n as f32, 0.0], meta_row(n, "draft"))
                .unwrap();
        }

        let matched: RoaringBitmap = [0u32, 1, 2, 3].into_iter().collect();
        db.update_rows(0, &matched, &[(1, Value::Text("archived".into()))], None)
            .expect("the copies fit exactly");

        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![4, 5, 6, 7, 8]
        );
    }

    /// Full replacement: the new embedding comes from the statement instead of
    /// the old row, while every column the statement did not name still copies
    /// across.
    #[test]
    fn update_rows_takes_a_replacement_vector() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let matched: RoaringBitmap = [1u32].into_iter().collect();
        db.update_rows(
            0,
            &matched,
            &[(1, Value::Text("v2".into()))],
            Some(&[9.0, 9.0]),
        )
        .unwrap();

        let (values, vector) = row_at(&db, 3);
        assert_eq!(vector, vec![9.0, 9.0], "the statement's embedding won");
        assert_eq!(
            values,
            vec![Value::Int(2), Value::Text("v2".into())],
            "the unnamed column still copied across"
        );
        assert_eq!(
            db.live_snapshot(0).unwrap().iter().collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    /// The binder already checks this, so reaching it means a caller built the
    /// record by hand — the same belt-and-braces `insert` keeps.
    #[test]
    fn a_replacement_vector_is_dimension_checked_before_logging() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[cfg_meta(0, 2, 64)], manual_opts()).unwrap();
        seed_three(&db);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).unwrap().len();
        let matched: RoaringBitmap = [1u32].into_iter().collect();
        match db.update_rows(0, &matched, &[], Some(&[9.0, 9.0, 9.0])) {
            Err(Error::DimensionMismatch { expected, got }) => {
                assert_eq!((expected, got), (2, 3));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
        assert_eq!(std::fs::metadata(&wal).unwrap().len(), before);
        assert_eq!(db.live_snapshot(0).unwrap().len(), 3);
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
        let filter = meta
            .lookup_range(
                0,
                RangeOp::Lt,
                &Value::Int(3),
                &db.live_snapshot(0).unwrap(),
            )
            .unwrap();
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
        let filter = meta
            .lookup_eq(
                1,
                &Value::Text("alice".into()),
                &db.live_snapshot(0).unwrap(),
            )
            .unwrap();
        assert_eq!(filter.iter().collect::<Vec<u32>>(), vec![2]);
        // Retiring a row clears its live bit and touches nothing else — the
        // values are still there for anyone holding an older snapshot.
        assert!(matches!(
            tuples.get(Ordinal(0), &[0]).unwrap(),
            RowGet::Live(_)
        ));
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
                db.insert(
                    0,
                    &[i as f32, 1.0],
                    meta_row(i, if i % 2 == 0 { "even" } else { "odd" }),
                )
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
            assert!(matches!(
                tuples.get(Ordinal(ord), &[0]).unwrap(),
                RowGet::Live(_)
            ));
        }
        // Retired, so absent from `live` — but its values survive.
        assert!(matches!(
            tuples.get(Ordinal(3), &[0]).unwrap(),
            RowGet::Live(_)
        ));
        assert!(!meta.live().contains(3));

        // Spot-check lookups and values.
        let evens = meta
            .lookup_eq(
                1,
                &Value::Text("even".into()),
                &db.live_snapshot(0).unwrap(),
            )
            .unwrap();
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
            assert_eq!(
                db.search(0, &[0.0, 1.0], 64).unwrap().len(),
                3,
                "victim {victim}"
            );
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
                meta.lookup_eq(0, &Value::Int(2), &db.live_snapshot(0).unwrap())
                    .unwrap()
                    .iter()
                    .collect::<Vec<u32>>(),
                vec![1],
                "victim {victim}"
            );
            assert!(matches!(
                tuples.get(Ordinal(0), &[1]).unwrap(),
                RowGet::Live(_)
            ));
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
            meta.lookup_eq(0, &Value::Int(7), &db.live_snapshot(0).unwrap())
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
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

    /// A version-3 catalog — written before DROP existed, so with no
    /// `next_id` — still opens, and its `next_id` is one past the highest id
    /// it registers: nothing was ever dropped under it.
    #[test]
    fn a_version_3_catalog_still_opens_and_derives_next_id() {
        let registry = vec![cfg(0, 2, 8), cfg(4, 2, 8)];
        let mut v3 = Vec::new();
        v3.extend_from_slice(CATALOG_MAGIC);
        v3.extend_from_slice(&3u32.to_le_bytes());
        v3.extend_from_slice(&7u64.to_le_bytes());
        let body = bincode::serialize(&registry).unwrap();
        v3.extend_from_slice(&(body.len() as u32).to_le_bytes());
        v3.extend_from_slice(&body);
        let crc = crc32(&v3);
        v3.extend_from_slice(&crc.to_le_bytes());

        let loaded = decode_catalog(&v3).expect("v3 decodes");
        assert_eq!(loaded.registry, registry);
        assert_eq!(loaded.last_lsn, 7);
        assert_eq!(loaded.next_id, 5);

        // And a database whose catalog is v3 opens and assigns the next id.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(CATALOG_FILE), &v3).unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();
        let schema = cfg(9, 2, 8).schema;
        assert_eq!(db.create_collection("fresh", 8, schema).unwrap(), 5);
    }

    /// The catalog remembers ids it issued even once nothing registers them.
    #[test]
    fn next_id_round_trips_through_the_catalog() {
        let bytes = encode_catalog(&[cfg(1, 2, 8)], 3, 42).unwrap();
        let loaded = decode_catalog(&bytes).unwrap();
        assert_eq!(loaded.next_id, 42);
        // Never below what the registry proves.
        let bytes = encode_catalog(&[cfg(50, 2, 8)], 3, 42).unwrap();
        assert_eq!(decode_catalog(&bytes).unwrap().next_id, 51);
        // An unknown future version is refused, not guessed at.
        let mut future = encode_catalog(&[], 0, 0).unwrap();
        future[4..8].copy_from_slice(&5u32.to_le_bytes());
        let n = future.len();
        let crc = crc32(&future[..n - 4]);
        future[n - 4..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(decode_catalog(&future), Err(Error::CorruptSnapshot(_))));
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
            assert_eq!(
                db.insert(0, &[1.0, 0.0], meta_row(1, "a")).unwrap(),
                Ordinal(0)
            );
            assert_eq!(
                db.insert(1, &[1.0, 0.0, 0.0, 0.0], meta_row(10, "b"))
                    .unwrap(),
                Ordinal(0)
            );
            assert_eq!(
                db.insert(0, &[2.0, 0.0], meta_row(2, "a")).unwrap(),
                Ordinal(1)
            );
            assert_eq!(
                db.insert(1, &[2.0, 0.0, 0.0, 0.0], meta_row(20, "b"))
                    .unwrap(),
                Ordinal(1)
            );
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
            meta1
                .lookup_eq(0, &Value::Int(20), &db.live_snapshot(0).unwrap())
                .unwrap()
                .iter()
                .collect::<Vec<u32>>(),
            vec![1]
        );
        assert_eq!(
            db.tuple_reader(1).unwrap().get(Ordinal(0), &[1]).unwrap(),
            RowGet::Live(vec![Value::Text("b".into())])
        );

        // Allocators resumed independently per collection.
        assert_eq!(
            db.insert(0, &[3.0, 0.0], meta_row(3, "a")).unwrap(),
            Ordinal(2)
        );
        assert_eq!(
            db.insert(1, &[3.0, 0.0, 0.0, 0.0], meta_row(30, "b"))
                .unwrap(),
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

        db.insert(id, &[1.0, 0.0, 0.0], vec![(0, Value::Int(7))])
            .unwrap();
        db.insert(id, &[0.0, 1.0, 0.0], vec![(0, Value::Int(8))])
            .unwrap();

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
        assert!(
            !dir.path().join("catalog.snap").exists(),
            "nothing applied yet"
        );
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

    /// `id` used to be a reserved column name, because it collided with the
    /// ordinal pseudo-column SEARCH returned. That pseudo-column is gone, so
    /// `id` is an ordinary name — and the conventional one for a primary key,
    /// which means reserving it would have forbidden the most natural schema a
    /// caller can write.
    #[test]
    fn a_column_named_id_is_an_ordinary_key() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], manual_opts()).unwrap();

        let schema = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(2).unwrap(),
            },
            ColumnSpec::Key {
                name: "id".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap();
        assert_eq!(schema.primary_key, Some(0));
        db.create_collection("things", 8, schema)
            .expect("`id` is a legal column name");
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

        assert!(
            db.collections().is_empty(),
            "nothing registered by any failed attempt"
        );

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
            (
                s.columns[0].id,
                s.columns[0].name.as_str(),
                s.columns[0].ordinal.get()
            ),
            (0, "author", 0)
        );
        assert_eq!(
            (
                s.columns[1].id,
                s.columns[1].name.as_str(),
                s.columns[1].ordinal.get()
            ),
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
                    sel.projection
                        .iter()
                        .map(|c| c.name.clone())
                        .collect::<Vec<_>>(),
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
        assert_eq!(
            set.bitmap(),
            &meta.live(),
            "snapshot must match the authority"
        );
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
        assert!(
            !Arc::ptr_eq(&before, &after),
            "the delete moved the version"
        );

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
