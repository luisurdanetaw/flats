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

use crate::error::{Error, Result};
use crate::index::index::{FlatIndex, Ordinal, Reader, SearchResult, Writer};
use crate::metadata::common::{Row, Schema};
use crate::metadata::index::MetadataIndex;
use crate::metadata::tuples::TupleStore;
use crate::metadata::{crc32, index as meta, tuples, write_snapshot_atomic};
use crate::wal::wal::{Apply, Lsn, Record, Wal, WalHandle};

/// Re-exported from `metadata::common` (it moved there in Phase 6 so
/// `Record::CreateCollection` can carry it — the WAL may only depend on plain
/// data types). Registered on first mention (a `Db::open` config or a
/// `create_collection` call) and persisted in `catalog.snap` from then on.
pub use crate::metadata::common::CollectionConfig;

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
}

impl Collection {
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
    Ok((
        CollectionWriters {
            flat: flat_w,
            meta: meta_w,
            tuple: tuple_w,
        },
        Collection {
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
    /// restart replays it. (Inserts don't need this: a failed insert stalls
    /// its own collection's store watermarks, which are already in the min.
    /// A failed create has no store watermark to stall — this is its stand-in.)
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
    fn apply(&mut self, lsn: Lsn, record: &Record) -> io::Result<()> {
        match record {
            Record::Insert {
                collection,
                ordinal,
                vector,
                metadata,
            } => {
                let w = self.writers(*collection)?;
                let ord32 = ordinal32(*ordinal)?;
                w.flat.write_at(*ordinal, vector).map_err(to_io)?;
                w.meta.insert_row(ord32, metadata).map_err(to_io)?;
                w.tuple.write_row(ord32, metadata).map_err(to_io)?;
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
                w.flat.delete(*ordinal).map_err(to_io)?;
                w.meta.remove_row(ord32).map_err(to_io)?;
                w.tuple.delete_row(ord32).map_err(to_io)?;
                w.flat.advance_applied_lsn(lsn.0);
                w.meta.advance_applied_lsn(lsn);
                w.tuple.advance_applied_lsn(lsn);
            }
            Record::CreateCollection { config } => {
                self.apply_create(lsn, config)?;
            }
        }
        // The catalog watermark advances only on success — a `?` above skips
        // this, so a failed record can never be claimed as reflected.
        self.last_applied = self.last_applied.max(lsn.0);
        Ok(())
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
        let ordinal = coll.alloc_ordinal()?;
        match self.wal_handle()?.append(Record::Insert {
            collection,
            ordinal,
            vector: vector.to_vec(),
            metadata: row,
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
        self.wal_handle()?.append(Record::CreateCollection { config })?;
        Ok(id)
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

    fn collection(&self, id: u32) -> Result<Arc<Collection>> {
        catalog_snapshot(&self.catalog)
            .get(&id)
            .cloned()
            .ok_or(Error::UnknownCollection { id })
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
fn bind_schema(schema: &Schema) -> crate::sql::bind::Schema {
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
        let filter = meta.lookup_range(0, RangeOp::Lt, &Value::Int(3)).unwrap();
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
        let filter = meta.lookup_eq(1, &Value::Text("alice".into())).unwrap();
        assert_eq!(filter.iter().collect::<Vec<u32>>(), vec![2]);
        assert_eq!(tuples.get(Ordinal(0), &[0]).unwrap(), RowGet::Deleted);
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
        assert_eq!(tuples.get(Ordinal(3), &[0]).unwrap(), RowGet::Deleted);

        // Spot-check lookups and values.
        let evens = meta.lookup_eq(1, &Value::Text("even".into())).unwrap();
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
                meta.lookup_eq(0, &Value::Int(2)).unwrap().iter().collect::<Vec<u32>>(),
                vec![1],
                "victim {victim}"
            );
            assert_eq!(tuples.get(Ordinal(0), &[1]).unwrap(), RowGet::Deleted);
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
            meta.lookup_eq(0, &Value::Int(7)).unwrap().iter().collect::<Vec<u32>>(),
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
        assert_eq!(
            db.tuple_reader(0).unwrap().get(Ordinal(0), &[0]).unwrap(),
            RowGet::Deleted
        );

        // Collection 1: untouched by collection 0's delete; both rows live.
        assert_eq!(db.search(1, &[1.0, 0.0, 0.0, 0.0], 10).unwrap().len(), 2);
        let meta1 = db.metadata_reader(1).unwrap();
        assert_eq!(meta1.live_count(), 2);
        assert_eq!(
            meta1.lookup_eq(0, &Value::Int(20)).unwrap().iter().collect::<Vec<u32>>(),
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
        fn apply(&mut self, _lsn: Lsn, _record: &Record) -> io::Result<()> {
            Ok(())
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
                .lookup_eq(0, &Value::Int(8))
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
}
