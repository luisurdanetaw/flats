//! Phase 4a — the metadata index: standalone, primitive (non-AST) API.
//!
//! Maps (column, value) → RoaringBitmap of ordinals. No WAL awareness, no
//! predicates, no query syntax — the executor composes those later out of
//! `lookup_eq` / `lookup_range` / bitmap AND/OR.
//!
//! # Storage shape (in memory)
//!
//! ```text
//! MetadataInner
//! ├── schema                                (immutable after create)
//! ├── columns: Vec<ColumnStore>             (parallel to schema.columns;
//! │   │                                      index == ColumnId)
//! │   ├── Int   → BTreeMap<i64,      RoaringBitmap>   ── sorted ⇒ ranges
//! │   ├── Float → BTreeMap<FloatKey, RoaringBitmap>   ── NaN kept out at insert
//! │   └── Text  → dict: HashMap<String, u32>          ── string → dense token id
//! │             + postings: Vec<RoaringBitmap>        ── token id → ordinals
//! ├── live: RoaringBitmap                   (ordinal alive ⇔ bit set)
//! └── applied_lsn: Lsn                      (in-memory watermark, see 4c)
//! ```
//!
//! Why BTreeMap and not HashMap for Int/Float: `lookup_range` walks a key
//! range in order — that IS a BTreeMap. Text gets the dictionary treatment
//! because ranges on TEXT are a non-goal (they return empty), so all we ever
//! need is exact-match, and interning keeps repeated strings cheap.
//!
//! # Deletes are LAZY — the `live` bitmap is the single source of truth
//!
//! `remove_row` only clears the ordinal's bit in `live`. Postings are NOT
//! scrubbed (that would need value lookups or a full scan). Instead, every
//! lookup ANDs its result with `live` before returning — that is the
//! correctness rule from the spec ("tombstoned ordinals must never leak"),
//! and it's what makes lazy deletion sound. Postings grow monotonically;
//! compaction is a future concern, deliberately not Phase 4a.
//!
//! # Concurrency: SWMR-shaped, Mutex-implemented
//!
//! `Arc<Mutex<MetadataInner>>` behind a `Writer` (not Clone — the type-level
//! single-writer invariant, same posture as FlatIndex) and a cloneable
//! `Reader`. A Mutex means readers serialize against the writer for now;
//! that's accepted (spec: real SWMR deferred). Keep every critical section
//! tiny and do NO file IO under the lock (see checkpoint).
//!
//! # Persistence: whole-snapshot, serialize-and-rename
//!
//! `checkpoint(last_lsn)` serializes the entire inner state to
//! `<dir>/metadata.snap.tmp`, fsyncs, renames over `metadata.snap`, fsyncs
//! the dir. Atomic-rename means a crash at ANY point leaves either the old
//! complete snapshot or the new complete snapshot — never a torn one. The
//! WAL (Phase 4c) replays everything past the snapshot's `last_lsn`.
//!
//! ## `metadata.snap` layout (all integers LE, per CLAUDE.md §5)
//!
//! ```text
//! offset  size   field
//! 0       4      magic          b"MET0"
//! 4       4      version        u32 = 1
//! 8       8      last_lsn       u64
//! 16      4      schema_len     u32
//! 20      var    schema         bincode(Schema)
//! ..      var    columns        one block per column, in ColumnId order:
//!                  Int:   entry_count u64, then per entry:
//!                           key i64, roaring bytes (self-delimiting —
//!                           RoaringBitmap::serialize_into/deserialize_from)
//!                  Float: entry_count u64, then per entry:
//!                           key f64 (to_le_bytes of the bits), roaring bytes
//!                  Text:  token_count u64, then per token id 0..n:
//!                           str_len u32, utf8 bytes, roaring bytes
//!                         (dict is rebuilt from this on load — token id is
//!                          the position, string is the key)
//! ..      var    live           roaring bytes
//! end-4   4      crc32          over ALL preceding bytes
//! ```
//!
//! No per-column type tags needed: the schema (already decoded) dictates each
//! block's shape. The trailing CRC makes "torn/garbage file" detectable;
//! detection policy is in `open`.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use roaring::RoaringBitmap;

use crate::error::{Error, Result};
use crate::metadata::common::{
    ColumnId, ColumnType, Lsn, Ordinal, RangeOp, Row, Schema, Value,
};

const MAGIC: &[u8; 4] = b"MET0";
const VERSION: u32 = 1;
const SNAP_FILE: &str = "metadata.snap";
const TMP_FILE: &str = "metadata.snap.tmp";

use crate::metadata::crc32;

// ---------------------------------------------------------------------------
// FloatKey — f64 as a BTreeMap key
// ---------------------------------------------------------------------------

/// f64 is not `Ord` (NaN breaks totality), so it can't key a BTreeMap
/// directly. This newtype supplies a total order via `f64::total_cmp`.
/// NaN never gets in (rejected at `Schema::validate_row`), so total_cmp's
/// NaN placement never actually matters — but using it everywhere keeps
/// Eq/Ord honest and the impls trivial.
///
/// total_cmp says -0.0 < +0.0, so they would be DIFFERENT keys; `new`
/// normalizes -0.0 → +0.0 so `WHERE x = 0.0` hits rows inserted with -0.0.
#[derive(Debug, Clone, Copy)]
pub struct FloatKey(f64);

impl FloatKey {
    /// Callers guarantee non-NaN (validate_row ran first, or the lookup path
    /// rejected NaN explicitly); the debug_assert documents that precondition.
    fn new(v: f64) -> FloatKey {
        debug_assert!(!v.is_nan(), "NaN must be rejected before key creation");
        // -0.0 == 0.0 under IEEE comparison, so this folds exactly the two
        // zeroes onto one key and touches nothing else.
        FloatKey(if v == 0.0 { 0.0 } else { v })
    }
}

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        // total_cmp, NOT f64::eq — one ordering everywhere, Eq stays honest.
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal
    }
}
impl Eq for FloatKey {}

impl PartialOrd for FloatKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloatKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Column stores
// ---------------------------------------------------------------------------

/// Per-column posting storage. The variant is fixed by the schema at create
/// time and never changes.
enum ColumnStore {
    Int(BTreeMap<i64, RoaringBitmap>),
    Float(BTreeMap<FloatKey, RoaringBitmap>),
    Text {
        /// string → token id. Token ids are dense: 0..postings.len().
        dict: HashMap<String, u32>,
        /// token id → ordinals. Indexed by token id.
        postings: Vec<RoaringBitmap>,
    },
}

/// Map a RangeOp to BTreeMap range bounds over one side, unbounded on the
/// other: Lt/Le bound above, Gt/Ge bound below.
fn range_bounds<K>(op: RangeOp, k: K) -> (Bound<K>, Bound<K>) {
    match op {
        RangeOp::Lt => (Bound::Unbounded, Bound::Excluded(k)),
        RangeOp::Le => (Bound::Unbounded, Bound::Included(k)),
        RangeOp::Gt => (Bound::Excluded(k), Bound::Unbounded),
        RangeOp::Ge => (Bound::Included(k), Bound::Unbounded),
    }
}

impl ColumnStore {
    fn empty(ty: ColumnType) -> ColumnStore {
        match ty {
            ColumnType::Int => ColumnStore::Int(BTreeMap::new()),
            ColumnType::Float => ColumnStore::Float(BTreeMap::new()),
            ColumnType::Text => ColumnStore::Text {
                dict: HashMap::new(),
                postings: Vec::new(),
            },
        }
    }

    fn column_type(&self) -> ColumnType {
        match self {
            ColumnStore::Int(_) => ColumnType::Int,
            ColumnStore::Float(_) => ColumnType::Float,
            ColumnStore::Text { .. } => ColumnType::Text,
        }
    }

    /// Add `ordinal` to the posting list for `value`.
    ///
    /// A type mismatch is unreachable when validate_row ran, but comes back
    /// as Err anyway (no panics in lib code). RoaringBitmap::insert is a
    /// set-insert — re-adding the same ordinal is a no-op, which is what
    /// makes replayed insert_rows idempotent for free.
    fn insert(&mut self, column: ColumnId, ordinal: Ordinal, value: &Value) -> Result<()> {
        match (self, value) {
            (ColumnStore::Int(map), Value::Int(k)) => {
                map.entry(*k).or_default().insert(ordinal.0);
                Ok(())
            }
            (ColumnStore::Float(map), Value::Float(f)) => {
                if f.is_nan() {
                    return Err(Error::NaNRejected { column });
                }
                map.entry(FloatKey::new(*f)).or_default().insert(ordinal.0);
                Ok(())
            }
            (ColumnStore::Text { dict, postings }, Value::Text(s)) => {
                let token = match dict.get(s) {
                    Some(&t) => t,
                    None => {
                        let t = postings.len() as u32;
                        postings.push(RoaringBitmap::new());
                        dict.insert(s.clone(), t);
                        t
                    }
                };
                postings[token as usize].insert(ordinal.0);
                Ok(())
            }
            (store, value) => Err(Error::TypeMismatch {
                column,
                expected: store.column_type(),
                got: value.column_type(),
            }),
        }
    }

    /// Bitmap of ordinals whose value == `value` (NOT yet masked by live —
    /// the caller does that once, centrally).
    ///
    /// (Clone is the simple-and-correct baseline. If profiles ever show these
    /// clones, the fix is Arc<RoaringBitmap> postings + copy-on-write — not
    /// Phase 4a's problem.)
    fn lookup_eq(&self, column: ColumnId, value: &Value) -> Result<RoaringBitmap> {
        match (self, value) {
            (ColumnStore::Int(map), Value::Int(k)) => {
                Ok(map.get(k).cloned().unwrap_or_default())
            }
            (ColumnStore::Float(map), Value::Float(f)) => {
                // A NaN probe compares equal to nothing; an error beats a
                // silently-empty result for what is always a caller bug.
                if f.is_nan() {
                    return Err(Error::NaNRejected { column });
                }
                Ok(map.get(&FloatKey::new(*f)).cloned().unwrap_or_default())
            }
            (ColumnStore::Text { dict, postings }, Value::Text(s)) => Ok(dict
                .get(s)
                .map(|&t| postings[t as usize].clone())
                .unwrap_or_default()),
            (store, value) => Err(Error::TypeMismatch {
                column,
                expected: store.column_type(),
                got: value.column_type(),
            }),
        }
    }

    /// Union of posting lists over a key range (NOT yet masked by live).
    /// TEXT yields empty — defense-in-depth per spec: the parser should
    /// reject range-on-TEXT, but this layer must not rely on that.
    fn lookup_range(&self, column: ColumnId, op: RangeOp, value: &Value) -> Result<RoaringBitmap> {
        match self {
            ColumnStore::Text { .. } => Ok(RoaringBitmap::new()),
            ColumnStore::Int(map) => {
                let k = match value {
                    Value::Int(k) => *k,
                    other => {
                        return Err(Error::TypeMismatch {
                            column,
                            expected: ColumnType::Int,
                            got: other.column_type(),
                        });
                    }
                };
                Ok(map
                    .range(range_bounds(op, k))
                    .fold(RoaringBitmap::new(), |acc, (_, bm)| acc | bm))
            }
            ColumnStore::Float(map) => {
                let f = match value {
                    Value::Float(f) => *f,
                    other => {
                        return Err(Error::TypeMismatch {
                            column,
                            expected: ColumnType::Float,
                            got: other.column_type(),
                        });
                    }
                };
                // Comparisons with NaN are all-false; an error beats a
                // silently-empty result for a caller bug.
                if f.is_nan() {
                    return Err(Error::NaNRejected { column });
                }
                Ok(map
                    .range(range_bounds(op, FloatKey::new(f)))
                    .fold(RoaringBitmap::new(), |acc, (_, bm)| acc | bm))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inner state + handles
// ---------------------------------------------------------------------------

struct MetadataInner {
    schema: Schema,
    /// Parallel to schema.columns; index IS the ColumnId.
    columns: Vec<ColumnStore>,
    /// THE liveness truth. Insert sets the bit, remove clears it, every
    /// lookup masks with it.
    live: RoaringBitmap,
    /// Highest LSN whose effects are in this in-memory state. Advanced by the
    /// applier (Phase 4c) after each apply; snapshotted by checkpoint.
    applied_lsn: Lsn,
}

impl MetadataInner {
    fn empty(schema: Schema) -> MetadataInner {
        let columns = schema.columns.iter().map(|c| ColumnStore::empty(c.ty)).collect();
        MetadataInner {
            schema,
            columns,
            live: RoaringBitmap::new(),
            applied_lsn: Lsn(0),
        }
    }
}

/// The single mutator. NOT Clone, NOT Sync-shared — exactly one exists per
/// collection and it lives in the WAL applier (Phase 4c). Pinned in tests:
/// `static_assertions::assert_not_impl_any!(Writer: Clone)`.
pub struct Writer {
    inner: Arc<Mutex<MetadataInner>>,
    dir: PathBuf,
}

/// Cheap cloneable read handle. Every method locks, computes, unlocks —
/// nothing borrowed from the inner state ever escapes the guard.
#[derive(Clone)]
pub struct Reader {
    inner: Arc<Mutex<MetadataInner>>,
}

/// Namespace for the constructors (mirrors `FlatIndex::create/open`).
pub struct MetadataIndex;

impl MetadataIndex {
    /// Create a fresh index in `dir` and persist an initial empty snapshot.
    ///
    /// Writing the empty snapshot immediately (rather than lazily on first
    /// checkpoint) means `open` never has a "file missing but collection
    /// exists" state to reason about, and Phase 4c's CreateCollection replay
    /// stays idempotent.
    pub fn create(dir: &Path, schema: Schema) -> Result<(Writer, Reader)> {
        let inner = Arc::new(Mutex::new(MetadataInner::empty(schema)));
        let mut writer = Writer {
            inner: inner.clone(),
            dir: dir.to_path_buf(),
        };
        let reader = Reader { inner };
        writer.checkpoint(Lsn(0))?;
        Ok((writer, reader))
    }

    /// Open an existing index. Returns the snapshot's `last_lsn` — Phase 4c
    /// feeds the min across all three subsystems to WAL recovery as
    /// `skip_through`.
    ///
    /// DESIGN DEVIATION FROM THE SPEC (deliberate — keep or revisit): the
    /// spec says `open(dir)`, but the corrupt-snapshot policy is "start empty
    /// at lsn 0 and let WAL replay rebuild" — and you cannot start empty
    /// without knowing the schema. The schema's source of truth is the
    /// caller (collection config / catalog), same as FlatIndex's dim. So
    /// `open` takes it, and uses the snapshot's embedded schema only as a
    /// cross-check.
    ///
    /// A missing or corrupt (magic/version/CRC/decode) snapshot is NOT an
    /// error: fall back to empty at Lsn(0) — the WAL holds the truth and
    /// replay rebuilds everything. A schema mismatch IS an error: it means
    /// the caller opened the wrong dir or the catalog is lying; rebuilding
    /// from WAL won't fix that.
    pub fn open(dir: &Path, schema: Schema) -> Result<(Writer, Reader, Lsn)> {
        let snap = dir.join(SNAP_FILE);
        let decoded = match std::fs::read(&snap) {
            Ok(bytes) => match decode_snapshot(&bytes) {
                Ok(v) => Some(v),
                Err(Error::CorruptSnapshot(why)) => {
                    // No logger yet — but losing this signal silently makes
                    // "my checkpoint never sticks" undebuggable.
                    eprintln!(
                        "flats: {} corrupt ({why}); starting empty, WAL replay rebuilds",
                        snap.display()
                    );
                    None
                }
                Err(e) => return Err(e),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let (inner, last_lsn) = match decoded {
            Some((inner, lsn)) => {
                if inner.schema != schema {
                    return Err(Error::SchemaMismatch);
                }
                (inner, lsn)
            }
            None => (MetadataInner::empty(schema), Lsn(0)),
        };
        let inner = Arc::new(Mutex::new(inner));
        Ok((
            Writer {
                inner: inner.clone(),
                dir: dir.to_path_buf(),
            },
            Reader { inner },
            last_lsn,
        ))
    }

    /// `open` if a snapshot exists, `create` (writing the initial empty
    /// snapshot) otherwise. This is what makes CREATE COLLECTION replay
    /// idempotent at the file level: a half-materialized collection from a
    /// crashed create is absorbed, not tripped over.
    pub fn open_or_create(dir: &Path, schema: Schema) -> Result<(Writer, Reader, Lsn)> {
        if dir.join(SNAP_FILE).exists() {
            Self::open(dir, schema)
        } else {
            let (w, r) = Self::create(dir, schema)?;
            Ok((w, r, Lsn(0)))
        }
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

impl Writer {
    /// Insert a full row at `ordinal`. Idempotent: replaying the same
    /// (ordinal, row) is harmless — bitmap inserts are set-inserts. (WAL
    /// replay only ever re-delivers IDENTICAL rows; there is no "update".)
    ///
    /// Validation runs BEFORE any mutation (types, NaN, completeness — see
    /// common.rs), so a bad row can never leave a half-inserted ordinal.
    ///
    /// NOTE re-inserting a REMOVED ordinal resurrects it (live bit set
    /// again). Fine in v1: ordinals are allocated once, never reused, and
    /// WAL order (Insert then Delete) is preserved on replay.
    pub fn insert_row(&mut self, ordinal: Ordinal, row: &Row) -> Result<()> {
        let mut inner = self.lock();
        inner.schema.validate_row(row)?;
        for (col, value) in row {
            inner.columns[*col as usize].insert(*col, ordinal, value)?;
        }
        inner.live.insert(ordinal.0);
        Ok(())
    }

    /// Tombstone `ordinal`. Idempotent (clearing a cleared bit is a no-op).
    ///
    /// Postings stay (lazy delete, see module doc). Removing an ordinal that
    /// was never inserted is also a silent no-op — replay may deliver Deletes
    /// for ordinals whose Insert was folded into the snapshot, and the bit is
    /// simply already correct.
    pub fn remove_row(&mut self, ordinal: Ordinal) -> Result<()> {
        self.lock().live.remove(ordinal.0);
        Ok(())
    }

    /// Record that this store's in-memory state now reflects `lsn`.
    /// Called by the applier after each apply (Phase 4c). Monotonic: keeps
    /// max(current, lsn).
    pub fn advance_applied_lsn(&mut self, lsn: Lsn) {
        let mut inner = self.lock();
        if lsn > inner.applied_lsn {
            inner.applied_lsn = lsn;
        }
    }

    /// The watermark last passed to `advance_applied_lsn`. Phase 4c's
    /// checkpoint reads this to know what `last_lsn` to persist.
    pub fn applied_lsn(&self) -> Lsn {
        self.lock().applied_lsn
    }

    /// Persist the whole state as `metadata.snap`, claiming durability
    /// through `last_lsn`. Crash-safe at every step (see module doc).
    ///
    /// Serialization happens under the lock (consistent view); the IO runs
    /// outside it (readers must not stall on disk).
    ///
    /// `last_lsn` contract (Phase 4c): must be ≤ the applied watermark, and
    /// the WAL may only be truncated up to the min of all three stores'
    /// persisted last_lsns, strictly AFTER all three checkpoints return.
    pub fn checkpoint(&mut self, last_lsn: Lsn) -> Result<()> {
        let bytes = {
            let inner = self.lock();
            Self::encode_snapshot(&inner, last_lsn)
        };
        crate::metadata::write_snapshot_atomic(&self.dir, TMP_FILE, SNAP_FILE, &bytes)
    }

    /// See the layout doc at the top of the module. Explicit to_le_bytes on
    /// every integer — never host order.
    fn encode_snapshot(inner: &MetadataInner, last_lsn: Lsn) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&last_lsn.0.to_le_bytes());

        // Serializing plain data into memory has no fallible step (no IO, no
        // size cap) — same reasoning as Record::encode in wal.rs.
        let schema_bytes =
            bincode::serialize(&inner.schema).expect("Schema serialization into memory cannot fail");
        buf.extend_from_slice(&(schema_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&schema_bytes);

        for store in &inner.columns {
            match store {
                ColumnStore::Int(map) => {
                    buf.extend_from_slice(&(map.len() as u64).to_le_bytes());
                    for (k, bm) in map {
                        buf.extend_from_slice(&k.to_le_bytes());
                        bm.serialize_into(&mut buf)
                            .expect("bitmap serialization into memory cannot fail");
                    }
                }
                ColumnStore::Float(map) => {
                    buf.extend_from_slice(&(map.len() as u64).to_le_bytes());
                    for (k, bm) in map {
                        buf.extend_from_slice(&k.0.to_le_bytes());
                        bm.serialize_into(&mut buf)
                            .expect("bitmap serialization into memory cannot fail");
                    }
                }
                ColumnStore::Text { dict, postings } => {
                    // Invert the dict: token id → string. Ids are dense
                    // 0..postings.len() by construction.
                    let mut names: Vec<&str> = vec![""; postings.len()];
                    for (s, &t) in dict {
                        names[t as usize] = s;
                    }
                    buf.extend_from_slice(&(postings.len() as u64).to_le_bytes());
                    for (name, bm) in names.iter().zip(postings) {
                        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
                        buf.extend_from_slice(name.as_bytes());
                        bm.serialize_into(&mut buf)
                            .expect("bitmap serialization into memory cannot fail");
                    }
                }
            }
        }

        inner
            .live
            .serialize_into(&mut buf)
            .expect("bitmap serialization into memory cannot fail");

        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn lock(&self) -> MutexGuard<'_, MetadataInner> {
        // Poisoning: only a panic while holding the lock poisons it, and
        // (validate-before-mutate) our critical sections don't have partial
        // states worth protecting. Recover the guard rather than unwrap.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

impl Reader {
    /// Ordinals where `col == value`, tombstones already masked out.
    ///
    /// The `& live` mask is THE correctness rule: tombstoned ordinals must
    /// never leak out of this module. Applied here, centrally, so no
    /// ColumnStore can forget it.
    pub fn lookup_eq(&self, col: ColumnId, value: &Value) -> Result<RoaringBitmap> {
        let inner = self.lock();
        let store = inner
            .columns
            .get(col as usize)
            .ok_or(Error::UnknownColumn { column: col })?;
        let bm = store.lookup_eq(col, value)?;
        Ok(bm & &inner.live)
    }

    /// Ordinals where `col <op> value`, tombstones masked. TEXT columns
    /// yield empty (defense-in-depth).
    pub fn lookup_range(&self, col: ColumnId, op: RangeOp, value: &Value) -> Result<RoaringBitmap> {
        let inner = self.lock();
        let store = inner
            .columns
            .get(col as usize)
            .ok_or(Error::UnknownColumn { column: col })?;
        let bm = store.lookup_range(col, op, value)?;
        Ok(bm & &inner.live)
    }

    /// Clone of the live bitmap. The executor (later) uses this as the
    /// starting candidate set for unfiltered scans.
    pub fn live(&self) -> RoaringBitmap {
        self.lock().live.clone()
    }

    pub fn live_count(&self) -> u64 {
        self.lock().live.len()
    }

    fn lock(&self) -> MutexGuard<'_, MetadataInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Snapshot decode (module-level: used by open, tested directly)
// ---------------------------------------------------------------------------

fn corrupt(why: impl Into<String>) -> Error {
    Error::CorruptSnapshot(why.into())
}

/// Take exactly `n` bytes off the front of the cursor.
fn take<'a>(cur: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if cur.len() < n {
        return Err(corrupt("truncated"));
    }
    let (head, rest) = cur.split_at(n);
    *cur = rest;
    Ok(head)
}

fn read_u32(cur: &mut &[u8]) -> Result<u32> {
    let b: [u8; 4] = take(cur, 4)?.try_into().map_err(|_| corrupt("truncated"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(cur: &mut &[u8]) -> Result<u64> {
    let b: [u8; 8] = take(cur, 8)?.try_into().map_err(|_| corrupt("truncated"))?;
    Ok(u64::from_le_bytes(b))
}

fn read_bitmap(cur: &mut &[u8]) -> Result<RoaringBitmap> {
    // deserialize_from on a &mut &[u8] Reader advances the cursor past the
    // bitmap's self-delimiting encoding.
    RoaringBitmap::deserialize_from(&mut *cur).map_err(|e| corrupt(format!("bitmap: {e}")))
}

/// Inverse of `Writer::encode_snapshot`.
///
/// The trailing CRC is verified FIRST — after that every read is over
/// verified bytes and decode errors can only mean version-logic bugs, not
/// torn files.
fn decode_snapshot(bytes: &[u8]) -> Result<(MetadataInner, Lsn)> {
    // magic + version + last_lsn + schema_len + trailing crc.
    const MIN_LEN: usize = 4 + 4 + 8 + 4 + 4;
    if bytes.len() < MIN_LEN {
        return Err(corrupt("truncated"));
    }
    let (body, crc_tail) = bytes.split_at(bytes.len() - 4);
    let stored_crc = u32::from_le_bytes(crc_tail.try_into().map_err(|_| corrupt("truncated"))?);
    if crc32(body) != stored_crc {
        return Err(corrupt("crc mismatch"));
    }

    let mut cur = body;
    if take(&mut cur, 4)? != MAGIC {
        return Err(corrupt("bad magic"));
    }
    let version = read_u32(&mut cur)?;
    if version != VERSION {
        return Err(corrupt(format!("unsupported version {version}")));
    }
    let last_lsn = Lsn(read_u64(&mut cur)?);

    let schema_len = read_u32(&mut cur)? as usize;
    let schema: Schema = bincode::deserialize(take(&mut cur, schema_len)?)
        .map_err(|e| corrupt(format!("schema: {e}")))?;

    let mut columns = Vec::with_capacity(schema.columns.len());
    for def in &schema.columns {
        match def.ty {
            ColumnType::Int => {
                let n = read_u64(&mut cur)?;
                let mut map = BTreeMap::new();
                for _ in 0..n {
                    let k = i64::from_le_bytes(
                        take(&mut cur, 8)?.try_into().map_err(|_| corrupt("truncated"))?,
                    );
                    map.insert(k, read_bitmap(&mut cur)?);
                }
                columns.push(ColumnStore::Int(map));
            }
            ColumnType::Float => {
                let n = read_u64(&mut cur)?;
                let mut map = BTreeMap::new();
                for _ in 0..n {
                    let f = f64::from_le_bytes(
                        take(&mut cur, 8)?.try_into().map_err(|_| corrupt("truncated"))?,
                    );
                    // Written keys are never NaN; a NaN here is corruption
                    // the CRC happened not to catch. Refuse rather than
                    // trip FloatKey's precondition.
                    if f.is_nan() {
                        return Err(corrupt("NaN float key"));
                    }
                    map.insert(FloatKey::new(f), read_bitmap(&mut cur)?);
                }
                columns.push(ColumnStore::Float(map));
            }
            ColumnType::Text => {
                let n = read_u64(&mut cur)?;
                let mut dict = HashMap::new();
                let mut postings = Vec::new();
                for token in 0..n {
                    let len = read_u32(&mut cur)? as usize;
                    let s = String::from_utf8(take(&mut cur, len)?.to_vec())
                        .map_err(|e| corrupt(format!("text key: {e}")))?;
                    dict.insert(s, token as u32);
                    postings.push(read_bitmap(&mut cur)?);
                }
                columns.push(ColumnStore::Text { dict, postings });
            }
        }
    }

    let live = read_bitmap(&mut cur)?;
    if !cur.is_empty() {
        return Err(corrupt("trailing bytes"));
    }

    Ok((
        MetadataInner {
            schema,
            columns,
            live,
            applied_lsn: last_lsn,
        },
        last_lsn,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::common::{ColumnSpec, ColumnType};
    use std::num::NonZeroUsize;

    /// Schema used across tests: (a INT, b FLOAT, c TEXT) alongside the vector.
    fn test_schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(1).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Scalar {
                name: "b".into(),
                ty: ColumnType::Float,
            },
            ColumnSpec::Scalar {
                name: "c".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap()
    }

    fn row(a: i64, b: f64, c: &str) -> Row {
        vec![
            (0, Value::Int(a)),
            (1, Value::Float(b)),
            (2, Value::Text(c.into())),
        ]
    }

    fn bm(ords: &[u32]) -> RoaringBitmap {
        ords.iter().copied().collect()
    }

    // Writer must stay the single mutator — type-level half of the invariant.
    static_assertions::assert_not_impl_any!(Writer: Clone);
    static_assertions::assert_impl_all!(Reader: Clone, Send);

    #[test]
    fn insert_lookup_round_trip_each_type() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();

        w.insert_row(Ordinal(0), &row(1, 1.5, "x")).unwrap();
        w.insert_row(Ordinal(1), &row(2, 2.5, "y")).unwrap();
        w.insert_row(Ordinal(2), &row(3, 3.5, "x")).unwrap(); // "x" repeats

        assert_eq!(r.lookup_eq(0, &Value::Int(2)).unwrap(), bm(&[1]));
        assert_eq!(r.lookup_eq(1, &Value::Float(3.5)).unwrap(), bm(&[2]));
        assert_eq!(r.lookup_eq(2, &Value::Text("x".into())).unwrap(), bm(&[0, 2]));

        // Never-inserted values → empty bitmap, not Err.
        assert_eq!(r.lookup_eq(0, &Value::Int(99)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_eq(1, &Value::Float(9.9)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_eq(2, &Value::Text("zzz".into())).unwrap(), bm(&[]));
    }

    #[test]
    fn range_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();

        w.insert_row(Ordinal(0), &row(1, 1.0, "p")).unwrap();
        w.insert_row(Ordinal(1), &row(2, 2.0, "q")).unwrap();
        w.insert_row(Ordinal(2), &row(3, 3.0, "r")).unwrap();

        // INT against 2.
        let v = Value::Int(2);
        assert_eq!(r.lookup_range(0, RangeOp::Lt, &v).unwrap(), bm(&[0]));
        assert_eq!(r.lookup_range(0, RangeOp::Le, &v).unwrap(), bm(&[0, 1]));
        assert_eq!(r.lookup_range(0, RangeOp::Gt, &v).unwrap(), bm(&[2]));
        assert_eq!(r.lookup_range(0, RangeOp::Ge, &v).unwrap(), bm(&[1, 2]));

        // FLOAT against 2.0.
        let v = Value::Float(2.0);
        assert_eq!(r.lookup_range(1, RangeOp::Lt, &v).unwrap(), bm(&[0]));
        assert_eq!(r.lookup_range(1, RangeOp::Le, &v).unwrap(), bm(&[0, 1]));
        assert_eq!(r.lookup_range(1, RangeOp::Gt, &v).unwrap(), bm(&[2]));
        assert_eq!(r.lookup_range(1, RangeOp::Ge, &v).unwrap(), bm(&[1, 2]));

        // Bound BETWEEN keys.
        let v = Value::Float(2.5);
        assert_eq!(r.lookup_range(1, RangeOp::Lt, &v).unwrap(), bm(&[0, 1]));
        assert_eq!(r.lookup_range(1, RangeOp::Gt, &v).unwrap(), bm(&[2]));

        // Bounds OUTSIDE the key range.
        assert_eq!(r.lookup_range(0, RangeOp::Lt, &Value::Int(0)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_range(0, RangeOp::Ge, &Value::Int(0)).unwrap(), bm(&[0, 1, 2]));
        assert_eq!(r.lookup_range(0, RangeOp::Gt, &Value::Int(100)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_range(0, RangeOp::Le, &Value::Int(100)).unwrap(), bm(&[0, 1, 2]));

        // -0.0 and +0.0 land on the same normalized key.
        let mut w2 = w;
        w2.insert_row(Ordinal(3), &row(0, -0.0, "z")).unwrap();
        assert_eq!(r.lookup_eq(1, &Value::Float(0.0)).unwrap(), bm(&[3]));
        assert_eq!(r.lookup_eq(1, &Value::Float(-0.0)).unwrap(), bm(&[3]));
        // And range bounds treat them identically: nothing is < -0.0 here.
        assert_eq!(r.lookup_range(1, RangeOp::Lt, &Value::Float(-0.0)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_range(1, RangeOp::Le, &Value::Float(-0.0)).unwrap(), bm(&[3]));
    }

    #[test]
    fn tombstones_never_leak() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();

        for ord in 0..3u32 {
            w.insert_row(Ordinal(ord), &row(ord as i64, ord as f64, "same"))
                .unwrap();
        }
        w.remove_row(Ordinal(1)).unwrap();

        assert_eq!(r.lookup_eq(2, &Value::Text("same".into())).unwrap(), bm(&[0, 2]));
        assert_eq!(r.lookup_eq(0, &Value::Int(1)).unwrap(), bm(&[]));
        assert_eq!(
            r.lookup_range(0, RangeOp::Ge, &Value::Int(0)).unwrap(),
            bm(&[0, 2])
        );
        assert_eq!(r.live(), bm(&[0, 2]));
        assert_eq!(r.live_count(), 2);
    }

    #[test]
    fn delete_and_insert_are_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();

        let the_row = row(7, 7.0, "seven");
        w.insert_row(Ordinal(0), &the_row).unwrap();
        // Insert twice with the identical row: set semantics, same state.
        w.insert_row(Ordinal(0), &the_row).unwrap();
        assert_eq!(r.lookup_eq(0, &Value::Int(7)).unwrap(), bm(&[0]));
        assert_eq!(r.live_count(), 1);

        // Remove twice → same state, no Err.
        w.remove_row(Ordinal(0)).unwrap();
        w.remove_row(Ordinal(0)).unwrap();
        assert_eq!(r.live_count(), 0);
        assert_eq!(r.lookup_eq(0, &Value::Int(7)).unwrap(), bm(&[]));

        // Remove a never-inserted ordinal → no Err.
        w.remove_row(Ordinal(999)).unwrap();
        assert_eq!(r.live_count(), 0);
    }

    #[test]
    fn nan_rejected_and_state_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();

        let bad = row(1, f64::NAN, "x");
        assert!(matches!(
            w.insert_row(Ordinal(0), &bad),
            Err(Error::NaNRejected { column: 1 })
        ));
        // Validate-before-mutate: nothing was half-inserted.
        assert_eq!(r.live_count(), 0);
        assert_eq!(r.lookup_eq(0, &Value::Int(1)).unwrap(), bm(&[]));
        assert_eq!(r.lookup_eq(2, &Value::Text("x".into())).unwrap(), bm(&[]));
    }

    #[test]
    fn unknown_column_and_type_mismatch_err() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();
        w.insert_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();

        assert!(matches!(
            r.lookup_eq(99, &Value::Int(1)),
            Err(Error::UnknownColumn { column: 99 })
        ));
        assert!(matches!(
            r.lookup_range(99, RangeOp::Lt, &Value::Int(1)),
            Err(Error::UnknownColumn { column: 99 })
        ));
        assert!(matches!(
            r.lookup_eq(0, &Value::Text("x".into())),
            Err(Error::TypeMismatch {
                column: 0,
                expected: ColumnType::Int,
                got: ColumnType::Text,
            })
        ));
        assert!(matches!(
            r.lookup_range(0, RangeOp::Lt, &Value::Float(1.0)),
            Err(Error::TypeMismatch { .. })
        ));

        // Insert with a wrong-typed value → TypeMismatch, state untouched.
        let bad = vec![
            (0, Value::Text("nope".into())),
            (1, Value::Float(1.0)),
            (2, Value::Text("x".into())),
        ];
        assert!(matches!(
            w.insert_row(Ordinal(1), &bad),
            Err(Error::TypeMismatch { column: 0, .. })
        ));
        assert_eq!(r.live_count(), 1);
    }

    #[test]
    fn range_on_text_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();
        w.insert_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();

        for op in [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge] {
            let got = r.lookup_range(2, op, &Value::Text("x".into())).unwrap();
            assert_eq!(got, bm(&[]), "range on TEXT must be empty, op {op:?}");
        }
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (mut w, _r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();
            w.insert_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
            w.insert_row(Ordinal(1), &row(2, 2.0, "y")).unwrap();
            w.insert_row(Ordinal(2), &row(3, 3.0, "x")).unwrap();
            w.remove_row(Ordinal(1)).unwrap();
            w.checkpoint(Lsn(7)).unwrap();
        }

        let (_w, r, lsn) = MetadataIndex::open(dir.path(), test_schema()).unwrap();
        assert_eq!(lsn, Lsn(7));
        assert_eq!(r.live(), bm(&[0, 2]));
        assert_eq!(r.lookup_eq(2, &Value::Text("x".into())).unwrap(), bm(&[0, 2]));
        assert_eq!(r.lookup_eq(0, &Value::Int(2)).unwrap(), bm(&[])); // tombstoned
        assert_eq!(
            r.lookup_range(1, RangeOp::Ge, &Value::Float(2.0)).unwrap(),
            bm(&[2])
        );
    }

    #[test]
    fn crc_fallback_to_empty() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join(SNAP_FILE);

        let populate = |dir: &Path| {
            let (mut w, _r) = MetadataIndex::create(dir, test_schema()).unwrap();
            w.insert_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
            w.checkpoint(Lsn(5)).unwrap();
        };
        let assert_empty_fallback = |dir: &Path| {
            let (_w, r, lsn) = MetadataIndex::open(dir, test_schema()).unwrap();
            assert_eq!(lsn, Lsn(0));
            assert_eq!(r.live_count(), 0);
            assert_eq!(r.lookup_eq(0, &Value::Int(1)).unwrap(), bm(&[]));
        };

        // Flip one byte in the middle.
        populate(dir.path());
        let mut bytes = std::fs::read(&snap).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&snap, &bytes).unwrap();
        assert_empty_fallback(dir.path());

        // Truncated file.
        populate(dir.path());
        let bytes = std::fs::read(&snap).unwrap();
        std::fs::write(&snap, &bytes[..bytes.len() / 2]).unwrap();
        assert_empty_fallback(dir.path());

        // Pure zeroes.
        std::fs::write(&snap, vec![0u8; 128]).unwrap();
        assert_empty_fallback(dir.path());
    }

    #[test]
    fn crash_mid_checkpoint_keeps_old_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (mut w, _r) = MetadataIndex::create(dir.path(), test_schema()).unwrap();
            w.insert_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
            w.checkpoint(Lsn(5)).unwrap();
        }
        // The crash artifact: garbage tmp that never got renamed.
        std::fs::write(dir.path().join(TMP_FILE), b"garbage from a crash").unwrap();

        let (mut w, r, lsn) = MetadataIndex::open(dir.path(), test_schema()).unwrap();
        assert_eq!(lsn, Lsn(5), "good snap must load; tmp must be ignored");
        assert_eq!(r.lookup_eq(0, &Value::Int(1)).unwrap(), bm(&[0]));

        // A fresh checkpoint overwrites the stale tmp and succeeds.
        w.insert_row(Ordinal(1), &row(2, 2.0, "y")).unwrap();
        w.checkpoint(Lsn(6)).unwrap();
        let (_w, r, lsn) = MetadataIndex::open(dir.path(), test_schema()).unwrap();
        assert_eq!(lsn, Lsn(6));
        assert_eq!(r.live_count(), 2);
    }

    #[test]
    fn schema_mismatch_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        {
            let schema = Schema::from_columns(vec![
                ColumnSpec::Vector {
                    name: "vector".into(),
                    dim: NonZeroUsize::new(1).unwrap(),
                },
                ColumnSpec::Scalar {
                    name: "a".into(),
                    ty: ColumnType::Int,
                },
            ])
            .unwrap();
            let (_w, _r) = MetadataIndex::create(dir.path(), schema).unwrap();
        }
        let other = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(1).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap();
        assert!(matches!(
            MetadataIndex::open(dir.path(), other),
            Err(Error::SchemaMismatch)
        ));
    }
}
