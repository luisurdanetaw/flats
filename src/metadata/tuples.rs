//! Phase 4b — the tuple store: raw metadata values, addressable by ordinal.
//!
//! The metadata index (4a) answers "WHICH ordinals match?" — bitmaps only; it
//! cannot give values back (a posting list knows membership, not the row).
//! SEARCH's RETURNING clause needs the actual string/int/float values. This
//! store is that inverse access path: ordinal → row values.
//!
//! Same ordinal space as FlatIndex and MetadataIndex (the ONE `Ordinal` type
//! from common.rs) — that is the implicit integration contract.
//!
//! # PERSISTENCE DECISION (spec says pick one and document it — this is it)
//!
//! **Chosen: whole-snapshot serialize-and-rename, identical to Phase 4a.**
//! The spec's alternative was the FlatIndex pattern (mmap + header +
//! double-buffered watermark). Reasons to NOT take it yet:
//!
//!   1. Variable-length values break the "appends at known offsets" premise
//!      that makes FlatIndex's mmap layout simple. A (fixed index region +
//!      data heap) file is a real allocator with growth/remap — the
//!      complexity jump is large and none of it is conceptually new.
//!   2. Metadata rows are small next to vectors (a 768-dim f32 vector is
//!      3 KiB; a row is tens of bytes), so rewriting the snapshot at
//!      checkpoint is cheap relative to work the system already does.
//!   3. One persistence pattern shared by 4a and 4b = one set of crash
//!      semantics to reason about (and one set of bugs).
//!
//! Revisit (graduate to mmap) when checkpoint time shows up in profiles —
//! that's a swap of `checkpoint`/`open` internals; the API doesn't move.
//!
//! # In-memory shape
//!
//! `slots: Vec<Slot>`, indexed directly by ordinal — the ordinal space is
//! dense and append-mostly (FlatIndex allocates sequentially), so a Vec beats
//! a HashMap; sparse writes just grow the Vec with `Vacant` filler.
//! Row values are stored POSITIONALLY (`Vec<Value>` where position ==
//! ColumnId — ids are dense 0..n by Schema construction), so `get` is an
//! index, not a search.
//!
//! Unlike 4a's lazy live-bitmap, deletion here is per-slot (`Tombstone`)
//! because the caller of `get` needs the distinction "was deleted" — the
//! spec's deleted-marker test.
//!
//! # `tuples.snap` layout
//!
//! ```text
//! offset  size  field
//! 0       4     magic       b"TUP0"
//! 4       4     version     u32 = 1
//! 8       8     last_lsn    u64
//! 16      4     body_len    u32
//! 20      var   body        bincode((Schema, Vec<Slot>))
//! end-4   4     crc32       over ALL preceding bytes
//! ```
//!
//! Contrast with 4a on purpose: there we hand-frame (roaring bitmaps have
//! their own serializer), here the whole body is plain serde data so ONE
//! bincode call does it. Same envelope (magic/version/lsn/crc + tmp-fsync-
//! rename-dirfsync), different body strategy.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::metadata::common::{ColumnId, Lsn, Ordinal, Row, Schema, Value};

const MAGIC: &[u8; 4] = b"TUP0";
const VERSION: u32 = 1;
const SNAP_FILE: &str = "tuples.snap";
const TMP_FILE: &str = "tuples.snap.tmp";

use crate::metadata::crc32;

/// One ordinal's storage cell.
//
// Serde: persisted inside the snapshot body. bincode tags variants
// positionally — append only, never reorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Slot {
    /// Never written (a hole from sparse ordinals, or beyond the highest
    /// write). Distinct from Tombstone on purpose.
    Vacant,
    /// Values positionally by ColumnId.
    Live(Vec<Value>),
    /// Written, then deleted. `get` reports this as `RowGet::Deleted`.
    Tombstone,
}

/// What `Reader::get` hands back. An enum instead of Option so the executor
/// can distinguish "deleted row" (normal: skip / surface as gone) from
/// "vacant" (an ordinal the flat index thinks exists but we never stored —
/// under 4c's apply protocol that means replay hasn't caught up or there's a
/// consistency bug; the executor may want to treat it loudly).
#[derive(Debug, Clone, PartialEq)]
pub enum RowGet {
    /// The requested columns' values, in the order requested.
    Live(Vec<Value>),
    /// The spec's "deleted-marker".
    Deleted,
    Missing,
}

struct TupleInner {
    schema: Schema,
    /// slots[ordinal] — dense, grows on demand, Vacant-filled.
    slots: Vec<Slot>,
    /// Same watermark protocol as 4a.
    applied_lsn: Lsn,
}

impl TupleInner {
    fn empty(schema: Schema) -> TupleInner {
        TupleInner {
            schema,
            slots: Vec::new(),
            applied_lsn: Lsn(0),
        }
    }

    /// Make sure `slots[ordinal]` exists, growing with Vacant filler.
    fn slot_mut(&mut self, ordinal: Ordinal) -> &mut Slot {
        let idx = ordinal.0 as usize;
        if idx >= self.slots.len() {
            self.slots.resize(idx + 1, Slot::Vacant);
        }
        &mut self.slots[idx]
    }
}

/// Single mutator; NOT Clone (same posture as 4a — pinned with
/// static_assertions in tests).
pub struct Writer {
    inner: Arc<Mutex<TupleInner>>,
    dir: PathBuf,
}

/// Cloneable read handle.
#[derive(Clone)]
pub struct Reader {
    inner: Arc<Mutex<TupleInner>>,
}

/// Namespace for constructors (mirrors MetadataIndex / FlatIndex).
pub struct TupleStore;

impl TupleStore {
    /// Identical protocol to `MetadataIndex::create`: empty inner, handles,
    /// initial `checkpoint(Lsn(0))` so `open` never sees a missing file for
    /// an existing collection.
    pub fn create(dir: &Path, schema: Schema) -> Result<(Writer, Reader)> {
        let inner = Arc::new(Mutex::new(TupleInner::empty(schema)));
        let mut writer = Writer {
            inner: inner.clone(),
            dir: dir.to_path_buf(),
        };
        let reader = Reader { inner };
        writer.checkpoint(Lsn(0))?;
        Ok((writer, reader))
    }

    /// Identical protocol to `MetadataIndex::open` (read the design deviation
    /// note there — same reasoning, same schema cross-check): a missing or
    /// corrupt snapshot falls back to empty at Lsn(0) (surfaced, not silent;
    /// WAL replay rebuilds), while a schema mismatch is a real error.
    pub fn open(dir: &Path, schema: Schema) -> Result<(Writer, Reader, Lsn)> {
        let snap = dir.join(SNAP_FILE);
        let decoded = match std::fs::read(&snap) {
            Ok(bytes) => match decode_snapshot(&bytes) {
                Ok(v) => Some(v),
                Err(Error::CorruptSnapshot(why)) => {
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
            None => (TupleInner::empty(schema), Lsn(0)),
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

    /// Same contract as `MetadataIndex::open_or_create` — see the idempotent-
    /// materialization note there.
    pub fn open_or_create(dir: &Path, schema: Schema) -> Result<(Writer, Reader, Lsn)> {
        if dir.join(SNAP_FILE).exists() {
            Self::open(dir, schema)
        } else {
            let (w, r) = Self::create(dir, schema)?;
            Ok((w, r, Lsn(0)))
        }
    }
}

impl Writer {
    /// Store the full row at `ordinal`. Idempotent under WAL replay (replay
    /// re-delivers an IDENTICAL row; overwriting a Live slot with the same
    /// values is a no-op in effect).
    ///
    /// Validation runs BEFORE any mutation — the same shared check as 4a, so
    /// nothing is half-written on Err (and NaN is rejected here too: one
    /// boundary, one rule, both stores).
    pub fn write_row(&mut self, ordinal: Ordinal, row: &Row) -> Result<()> {
        let mut inner = self.lock();
        inner.schema.validate_row(row)?;
        // Reshape the (ColumnId, Value) row into positional Vec<Value>.
        // validate_row proved every column appears exactly once, so every
        // position gets filled; ids are dense 0..n by Schema construction.
        let mut values = vec![Value::Int(0); row.len()];
        for (col, value) in row {
            values[*col as usize] = value.clone();
        }
        *inner.slot_mut(ordinal) = Slot::Live(values);
        Ok(())
    }

    /// Tombstone `ordinal`. Idempotent.
    ///
    /// Beyond-the-end or Vacant ordinals grow and get marked anyway — a
    /// replayed Delete may arrive when the snapshot already folded the
    /// Insert away; the marker must still stick.
    pub fn delete_row(&mut self, ordinal: Ordinal) -> Result<()> {
        *self.lock().slot_mut(ordinal) = Slot::Tombstone;
        Ok(())
    }

    /// Same contract as `metadata::index::Writer::advance_applied_lsn`:
    /// monotonic, keeps max(current, lsn).
    pub fn advance_applied_lsn(&mut self, lsn: Lsn) {
        let mut inner = self.lock();
        if lsn > inner.applied_lsn {
            inner.applied_lsn = lsn;
        }
    }

    /// Same contract as 4a's `applied_lsn`.
    pub fn applied_lsn(&self) -> Lsn {
        self.lock().applied_lsn
    }

    /// Same crash-safe dance as 4a — encode under lock, IO outside it:
    /// write tmp → tmp.sync_all() → rename → fsync dir.
    pub fn checkpoint(&mut self, last_lsn: Lsn) -> Result<()> {
        let bytes = {
            let inner = self.lock();
            Self::encode_snapshot(&inner, last_lsn)
        };
        crate::metadata::write_snapshot_atomic(&self.dir, TMP_FILE, SNAP_FILE, &bytes)
    }

    /// See the layout doc at the top of the module. Explicit to_le_bytes on
    /// every integer — never host order.
    fn encode_snapshot(inner: &TupleInner, last_lsn: Lsn) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&last_lsn.0.to_le_bytes());

        // Plain data into memory — no fallible step (same reasoning as
        // Record::encode in wal.rs).
        let body = bincode::serialize(&(&inner.schema, &inner.slots))
            .expect("snapshot body serialization into memory cannot fail");
        buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&body);

        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn lock(&self) -> MutexGuard<'_, TupleInner> {
        // Same poisoning posture as 4a: recover via into_inner, don't unwrap.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Reader {
    /// Fetch `columns`' values at `ordinal`, in the order requested.
    ///
    /// An unknown ColumnId is a caller bug and errs loudly — even for a
    /// Vacant or Tombstoned slot.
    pub fn get(&self, ordinal: Ordinal, columns: &[ColumnId]) -> Result<RowGet> {
        let inner = self.lock();
        for col in columns {
            if *col as usize >= inner.schema.columns.len() {
                return Err(Error::UnknownColumn { column: *col });
            }
        }
        match inner.slots.get(ordinal.0 as usize) {
            None | Some(Slot::Vacant) => Ok(RowGet::Missing),
            Some(Slot::Tombstone) => Ok(RowGet::Deleted),
            Some(Slot::Live(values)) => Ok(RowGet::Live(
                columns.iter().map(|&c| values[c as usize].clone()).collect(),
            )),
        }
    }

    fn lock(&self) -> MutexGuard<'_, TupleInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Inverse of encode_snapshot. The trailing CRC is verified FIRST (see 4a's
/// decode_snapshot for the rationale); any failure below is CorruptSnapshot,
/// which `open` turns into the empty-at-Lsn(0) fallback.
fn decode_snapshot(bytes: &[u8]) -> Result<(TupleInner, Lsn)> {
    let corrupt = |why: &str| Error::CorruptSnapshot(why.into());

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

    if &head[0..4] != MAGIC {
        return Err(corrupt("bad magic"));
    }
    let version = u32::from_le_bytes(head[4..8].try_into().map_err(|_| corrupt("truncated"))?);
    if version != VERSION {
        return Err(Error::CorruptSnapshot(format!("unsupported version {version}")));
    }
    let last_lsn = Lsn(u64::from_le_bytes(
        head[8..16].try_into().map_err(|_| corrupt("truncated"))?,
    ));
    let body_len =
        u32::from_le_bytes(head[16..20].try_into().map_err(|_| corrupt("truncated"))?) as usize;
    let body = head
        .get(20..20 + body_len)
        .ok_or_else(|| corrupt("body length overruns file"))?;
    if head.len() != 20 + body_len {
        return Err(corrupt("trailing bytes"));
    }

    let (schema, slots): (Schema, Vec<Slot>) = bincode::deserialize(body)
        .map_err(|e| Error::CorruptSnapshot(format!("body: {e}")))?;

    Ok((
        TupleInner {
            schema,
            slots,
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
    use crate::metadata::common::ColumnType;

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("a".into(), ColumnType::Int),
            ("b".into(), ColumnType::Float),
            ("c".into(), ColumnType::Text),
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

    static_assertions::assert_not_impl_any!(Writer: Clone);
    static_assertions::assert_impl_all!(Reader: Clone, Send);

    #[test]
    fn write_get_round_trip_per_type() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = TupleStore::create(dir.path(), test_schema()).unwrap();
        w.write_row(Ordinal(0), &row(7, 2.5, "hello")).unwrap();

        // Each column individually.
        assert_eq!(
            r.get(Ordinal(0), &[0]).unwrap(),
            RowGet::Live(vec![Value::Int(7)])
        );
        assert_eq!(
            r.get(Ordinal(0), &[1]).unwrap(),
            RowGet::Live(vec![Value::Float(2.5)])
        );
        assert_eq!(
            r.get(Ordinal(0), &[2]).unwrap(),
            RowGet::Live(vec![Value::Text("hello".into())])
        );

        // Reversed request order: values come back in REQUEST order, not
        // schema order.
        assert_eq!(
            r.get(Ordinal(0), &[2, 1, 0]).unwrap(),
            RowGet::Live(vec![
                Value::Text("hello".into()),
                Value::Float(2.5),
                Value::Int(7),
            ])
        );
    }

    #[test]
    fn sparse_ordinals() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (mut w, r) = TupleStore::create(dir.path(), test_schema()).unwrap();
            w.write_row(Ordinal(0), &row(0, 0.0, "zero")).unwrap();
            w.write_row(Ordinal(1000), &row(1000, 1000.0, "k")).unwrap();

            assert!(matches!(r.get(Ordinal(0), &[0]).unwrap(), RowGet::Live(_)));
            assert!(matches!(r.get(Ordinal(1000), &[0]).unwrap(), RowGet::Live(_)));
            assert_eq!(r.get(Ordinal(500), &[0]).unwrap(), RowGet::Missing);
            assert_eq!(r.get(Ordinal(2000), &[0]).unwrap(), RowGet::Missing);

            w.checkpoint(Lsn(3)).unwrap();
        }
        // Vacant holes survive persistence.
        let (_w, r, lsn) = TupleStore::open(dir.path(), test_schema()).unwrap();
        assert_eq!(lsn, Lsn(3));
        assert!(matches!(r.get(Ordinal(1000), &[2]).unwrap(), RowGet::Live(_)));
        assert_eq!(r.get(Ordinal(500), &[0]).unwrap(), RowGet::Missing);
        assert_eq!(r.get(Ordinal(2000), &[0]).unwrap(), RowGet::Missing);
    }

    #[test]
    fn tombstoned_ordinal_returns_deleted_marker() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = TupleStore::create(dir.path(), test_schema()).unwrap();

        w.write_row(Ordinal(5), &row(5, 5.0, "five")).unwrap();
        w.delete_row(Ordinal(5)).unwrap();
        assert_eq!(r.get(Ordinal(5), &[0]).unwrap(), RowGet::Deleted);

        // Idempotent.
        w.delete_row(Ordinal(5)).unwrap();
        assert_eq!(r.get(Ordinal(5), &[0]).unwrap(), RowGet::Deleted);

        // Delete on a vacant ordinal: the marker must still stick (a
        // replayed Delete whose Insert was folded into the snapshot).
        w.delete_row(Ordinal(9)).unwrap();
        assert_eq!(r.get(Ordinal(9), &[0]).unwrap(), RowGet::Deleted);
    }

    #[test]
    fn unknown_column_in_get_errs() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = TupleStore::create(dir.path(), test_schema()).unwrap();
        w.write_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();

        // Live slot.
        assert!(matches!(
            r.get(Ordinal(0), &[99]),
            Err(Error::UnknownColumn { column: 99 })
        ));
        // Vacant slot: still a loud error, not Missing.
        assert!(matches!(
            r.get(Ordinal(50), &[99]),
            Err(Error::UnknownColumn { column: 99 })
        ));
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (mut w, _r) = TupleStore::create(dir.path(), test_schema()).unwrap();
            w.write_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
            w.write_row(Ordinal(1), &row(2, 2.0, "y")).unwrap();
            w.write_row(Ordinal(3), &row(4, 4.0, "w")).unwrap(); // 2 stays Vacant
            w.delete_row(Ordinal(1)).unwrap();
            w.checkpoint(Lsn(9)).unwrap();
        }

        let (_w, r, lsn) = TupleStore::open(dir.path(), test_schema()).unwrap();
        assert_eq!(lsn, Lsn(9));
        assert_eq!(
            r.get(Ordinal(0), &[0, 2]).unwrap(),
            RowGet::Live(vec![Value::Int(1), Value::Text("x".into())])
        );
        assert_eq!(r.get(Ordinal(1), &[0]).unwrap(), RowGet::Deleted);
        assert_eq!(r.get(Ordinal(2), &[0]).unwrap(), RowGet::Missing);
        assert!(matches!(r.get(Ordinal(3), &[1]).unwrap(), RowGet::Live(_)));
    }

    #[test]
    fn crc_fallback_to_empty() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join(SNAP_FILE);

        let populate = |dir: &Path| {
            let (mut w, _r) = TupleStore::create(dir, test_schema()).unwrap();
            w.write_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
            w.checkpoint(Lsn(5)).unwrap();
        };
        let assert_empty_fallback = |dir: &Path| {
            let (_w, r, lsn) = TupleStore::open(dir, test_schema()).unwrap();
            assert_eq!(lsn, Lsn(0));
            assert_eq!(r.get(Ordinal(0), &[0]).unwrap(), RowGet::Missing);
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
    fn validation_matches_metadata_index() {
        let dir = tempfile::tempdir().unwrap();
        let (mut w, r) = TupleStore::create(dir.path(), test_schema()).unwrap();
        w.write_row(Ordinal(0), &row(1, 1.0, "x")).unwrap();
        let before = r.get(Ordinal(0), &[0, 1, 2]).unwrap();

        // NaN float.
        assert!(matches!(
            w.write_row(Ordinal(0), &row(2, f64::NAN, "y")),
            Err(Error::NaNRejected { column: 1 })
        ));
        assert_eq!(r.get(Ordinal(0), &[0, 1, 2]).unwrap(), before);

        // Wrong type.
        let bad = vec![
            (0, Value::Text("nope".into())),
            (1, Value::Float(1.0)),
            (2, Value::Text("x".into())),
        ];
        assert!(matches!(
            w.write_row(Ordinal(0), &bad),
            Err(Error::TypeMismatch { column: 0, .. })
        ));
        assert_eq!(r.get(Ordinal(0), &[0, 1, 2]).unwrap(), before);

        // Missing column.
        let short = vec![(0, Value::Int(1))];
        assert!(matches!(
            w.write_row(Ordinal(0), &short),
            Err(Error::IncompleteRow)
        ));
        assert_eq!(r.get(Ordinal(0), &[0, 1, 2]).unwrap(), before);
    }
}
