//! Mmap-backed flat vector index with a lock-free single-writer / many-reader
//! (SWMR) split.
//!
//! # Handles
//! [`FlatIndex::create`]/[`open`](FlatIndex::open) return a `(Writer, Reader)`
//! pair over a shared [`FlatIndexInner`]:
//!   * [`Writer`] is `!Clone` and uniquely owned (by the WAL apply thread). It
//!     is the *only* thing that mutates the mapping. `&mut self` on its methods
//!     is honest: no aliasing is possible because there is exactly one Writer.
//!   * [`Reader`] is `Clone` and handed to any number of query threads. Reader
//!     methods take `&self` and only ever read.
//!
//! The mutex that used to guard the index is gone. Reads run fully in parallel.
//!
//! # How concurrency is made sound (no lock)
//! Publication is via the `AtomicU32` `count` with Release/Acquire:
//!   * The Writer writes a vector into the slot for an ordinal that is `>=` the
//!     currently published `count`, then `count.fetch_max(ordinal+1, Release)`.
//!   * A Reader does `count.load(Acquire)` and scans only ordinals `< count`.
//!
//! The acquire load pairs with the release store, so a Reader that observes a
//! given count also observes every byte the Writer wrote before publishing it.
//! Readers never look at the slot currently being written, so there is no torn
//! read and no data race on the vector region.
//!
//! Tombstones are the one piece of *in-place* mutation below `count` (a Reader
//! may read a bit the Writer is flipping). That byte is therefore touched
//! exclusively through atomic ops (`AtomicU8`), never as a plain `&mut u8`, so
//! the access stays well-defined. The header region (double-buffered slots) is
//! writer-exclusive and only read at open, so it needs no synchronization.
//!
//! # Why a cached raw pointer, not `UnsafeCell<MmapMut>`
//! We keep the `MmapMut` for ownership but never form a `&`/`&mut` to it after
//! construction. Instead we capture its base pointer once (with write
//! provenance, from `as_mut_ptr` while we still hold `&mut`) and do every access
//! through that pointer. Going through `UnsafeCell<MmapMut>` would force a
//! transient `&mut MmapMut` (to call `as_mut_ptr`) on each write while readers
//! hold `&MmapMut` — itself an aliasing violation. Caching the base sidesteps
//! that entirely; the raw pointer is the explicit interior-mutability handle.
//!
//! # On-disk layout (version 2) — unchanged by the split
//! ```text
//! page 0:  header slot A   (64 bytes used)
//! page 1:  header slot B   (64 bytes used)
//! page 2+: tombstone bitset (ceil(capacity/8) bytes)
//! page N:  vectors          (f32 * dim * capacity), page-aligned start
//! ```
//! Each header slot (little-endian): magic u32, version u32, dim u32, flags u32,
//! count u32, _pad u32, last_lsn u64, capacity u64, seq u64, crc32 u32 (over the
//! first 48 bytes). Two slots on separate pages are alternated each checkpoint
//! (newest valid `seq` wins on open) so a torn header flush can't lose the
//! watermark. See the checkpoint ordering on [`Writer`].

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering as AtomicOrdering};

use memmap2::MmapMut;

use crate::error::{Error, Result};
use crate::simd::dot;

const MAGIC: u32 = u32::from_le_bytes(*b"FLAT");
const VERSION: u32 = 2;
const F32_BYTES: usize = std::mem::size_of::<f32>(); // 4
const PAGE: usize = 4096;

// Header geometry. Two slots, each on its own page; data begins on page 2.
const SLOT_BYTES: usize = 64;
const SLOT_CRC_COVERAGE: usize = 48; // crc covers bytes [0, 48) of a slot
const SLOT0_OFFSET: usize = 0;
const SLOT1_OFFSET: usize = PAGE;
const BITSET_OFFSET: usize = 2 * PAGE;

// Field offsets within a slot.
const F_MAGIC: usize = 0;
const F_VERSION: usize = 4;
const F_DIM: usize = 8;
const F_FLAGS: usize = 12;
const F_COUNT: usize = 16;
const F_LAST_LSN: usize = 24;
const F_CAPACITY: usize = 32;
const F_SEQ: usize = 40;
const F_CRC: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ordinal(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub id: Ordinal,
    pub score: f32, // dot product: higher = more similar
}
impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for SearchResult {}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

/// Decoded contents of one header slot.
#[derive(Debug, Clone, Copy)]
struct Slot {
    dim: u32,
    count: u32,
    last_lsn: u64,
    capacity: u64,
    seq: u64,
}

/// Shared state behind both handles. Mutated only by the unique [`Writer`];
/// read by any number of [`Reader`]s, synchronized through `count`.
struct FlatIndexInner {
    /// Owns the mapping. Never referenced after construction — all access goes
    /// through `base` (see module docs). Kept here so the mapping outlives the
    /// handles; dropped (unmapped) when the last `Arc` goes away.
    _mmap: MmapMut,
    /// Base of the mapping, captured from `as_mut_ptr()` at construction so it
    /// carries write provenance over the whole region.
    base: *mut u8,
    /// Mapping length in bytes.
    len: usize,
    dim: NonZeroUsize,
    capacity: usize,
    vectors_offset: usize,
    /// High-water mark: `max(written ordinal) + 1`. THE publication point.
    count: AtomicU32,
}

// SAFETY: `base` is a raw pointer into the shared mapping, which is what makes
// the auto-derived !Send/!Sync. Sharing across threads is sound because:
//   1. There is exactly one writer — `Writer` is `!Clone` and uniquely owned —
//      so writes are never concurrent with each other.
//   2. Vector slots are written only at ordinals >= the published `count` and
//      made visible solely via the Release store on `count`; Readers Acquire-
//      load `count` and read only ordinals below it, so no reference ever
//      aliases a region being written.
//   3. The tombstone bitset (the only in-place mutation) is touched exclusively
//      through `AtomicU8`.
//   4. The header region is writer-exclusive and only read single-threaded at
//      open.
unsafe impl Send for FlatIndexInner {}
unsafe impl Sync for FlatIndexInner {}

impl FlatIndexInner {
    fn count(&self) -> usize {
        self.count.load(AtomicOrdering::Acquire) as usize
    }

    /// Committed vectors `[0, count)` as a flat `&[f32]`. Reader-side.
    fn committed_vectors(&self) -> &[f32] {
        let n = self.count();
        let floats = n * self.dim.get();
        // SAFETY:
        // - `base + vectors_offset` starts the f32 region; `vectors_offset` is
        //   page-aligned (=> 4-aligned), so the pointer is aligned for f32.
        // - `floats <= capacity*dim`, and the mapping is sized for that; in
        //   bounds.
        // - `n` came from an Acquire load that pairs with the Writer's Release
        //   store, so all `floats` are fully written and are never mutated again
        //   (writes only target ordinals >= the published count). No `&mut` to
        //   this region is ever formed, so this shared slice cannot alias one.
        // - The mapping outlives `&self` (held by the Arc); no remap occurs.
        unsafe {
            let p = self.base.add(self.vectors_offset) as *const f32;
            std::slice::from_raw_parts(p, floats)
        }
    }

    /// The tombstone byte holding `ordinal`'s bit, viewed as an atomic.
    fn tombstone_byte(&self, ordinal: usize) -> &AtomicU8 {
        // SAFETY: `BITSET_OFFSET + ordinal/8` is within the bitset region
        // (ordinal < capacity). This byte is accessed ONLY through this
        // `&AtomicU8` (load here, fetch_or in `Writer::delete`), never as a plain
        // `&mut u8`, so concurrent reader/writer access is well-defined atomic
        // access. The pointer carries the mapping's provenance.
        unsafe { &*(self.base.add(BITSET_OFFSET + ordinal / 8) as *const AtomicU8) }
    }

    fn is_deleted(&self, ordinal: usize) -> bool {
        let byte = self.tombstone_byte(ordinal).load(AtomicOrdering::Relaxed);
        (byte >> (ordinal % 8)) & 1 == 1
    }

    /// Atomically set `ordinal`'s tombstone bit. Idempotent. Sound from any
    /// thread: the bitset is an atomic structure (Relaxed suffices — the bit is
    /// an independent flag; the vector it refers to was published via `count`).
    fn set_deleted(&self, ordinal: usize) {
        self.tombstone_byte(ordinal)
            .fetch_or(1 << (ordinal % 8), AtomicOrdering::Relaxed);
    }
}

/// Geometry helpers (independent of any handle).
fn bitset_bytes(capacity: usize) -> usize {
    capacity.div_ceil(8)
}
fn vectors_offset(capacity: usize) -> usize {
    let end = BITSET_OFFSET + bitset_bytes(capacity);
    end.div_ceil(PAGE) * PAGE
}
fn bytes_for(dim: usize, capacity: usize) -> Result<usize> {
    let vec_bytes = dim
        .checked_mul(capacity)
        .and_then(|v| v.checked_mul(F32_BYTES))
        .ok_or(Error::CapacityOverflow { dim, capacity })?;
    vectors_offset(capacity)
        .checked_add(vec_bytes)
        .ok_or(Error::CapacityOverflow { dim, capacity })
}
fn slot_offset(slot: usize) -> usize {
    if slot == 0 { SLOT0_OFFSET } else { SLOT1_OFFSET }
}

/// Factory for the `(Writer, Reader)` pair. Holds no state itself.
pub struct FlatIndex;

impl FlatIndex {
    /// Creates a fresh index file pre-mapped for up to `capacity` vectors,
    /// overwriting any existing file. Returns the writer + a reader.
    pub fn create(path: &Path, dim: NonZeroUsize, capacity: usize) -> Result<(Writer, Reader)> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let len = bytes_for(dim.get(), capacity)?;
        file.set_len(len as u64)?; // sparse: only touched pages occupy disk

        // SAFETY: the file is sized to `len`; the mapping covers exactly it.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Both slots written identically at seq 0; slot 0 is active by tie-break.
        let slot = Slot {
            dim: dim.get() as u32,
            count: 0,
            last_lsn: 0,
            capacity: capacity as u64,
            seq: 0,
        };
        write_slot(&mut mmap[SLOT0_OFFSET..SLOT0_OFFSET + SLOT_BYTES], &slot);
        write_slot(&mut mmap[SLOT1_OFFSET..SLOT1_OFFSET + SLOT_BYTES], &slot);
        mmap.flush()?; // make the freshly created file valid on disk

        Ok(build(mmap, dim, capacity, 0, 0, 0, 0, 0))
    }

    /// Opens an existing index file, selecting the freshest valid header slot.
    pub fn open(path: &Path) -> Result<(Writer, Reader)> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        // SAFETY: file exists; we validate the header before any access and
        // never read past the length the header claims. Single-threaded here.
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let s0 = read_slot(&mmap, 0);
        let s1 = read_slot(&mmap, 1);
        let (active_slot, slot) = match (s0, s1) {
            (Some(a), Some(b)) => {
                if b.seq > a.seq {
                    (1, b)
                } else {
                    (0, a)
                }
            }
            (Some(a), None) => (0, a),
            (None, Some(b)) => (1, b),
            (None, None) => {
                let magic = read_u32(&mmap, SLOT0_OFFSET + F_MAGIC);
                if magic != MAGIC {
                    return Err(Error::BadMagic { got: magic });
                }
                let version = read_u32(&mmap, SLOT0_OFFSET + F_VERSION);
                if version != VERSION {
                    return Err(Error::UnsupportedVersion { got: version });
                }
                return Err(Error::CorruptHeader);
            }
        };

        let dim = NonZeroUsize::new(slot.dim as usize).ok_or(Error::InvalidDimension)?;
        let capacity = slot.capacity as usize;
        if mmap.len() < bytes_for(dim.get(), capacity)? {
            return Err(Error::CorruptHeader);
        }

        Ok(build(
            mmap,
            dim,
            capacity,
            slot.count,
            slot.last_lsn, // resume the live applied-lsn from the durable one
            slot.last_lsn,
            active_slot,
            slot.seq,
        ))
    }
}

/// Wire up a `(Writer, Reader)` over a fresh `Arc<FlatIndexInner>`.
#[allow(clippy::too_many_arguments)]
fn build(
    mut mmap: MmapMut,
    dim: NonZeroUsize,
    capacity: usize,
    count: u32,
    last_applied: u64,
    checkpoint_lsn: u64,
    active_slot: usize,
    seq: u64,
) -> (Writer, Reader) {
    // Capture the write-provenance base BEFORE moving the mmap (and before any
    // sharing). The pointer targets the OS mapping, whose address is stable
    // regardless of where the `MmapMut` struct lives.
    let base = mmap.as_mut_ptr();
    let len = mmap.len();
    let inner = Arc::new(FlatIndexInner {
        _mmap: mmap,
        base,
        len,
        dim,
        capacity,
        vectors_offset: vectors_offset(capacity),
        count: AtomicU32::new(count),
    });
    let writer = Writer {
        inner: Arc::clone(&inner),
        last_applied,
        checkpoint_lsn,
        active_slot,
        seq,
        staged: None,
    };
    let reader = Reader { inner };
    (writer, reader)
}

// ---------------------------------------------------------------------------
// Writer — the unique mutator
// ---------------------------------------------------------------------------

/// The single writer handle. **Not `Clone`** — exactly one exists per index, so
/// `&mut self` truly means exclusive access and no write can race another.
/// Owned by the WAL apply thread.
pub struct Writer {
    inner: Arc<FlatIndexInner>,
    /// Highest LSN applied (in memory). Its value at checkpoint becomes the
    /// durable `last_lsn`. Writer-only, so a plain field.
    last_applied: u64,
    /// Durable checkpoint watermark on disk (active slot's `last_lsn`).
    checkpoint_lsn: u64,
    /// Active header slot (0/1); the next checkpoint writes the other one.
    active_slot: usize,
    seq: u64,
    /// Staged-but-not-active checkpoint: `(slot, seq, count, last_lsn)`.
    staged: Option<(usize, u64, u32, u64)>,
}

impl Writer {
    /// Positional, idempotent write: place `vector` at `ordinal`'s slot.
    ///
    /// The ordinal IS the position — never an append. Re-writing the same
    /// `(ordinal, vector)` is a no-op in effect, which makes WAL replay safe.
    pub fn write_at(&mut self, ordinal: u64, vector: &[f32]) -> Result<()> {
        let dim = self.inner.dim.get();
        if vector.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }
        if ordinal >= self.inner.capacity as u64 {
            return Err(Error::CapacityExceeded {
                capacity: self.inner.capacity,
            });
        }
        let ordinal = ordinal as usize;
        let offset = self.inner.vectors_offset + ordinal * dim * F32_BYTES;
        let byte_len = dim * F32_BYTES;

        // SAFETY: `[offset, offset+byte_len)` is within the mapping (ordinal <
        // capacity). We are the unique Writer. This slot is at `ordinal`, made
        // visible to Readers only by the Release store below, so no Reader can
        // observe these bytes mid-write. We write through `base` (whole-mapping
        // write provenance) with a non-overlapping copy — no `&mut` slice that
        // could alias a Reader's `&[f32]` is ever formed. Host is assumed
        // little-endian (the on-disk format is LE); byte-swap here for BE.
        unsafe {
            let dst = self.inner.base.add(offset);
            std::ptr::copy_nonoverlapping(vector.as_ptr() as *const u8, dst, byte_len);
        }
        // Publish. Release pairs with Readers' Acquire load; fetch_max keeps the
        // high-water mark monotonic and makes out-of-order replay a count no-op.
        self.inner
            .count
            .fetch_max(ordinal as u32 + 1, AtomicOrdering::Release);
        Ok(())
    }

    /// Append `vector` at the current high-water mark (convenience for callers
    /// that don't manage ordinals; the WAL apply path uses `write_at`).
    pub fn insert(&mut self, vector: &[f32]) -> Result<Ordinal> {
        let ordinal = self.inner.count() as u64;
        self.write_at(ordinal, vector)?;
        Ok(Ordinal(ordinal as u32))
    }

    /// Tombstone `ordinal`. Idempotent; never errors on a re-set.
    pub fn delete(&mut self, ordinal: u64) -> Result<()> {
        if ordinal >= self.inner.capacity as u64 {
            return Err(Error::CapacityExceeded {
                capacity: self.inner.capacity,
            });
        }
        // In-place flip below `count`, where Readers may be reading — hence the
        // atomic RMW inside `set_deleted`.
        self.inner.set_deleted(ordinal as usize);
        Ok(())
    }

    /// Record that the engine applied up to `lsn`. Its max becomes the durable
    /// watermark at the next checkpoint.
    pub fn advance_applied_lsn(&mut self, lsn: u64) {
        self.last_applied = self.last_applied.max(lsn);
    }

    /// Durable checkpoint watermark currently on disk. Read at startup to decide
    /// which WAL prefix is already folded in.
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn
    }

    /// Current high-water mark.
    pub fn len(&self) -> usize {
        self.inner.count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ---- checkpoint steps (in this order; see crash-safety note) ----

    /// Snapshot `(count, last_lsn)` to checkpoint. Read BEFORE `sync_data` so we
    /// never persist a watermark/count covering bytes the sync didn't flush.
    pub fn begin_checkpoint(&self) -> (u32, u64) {
        (
            self.inner.count.load(AtomicOrdering::Acquire),
            self.last_applied,
        )
    }

    /// Step (a): flush the bitset + vector pages.
    pub fn sync_data(&self) -> Result<()> {
        let len = self.inner.len - BITSET_OFFSET;
        self.inner._mmap.flush_range(BITSET_OFFSET, len)?;
        Ok(())
    }

    /// Step (b): write the snapshot into the *spare* slot (not yet active).
    pub fn stage_watermark(&mut self, count: u32, last_lsn: u64) -> Result<()> {
        let target = 1 - self.active_slot;
        let seq = self.seq + 1;
        let slot = Slot {
            dim: self.inner.dim.get() as u32,
            count,
            last_lsn,
            capacity: self.inner.capacity as u64,
            seq,
        };
        // SAFETY: the header region is writer-exclusive — Readers never touch it
        // at runtime — so a `&mut` over this slot's bytes cannot alias anyone.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                self.inner.base.add(slot_offset(target)),
                SLOT_BYTES,
            );
            write_slot(bytes, &slot);
        }
        self.staged = Some((target, seq, count, last_lsn));
        Ok(())
    }

    /// Step (c): flush the staged slot's page, then make it active. Only after
    /// this returns Ok is `last_lsn` durable and the WAL safe to truncate.
    pub fn sync_header(&mut self) -> Result<()> {
        let (slot, seq, _count, last_lsn) = match self.staged.take() {
            Some(s) => s,
            None => return Ok(()),
        };
        self.inner._mmap.flush_range(slot_offset(slot), SLOT_BYTES)?;
        self.active_slot = slot;
        self.seq = seq;
        self.checkpoint_lsn = last_lsn;
        Ok(())
    }

    /// Convenience: full checkpoint (a → b → c). Returns the durable watermark.
    pub fn sync(&mut self) -> Result<u64> {
        let (count, last_lsn) = self.begin_checkpoint();
        self.sync_data()?;
        self.stage_watermark(count, last_lsn)?;
        self.sync_header()?;
        Ok(last_lsn)
    }
}

// ---------------------------------------------------------------------------
// Reader — cloneable, read-only
// ---------------------------------------------------------------------------

/// A read handle. `Clone` it freely to query threads; all clones share the same
/// mapping and run fully in parallel (no lock).
#[derive(Clone)]
pub struct Reader {
    inner: Arc<FlatIndexInner>,
}

impl Reader {
    /// Number of committed vectors (high-water mark).
    pub fn len(&self) -> usize {
        self.inner.count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tombstone an ordinal that was never durably committed — e.g. one whose
    /// `insert` reserved the slot but whose WAL append failed, leaving a
    /// zero-filled hole that would otherwise surface in search with score 0.
    ///
    /// Sound to call from any thread (it is on `Reader` precisely because the
    /// engine holds Readers, not the Writer): the tombstone bitset is an atomic
    /// structure, unlike the vector region whose single-writer publication rides
    /// on `count`. This is NOT a general delete — use the Writer / WAL for those.
    pub fn tombstone_uncommitted(&self, ordinal: u64) -> Result<()> {
        if ordinal >= self.inner.capacity as u64 {
            return Err(Error::CapacityExceeded {
                capacity: self.inner.capacity,
            });
        }
        self.inner.set_deleted(ordinal as usize);
        Ok(())
    }

    /// Brute-force top-`k` search. Tombstoned ordinals are skipped. Results are
    /// most-similar (highest dot product) first.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let dim = self.inner.dim.get();
        if query.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(Error::InvalidTopK { k });
        }

        let vectors = self.inner.committed_vectors();
        let mut heap = BinaryHeap::with_capacity(k);

        for (id, v) in vectors.chunks_exact(dim).enumerate() {
            if self.inner.is_deleted(id) {
                continue;
            }
            let hit = SearchResult {
                id: Ordinal(id as u32),
                score: dot(query, v),
            };
            if heap.len() < k {
                heap.push(Reverse(hit));
            } else if let Some(Reverse(worst)) = heap.peek()
                && hit.score > worst.score
            {
                heap.pop();
                heap.push(Reverse(hit));
            }
        }

        let mut out: Vec<SearchResult> = heap.into_iter().map(|Reverse(h)| h).collect();
        out.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }
}

// ---- slot read/write helpers ----

/// Write a slot into its 64-byte region (`bytes` is exactly one slot).
fn write_slot(bytes: &mut [u8], s: &Slot) {
    write_u32(bytes, F_MAGIC, MAGIC);
    write_u32(bytes, F_VERSION, VERSION);
    write_u32(bytes, F_DIM, s.dim);
    write_u32(bytes, F_FLAGS, 0);
    write_u32(bytes, F_COUNT, s.count);
    write_u32(bytes, 20, 0); // pad
    write_u64(bytes, F_LAST_LSN, s.last_lsn);
    write_u64(bytes, F_CAPACITY, s.capacity);
    write_u64(bytes, F_SEQ, s.seq);
    let crc = crc32(&bytes[..SLOT_CRC_COVERAGE]);
    write_u32(bytes, F_CRC, crc);
}

/// Read and validate one slot from the whole mapping. None if out of range,
/// wrong magic/version, or CRC mismatch (torn or never-written slot).
fn read_slot(mmap: &[u8], slot: usize) -> Option<Slot> {
    let base = slot_offset(slot);
    if base + SLOT_BYTES > mmap.len() {
        return None;
    }
    if read_u32(mmap, base + F_MAGIC) != MAGIC {
        return None;
    }
    if read_u32(mmap, base + F_VERSION) != VERSION {
        return None;
    }
    let stored_crc = read_u32(mmap, base + F_CRC);
    if crc32(&mmap[base..base + SLOT_CRC_COVERAGE]) != stored_crc {
        return None;
    }
    Some(Slot {
        dim: read_u32(mmap, base + F_DIM),
        count: read_u32(mmap, base + F_COUNT),
        last_lsn: read_u64(mmap, base + F_LAST_LSN),
        capacity: read_u64(mmap, base + F_CAPACITY),
        seq: read_u64(mmap, base + F_SEQ),
    })
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

// ---- little-endian primitives ----

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().expect("4-byte read"))
}
fn write_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn read_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(buf[offset..offset + 8].try_into().expect("8-byte read"))
}
fn write_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn dim(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("non-zero")
    }
    fn temp_path(dir: &TempDir, name: &str) -> std::path::PathBuf {
        dir.path().join(name)
    }

    #[test]
    fn create_insert_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "roundtrip.bin");
        let (mut w, r) = FlatIndex::create(&path, dim(2), 16).unwrap();

        w.insert(&[1.0, 0.0]).unwrap();
        w.insert(&[2.0, 0.0]).unwrap();
        w.insert(&[0.0, 1.0]).unwrap();

        let results = r.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results[0].id, Ordinal(1));
        assert_eq!(results[1].id, Ordinal(0));
    }

    #[test]
    fn reopen_sees_synced_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "reopen.bin");
        {
            let (mut w, _r) = FlatIndex::create(&path, dim(3), 8).unwrap();
            w.insert(&[1.0, 2.0, 3.0]).unwrap();
            w.insert(&[4.0, 5.0, 6.0]).unwrap();
            w.sync().unwrap();
        }
        let (_w, r) = FlatIndex::open(&path).unwrap();
        assert_eq!(r.len(), 2);
        let results = r.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, Ordinal(1));
    }

    #[test]
    fn write_at_is_positional_and_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "posn.bin");
        let (mut w, r) = FlatIndex::create(&path, dim(2), 16).unwrap();

        w.write_at(5, &[1.0, 1.0]).unwrap();
        assert_eq!(r.len(), 6); // sparse high-water

        w.write_at(5, &[1.0, 1.0]).unwrap(); // idempotent
        assert_eq!(r.len(), 6);

        let results = r.search(&[1.0, 1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, Ordinal(5));
        assert_eq!(results[0].score, 2.0);
    }

    #[test]
    fn delete_is_idempotent_and_hidden_from_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "del.bin");
        let (mut w, r) = FlatIndex::create(&path, dim(2), 8).unwrap();
        w.write_at(0, &[1.0, 0.0]).unwrap();
        w.write_at(1, &[2.0, 0.0]).unwrap();

        w.delete(1).unwrap();
        w.delete(1).unwrap(); // idempotent, no error

        let results = r.search(&[1.0, 0.0], 8).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, Ordinal(0));
    }

    #[test]
    fn checkpoint_persists_watermark_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "wm.bin");
        {
            let (mut w, _r) = FlatIndex::create(&path, dim(2), 8).unwrap();
            w.write_at(0, &[1.0, 0.0]).unwrap();
            w.advance_applied_lsn(42);
            assert_eq!(w.sync().unwrap(), 42);
        }
        let (w, r) = FlatIndex::open(&path).unwrap();
        assert_eq!(w.checkpoint_lsn(), 42);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn double_buffer_survives_one_corrupt_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "dbuf.bin");
        {
            let (mut w, _r) = FlatIndex::create(&path, dim(2), 8).unwrap();
            w.write_at(0, &[9.0, 9.0]).unwrap();
            w.advance_applied_lsn(7);
            w.sync().unwrap(); // writes slot 1, seq 1 -> active
        }
        // Corrupt the active slot (slot 1, page 1). Open must fall back to slot 0.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(SLOT1_OFFSET as u64)).unwrap();
            f.write_all(&[0xAA; 16]).unwrap();
            f.sync_all().unwrap();
        }
        let (w, _r) = FlatIndex::open(&path).unwrap();
        assert_eq!(w.checkpoint_lsn(), 0); // fell back to the create-time slot
    }

    #[test]
    fn insert_rejects_wrong_dim() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "wrongdim.bin");
        let (mut w, _r) = FlatIndex::create(&path, dim(3), 4).unwrap();
        assert!(matches!(
            w.write_at(0, &[1.0, 2.0]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "cap.bin");
        let (mut w, _r) = FlatIndex::create(&path, dim(1), 2).unwrap();
        w.write_at(0, &[1.0]).unwrap();
        w.write_at(1, &[2.0]).unwrap();
        assert!(matches!(
            w.write_at(2, &[3.0]),
            Err(Error::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn open_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "badmagic.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        assert!(matches!(FlatIndex::open(&path), Err(Error::BadMagic { .. })));
    }
}
