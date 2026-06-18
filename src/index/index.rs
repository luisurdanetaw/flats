//! Mmap-backed flat vector index.
//!
//! # On-disk layout (version 2)
//! ```text
//! page 0:  header slot A   (64 bytes used, rest of page reserved)
//! page 1:  header slot B   (64 bytes used, rest of page reserved)
//! page 2+: tombstone bitset (ceil(capacity/8) bytes)
//! page N:  vectors          (f32 * dim * capacity), page-aligned start
//! ```
//! Each header slot (all little-endian):
//!   offset 0  : magic    u32  = b"FLAT"
//!   offset 4  : version  u32  = 2
//!   offset 8  : dim      u32
//!   offset 12 : flags    u32  (reserved, 0)
//!   offset 16 : count    u32  (high-water mark: max written ordinal + 1)
//!   offset 20 : _pad     u32
//!   offset 24 : last_lsn u64  (durable checkpoint watermark — see below)
//!   offset 32 : capacity u64  (fixed at creation)
//!   offset 40 : seq      u64  (monotonic checkpoint counter; newest slot wins)
//!   offset 48 : crc32    u32  (CRC32 of bytes 0..48 of the slot)
//!
//! # Why two header slots (double buffering)
//! The header is the index's *commit record*: `last_lsn` says "every WAL record
//! up to this LSN is durably folded into the data below." A checkpoint must
//! update it atomically w.r.t. crashes. A single header risks a torn write that
//! loses the watermark entirely, which — because the WAL gets truncated up to
//! `last_lsn` — would be unrecoverable. So we keep two slots on *separate pages*
//! and alternate: a checkpoint writes the inactive slot (with `seq+1`) and
//! msyncs only that page. A crash mid-write corrupts at most the slot being
//! written; the other still holds the previous good checkpoint. On open we pick
//! the valid slot with the highest `seq`. Separate pages matter: writeback can
//! tear at page granularity, so co-locating the slots would let one torn page
//! kill both.
//!
//! # Durability model
//! This index is a *materialized view* of the WAL; the WAL's fsync is the
//! durability guarantee. Live writes (`write_at`/`delete`) only touch the page
//! cache and bump in-memory state — they do NOT msync. Durability is advanced
//! only at CHECKPOINTS, driven by the engine's flusher in a strict order:
//!   a. `sync_data`      — msync the bitset + vector pages
//!   b. `stage_watermark`— write the new `(count, last_lsn)` into the spare slot
//!   c. `sync_header`    — msync that slot's page, then make it active
//! Only after (c) is the watermark durable, which is the precondition for the
//! WAL to truncate. Reordering these is a data-loss bug.
//!
//! # Positional, idempotent writes
//! `write_at(ordinal, v)` writes vector `v` into the slot for `ordinal`. It is
//! idempotent: replaying the same record writes the same bytes to the same
//! place. `delete(ordinal)` sets a tombstone bit; setting it twice is a no-op.
//! This is what makes WAL replay safe to over-apply — recovery can replay a
//! record that was already folded in without creating duplicates.
//!
//! # Concurrency (SWMR) — NOTE
//! The lock-free single-writer/multi-reader story is deferred: `write_at` /
//! `delete` currently take `&mut self`, so the engine serializes them (the WAL
//! commit thread is the sole writer) and readers go through the same lock. The
//! atomic `count` is already in place so this can become lock-free later without
//! a format change.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};

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

pub struct FlatIndex {
    mmap: MmapMut,
    dim: NonZeroUsize,
    /// Max vectors the mapping can hold (fixed at creation; no remap yet).
    capacity: usize,
    /// High-water mark: `max(written ordinal) + 1`. Bounds the search scan and
    /// is persisted to the header at checkpoint. Atomic so the eventual
    /// lock-free reader can observe it consistently.
    count: AtomicU32,
    /// Highest LSN whose record has been applied into this mapping (live, in
    /// memory). Its value at checkpoint time becomes the durable `last_lsn`.
    last_applied: AtomicU64,
    /// Durable checkpoint watermark currently on disk (header `last_lsn` of the
    /// active slot). Recovery skips WAL frames at or below this.
    checkpoint_lsn: u64,
    /// Which header slot (0/1) is currently authoritative.
    active_slot: usize,
    /// `seq` of the active slot. The next checkpoint writes `seq + 1`.
    seq: u64,
    /// Byte offset where the vector region starts (page-aligned).
    vectors_offset: usize,
    /// A staged-but-not-yet-active checkpoint: `(slot, seq, count, last_lsn)`.
    /// Set by `stage_watermark`, committed by `sync_header`.
    staged: Option<(usize, u64, u32, u64)>,
}

impl FlatIndex {
    fn bitset_bytes(capacity: usize) -> usize {
        capacity.div_ceil(8)
    }

    fn vectors_offset(capacity: usize) -> usize {
        // Data starts after the bitset, rounded up to a page so the vector
        // region is page-aligned (and therefore comfortably f32-aligned).
        let end = BITSET_OFFSET + Self::bitset_bytes(capacity);
        end.div_ceil(PAGE) * PAGE
    }

    /// Total file/mapping length needed for `capacity` vectors of `dim`.
    fn bytes_for(dim: usize, capacity: usize) -> Result<usize> {
        let vec_bytes = dim
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(F32_BYTES))
            .ok_or(Error::CapacityOverflow { dim, capacity })?;
        Self::vectors_offset(capacity)
            .checked_add(vec_bytes)
            .ok_or(Error::CapacityOverflow { dim, capacity })
    }

    /// Creates a fresh index file at `path`, pre-mapped for up to `capacity`
    /// vectors. Overwrites any existing file.
    pub fn create(path: &Path, dim: NonZeroUsize, capacity: usize) -> Result<FlatIndex> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let len = Self::bytes_for(dim.get(), capacity)?;
        // Sparse: logical size is `len`, only touched pages occupy disk.
        file.set_len(len as u64)?;

        // SAFETY: the file is sized to `len`; the mapping covers exactly it. We
        // are the only writer (single-writer invariant upheld by caller).
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write both slots identically at seq 0 so either is a valid starting
        // point; slot 0 is active by tie-break. The first checkpoint writes
        // slot 1 with seq 1.
        let slot = Slot {
            dim: dim.get() as u32,
            count: 0,
            last_lsn: 0,
            capacity: capacity as u64,
            seq: 0,
        };
        write_slot(&mut mmap, 0, &slot);
        write_slot(&mut mmap, 1, &slot);
        mmap.flush()?; // make the freshly created file valid on disk

        Ok(FlatIndex {
            mmap,
            dim,
            capacity,
            count: AtomicU32::new(0),
            last_applied: AtomicU64::new(0),
            checkpoint_lsn: 0,
            active_slot: 0,
            seq: 0,
            vectors_offset: Self::vectors_offset(capacity),
            staged: None,
        })
    }

    /// Opens an existing index file, selecting the freshest valid header slot.
    pub fn open(path: &Path) -> Result<FlatIndex> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        // SAFETY: file exists; single writer. We validate the header below and
        // never read past the length the header claims.
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let s0 = read_slot(&mmap, 0);
        let s1 = read_slot(&mmap, 1);

        // Pick the valid slot with the highest seq.
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
                // Neither slot validates — report the most specific reason from
                // slot 0's raw bytes.
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
        let vectors_offset = Self::vectors_offset(capacity);

        // Sanity: the file must be large enough for the region the header
        // describes, or the mapping is corrupt/truncated.
        let need = Self::bytes_for(dim.get(), capacity)?;
        if mmap.len() < need {
            return Err(Error::CorruptHeader);
        }

        Ok(FlatIndex {
            mmap,
            dim,
            capacity,
            count: AtomicU32::new(slot.count),
            // Resume the live watermark from the durable one; replay advances it.
            last_applied: AtomicU64::new(slot.last_lsn),
            checkpoint_lsn: slot.last_lsn,
            active_slot,
            seq: slot.seq,
            vectors_offset,
            staged: None,
        })
    }

    /// Current high-water mark (number of ordinal slots in use).
    pub fn len(&self) -> usize {
        self.count.load(AtomicOrdering::Acquire) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Durable checkpoint watermark from the active header slot. The engine
    /// reads this at startup to know which WAL prefix is already folded in.
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn
    }

    /// Positional, idempotent write: place `vector` at `ordinal`'s slot.
    ///
    /// The ordinal IS the position — this never appends. Writing the same
    /// `(ordinal, vector)` again is a no-op in effect (same bytes, same place),
    /// which is what makes WAL replay safe. Bumps the high-water mark to
    /// `ordinal + 1` if this is the furthest slot written so far.
    pub fn write_at(&mut self, ordinal: u64, vector: &[f32]) -> Result<()> {
        let dim = self.dim.get();
        if vector.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }
        if ordinal >= self.capacity as u64 {
            return Err(Error::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let ordinal = ordinal as usize;

        let offset = self.vectors_offset + ordinal * dim * F32_BYTES;
        let byte_len = dim * F32_BYTES;

        // SAFETY: offset..offset+byte_len is within the mapping (ordinal <
        // capacity), and we are the only writer.
        let dst = &mut self.mmap[offset..offset + byte_len];
        // Raw little-endian copy of the f32s (host assumed LE; see note below).
        let src = unsafe { std::slice::from_raw_parts(vector.as_ptr() as *const u8, byte_len) };
        dst.copy_from_slice(src);

        // Publish the high-water mark. `fetch_max` keeps it monotonic and makes
        // out-of-order replay (ordinal lower than current) a no-op on count.
        // Release pairs with readers' Acquire load so they see the bytes above.
        self.count
            .fetch_max(ordinal as u32 + 1, AtomicOrdering::Release);
        Ok(())
    }

    /// Append `vector` at the current high-water mark. Convenience for callers
    /// that don't manage ordinals themselves (tests, simple embedders). The WAL
    /// apply path uses `write_at`, not this.
    pub fn insert(&mut self, vector: &[f32]) -> Result<Ordinal> {
        let ordinal = self.count.load(AtomicOrdering::Acquire) as u64;
        self.write_at(ordinal, vector)?;
        Ok(Ordinal(ordinal as u32))
    }

    /// Tombstone `ordinal`. Idempotent: setting an already-set bit is a no-op
    /// and never errors. Tombstoned ordinals are skipped by `search`.
    pub fn delete(&mut self, ordinal: u64) -> Result<()> {
        if ordinal >= self.capacity as u64 {
            return Err(Error::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let ordinal = ordinal as usize;
        let byte = BITSET_OFFSET + ordinal / 8;
        let bit = (ordinal % 8) as u8;
        // SAFETY: byte < vectors_offset (bitset region) which is within the map.
        self.mmap[byte] |= 1 << bit;
        Ok(())
    }

    fn is_deleted(&self, ordinal: usize) -> bool {
        let byte = BITSET_OFFSET + ordinal / 8;
        let bit = (ordinal % 8) as u8;
        (self.mmap[byte] >> bit) & 1 == 1
    }

    /// Record that the engine has applied up to `lsn` into this mapping. Called
    /// by the apply path after `write_at`/`delete`; its max becomes the durable
    /// watermark at the next checkpoint.
    pub fn advance_applied_lsn(&self, lsn: u64) {
        self.last_applied.fetch_max(lsn, AtomicOrdering::Release);
    }

    /// Read-only view of the vector region `[0, count)`.
    fn vectors(&self) -> &[f32] {
        let count = self.len();
        let floats = count * self.dim.get();
        // SAFETY:
        // - The vector region starts at `vectors_offset`, page-aligned (=> >=
        //   4-aligned), so the f32 pointer is aligned. ✓
        // - `floats <= capacity*dim`, and the mapping is sized for that. ✓
        // - `count` loaded Acquire pairs with the writer's Release, so the bytes
        //   are visible & initialized. ✓
        // - The mapping outlives `&self`; no remap while borrowed. ✓
        let base = unsafe { self.mmap.as_ptr().add(self.vectors_offset) as *const f32 };
        unsafe { std::slice::from_raw_parts(base, floats) }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let dim = self.dim.get();
        if query.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(Error::InvalidTopK { k });
        }

        let vectors = self.vectors();
        let mut heap = BinaryHeap::with_capacity(k);

        for (id, v) in vectors.chunks_exact(dim).enumerate() {
            // Skip tombstoned ordinals — they must never surface in results.
            if self.is_deleted(id) {
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
        out.sort_unstable_by(|a, b| b.score.total_cmp(&a.score)); // most similar first
        Ok(out)
    }

    // ---- checkpoint steps (driven by the engine flusher, in this order) ----

    /// Snapshot `(count, last_lsn)` to checkpoint. Must be read BEFORE
    /// `sync_data` so we never persist a watermark/count covering bytes that
    /// weren't flushed: both refer to writes that happened-before this load, and
    /// `sync_data` flushes at least those.
    pub fn begin_checkpoint(&self) -> (u32, u64) {
        (
            self.count.load(AtomicOrdering::Acquire),
            self.last_applied.load(AtomicOrdering::Acquire),
        )
    }

    /// Step (a): flush the bitset + vector pages. After this, all data up to the
    /// `begin_checkpoint` snapshot is durable on disk.
    pub fn sync_data(&self) -> Result<()> {
        let len = self.mmap.len() - BITSET_OFFSET;
        self.mmap.flush_range(BITSET_OFFSET, len)?;
        Ok(())
    }

    /// Step (b): write the snapshot into the *spare* slot (not yet active). Does
    /// not flush — a crash here leaves the active slot untouched.
    pub fn stage_watermark(&mut self, count: u32, last_lsn: u64) -> Result<()> {
        let target = 1 - self.active_slot;
        let seq = self.seq + 1;
        let slot = Slot {
            dim: self.dim.get() as u32,
            count,
            last_lsn,
            capacity: self.capacity as u64,
            seq,
        };
        write_slot(&mut self.mmap, target, &slot);
        self.staged = Some((target, seq, count, last_lsn));
        Ok(())
    }

    /// Step (c): flush the staged slot's page, then make it active. Only after
    /// this returns Ok is `last_lsn` durable and the WAL safe to truncate.
    pub fn sync_header(&mut self) -> Result<()> {
        let (slot, seq, _count, last_lsn) = match self.staged.take() {
            Some(s) => s,
            None => return Ok(()), // nothing staged; no-op
        };
        let off = if slot == 0 { SLOT0_OFFSET } else { SLOT1_OFFSET };
        self.mmap.flush_range(off, SLOT_BYTES)?;
        // Commit point: the new slot is durable, so adopt it.
        self.active_slot = slot;
        self.seq = seq;
        self.checkpoint_lsn = last_lsn;
        Ok(())
    }

    /// Convenience: run a full checkpoint (a → b → c) in order. Used by tests
    /// and by callers that don't need to interleave WAL truncation between
    /// steps. Returns the durable watermark.
    pub fn sync(&mut self) -> Result<u64> {
        let (count, last_lsn) = self.begin_checkpoint();
        self.sync_data()?;
        self.stage_watermark(count, last_lsn)?;
        self.sync_header()?;
        Ok(last_lsn)
    }
}

// ---- slot read/write helpers ----

fn write_slot(mmap: &mut [u8], slot: usize, s: &Slot) {
    let base = if slot == 0 { SLOT0_OFFSET } else { SLOT1_OFFSET };
    write_u32(mmap, base + F_MAGIC, MAGIC);
    write_u32(mmap, base + F_VERSION, VERSION);
    write_u32(mmap, base + F_DIM, s.dim);
    write_u32(mmap, base + F_FLAGS, 0);
    write_u32(mmap, base + F_COUNT, s.count);
    write_u32(mmap, base + 20, 0); // pad
    write_u64(mmap, base + F_LAST_LSN, s.last_lsn);
    write_u64(mmap, base + F_CAPACITY, s.capacity);
    write_u64(mmap, base + F_SEQ, s.seq);
    let crc = crc32(&mmap[base..base + SLOT_CRC_COVERAGE]);
    write_u32(mmap, base + F_CRC, crc);
}

/// Read and validate one slot. Returns None if out of range, wrong magic/
/// version, or CRC mismatch (a torn or never-written slot).
fn read_slot(mmap: &[u8], slot: usize) -> Option<Slot> {
    let base = if slot == 0 { SLOT0_OFFSET } else { SLOT1_OFFSET };
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

fn read_u32(mmap: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(mmap[offset..offset + 4].try_into().expect("4-byte read"))
}

fn write_u32(mmap: &mut [u8], offset: usize, value: u32) {
    mmap[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_u64(mmap: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(mmap[offset..offset + 8].try_into().expect("8-byte read"))
}

fn write_u64(mmap: &mut [u8], offset: usize, value: u64) {
    mmap[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
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
        let mut idx = FlatIndex::create(&path, dim(2), 16).unwrap();

        idx.insert(&[1.0, 0.0]).unwrap(); // ord 0
        idx.insert(&[2.0, 0.0]).unwrap(); // ord 1
        idx.insert(&[0.0, 1.0]).unwrap(); // ord 2

        let results = idx.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results[0].id, Ordinal(1));
        assert_eq!(results[1].id, Ordinal(0));
    }

    #[test]
    fn reopen_sees_synced_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "reopen.bin");
        {
            let mut idx = FlatIndex::create(&path, dim(3), 8).unwrap();
            idx.insert(&[1.0, 2.0, 3.0]).unwrap();
            idx.insert(&[4.0, 5.0, 6.0]).unwrap();
            idx.sync().unwrap();
        }
        let idx = FlatIndex::open(&path).unwrap();
        assert_eq!(idx.len(), 2);
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, Ordinal(1));
    }

    #[test]
    fn write_at_is_positional_and_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "posn.bin");
        let mut idx = FlatIndex::create(&path, dim(2), 16).unwrap();

        // Positional: writing ordinal 5 leaves a sparse high-water of 6.
        idx.write_at(5, &[1.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 6);

        // Idempotent: applying the same record twice yields identical state.
        idx.write_at(5, &[1.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 6);

        let results = idx.search(&[1.0, 1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, Ordinal(5));
        assert_eq!(results[0].score, 2.0);
    }

    #[test]
    fn delete_is_idempotent_and_hidden_from_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "del.bin");
        let mut idx = FlatIndex::create(&path, dim(2), 8).unwrap();
        idx.write_at(0, &[1.0, 0.0]).unwrap();
        idx.write_at(1, &[2.0, 0.0]).unwrap();

        idx.delete(1).unwrap();
        idx.delete(1).unwrap(); // idempotent, no error

        let results = idx.search(&[1.0, 0.0], 8).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, Ordinal(0)); // ordinal 1 tombstoned, excluded
    }

    #[test]
    fn checkpoint_persists_watermark_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "wm.bin");
        {
            let mut idx = FlatIndex::create(&path, dim(2), 8).unwrap();
            idx.write_at(0, &[1.0, 0.0]).unwrap();
            idx.advance_applied_lsn(42);
            let durable = idx.sync().unwrap();
            assert_eq!(durable, 42);
        }
        let idx = FlatIndex::open(&path).unwrap();
        assert_eq!(idx.checkpoint_lsn(), 42);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn double_buffer_survives_one_corrupt_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "dbuf.bin");
        {
            let mut idx = FlatIndex::create(&path, dim(2), 8).unwrap();
            idx.write_at(0, &[9.0, 9.0]).unwrap();
            idx.advance_applied_lsn(7);
            idx.sync().unwrap(); // writes slot 1, seq 1 -> active
        }
        // Corrupt the *active* slot (slot 1, page 1). Open must fall back to the
        // still-valid slot 0 (the create-time checkpoint, seq 0).
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(SLOT1_OFFSET as u64)).unwrap();
            f.write_all(&[0xAA; 16]).unwrap(); // clobber magic/version/...
            f.sync_all().unwrap();
        }
        let idx = FlatIndex::open(&path).unwrap();
        // Fell back to slot 0: that checkpoint predated the write/advance.
        assert_eq!(idx.checkpoint_lsn(), 0);
    }

    #[test]
    fn insert_rejects_wrong_dim() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "wrongdim.bin");
        let mut idx = FlatIndex::create(&path, dim(3), 4).unwrap();
        assert!(matches!(
            idx.write_at(0, &[1.0, 2.0]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "cap.bin");
        let mut idx = FlatIndex::create(&path, dim(1), 2).unwrap();
        idx.write_at(0, &[1.0]).unwrap();
        idx.write_at(1, &[2.0]).unwrap();
        assert!(matches!(
            idx.write_at(2, &[3.0]),
            Err(Error::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn open_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_path(&dir, "badmagic.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        assert!(matches!(
            FlatIndex::open(&path),
            Err(Error::BadMagic { .. })
        ));
    }
}
