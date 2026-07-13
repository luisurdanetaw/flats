//! Metadata layer (Phase 4).
//!
//! Three sub-phases, built in order:
//!
//!   * **4a — [`index`]**: the metadata *index*. Answers "which ordinals have
//!     column X = / < / > value V?" as RoaringBitmaps. Knows set membership
//!     only; cannot give values back.
//!   * **4b — [`tuples`]**: the tuple *store*. Answers "what are the actual
//!     values at ordinal N?" for SEARCH's RETURNING clause. The inverse
//!     access path of 4a.
//!   * **4c — WAL integration**: `Record::Insert` grows a metadata row and the
//!     engine's `IndexApplier` fans each record out to all three subsystems
//!     (FlatIndex + MetadataIndex + TupleStore). 4c edits `wal.rs` and
//!     `engine/mod.rs`, not this module — the full guide lives in
//!     `docs/phase-4c-wal-integration.md`.
//!
//! Shared plain-data vocabulary (Ordinal, Lsn, Value, Schema, …) lives in
//! [`common`]; both stores speak only those types.

pub mod common;
pub mod index;
pub mod tuples;

pub use common::{ColumnId, ColumnType, Lsn, Ordinal, RangeOp, Row, Schema, Value};
pub use index::MetadataIndex;
pub use tuples::TupleStore;

/// Shared CRC-32 over a byte slice — the integrity check trailing every
/// snapshot file (metadata.snap, tuples.snap, catalog.snap).
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Shared crash-safe snapshot persistence for both stores: write to a temp
/// file, make its bytes durable, atomically rename over the live snapshot,
/// then fsync the directory so the rename itself survives a crash. A crash at
/// ANY point leaves either the old complete snapshot or the new complete one —
/// never a torn file. (A leftover .tmp from a crash is simply overwritten by
/// the next checkpoint and never read by open().)
pub(crate) fn write_snapshot_atomic(
    dir: &std::path::Path,
    tmp_name: &str,
    snap_name: &str,
    bytes: &[u8],
) -> crate::error::Result<()> {
    use std::io::Write as _;
    let tmp = dir.join(tmp_name);
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        // File bytes durable BEFORE the rename: without this, the rename can
        // hit disk before the data and a crash leaves a complete-looking
        // snapshot full of garbage (the CRC would catch it, but a checkpoint
        // would be silently lost).
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(snap_name))?;
    // The rename is a directory-entry mutation; fsync the dir or it can
    // evaporate on crash.
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}
