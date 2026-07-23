use thiserror::Error;

use crate::metadata::common::ColumnType;

#[derive(Error, Debug)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("dimension cannot be 0")]
    InvalidDimension,

    #[error("invalid top k: {k} - k should be a natural number > 0")]
    InvalidTopK { k: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("index file's header is corrupt")]
    CorruptHeader,

    #[error("bad magic number: {got}")]
    BadMagic { got: u32 },

    #[error("unsupported index file version: {got}")]
    UnsupportedVersion { got: u32 },

    #[error("collection is at capacity: {capacity}")]
    CapacityExceeded { capacity: usize },

    #[error("capacity overflow: dim={dim}, capacity={capacity} exceeds usize")]
    CapacityOverflow { dim: usize, capacity: usize },

    #[error("unknown collection: {id}")]
    UnknownCollection { id: u32 },

    /// A config supplied to `Db::open` disagrees (dim, capacity, or schema)
    /// with what the persisted catalog has registered under the same id.
    #[error("collection {id}'s config conflicts with the persisted catalog")]
    CollectionConfigMismatch { id: u32 },

    #[error("collection name already exists: {name}")]
    CollectionExists { name: String },

    #[error("collection name must be non-empty")]
    InvalidCollectionName,

    #[error("capacity must be > 0")]
    InvalidCapacity,

    /// Schema uses a column name the engine reserves (the `id` pseudo-column
    /// SEARCH returns is the ordinal, not a stored column).
    #[error("reserved column name: {0}")]
    ReservedColumn(String),

    // -- metadata layer (Phase 4) ------------------------------------------
    // Callers in Phase 4c only distinguish "retryable via WAL replay" (Io)
    // from "caller bug" (everything below). Keep variants coarse.
    #[error("unknown column: {column}")]
    UnknownColumn { column: u32 },

    #[error("type mismatch on column {column}: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        column: u32,
        expected: ColumnType,
        got: ColumnType,
    },

    /// NaN handed to a Float column. Rejected at the boundary so the
    /// BTreeMap ordering invariant can never be violated.
    #[error("NaN rejected for float column {column}")]
    NaNRejected { column: u32 },

    /// Row missing a column, or has extras/duplicates (no NULLs in Phase 4).
    #[error("row does not match schema (every column exactly once, no NULLs)")]
    IncompleteRow,

    #[error("duplicate column in schema: {0}")]
    DuplicateColumn(String),

    /// A schema must declare exactly one VECTOR column — every collection is
    /// provisioned with exactly one flat vector index.
    #[error("schema must have exactly one vector column, found {found}")]
    VectorColumnCount { found: usize },

    /// Snapshot file exists but failed magic/version/CRC checks.
    /// Phase 4a/4b policy: `open()` swallows this internally ("start empty at
    /// lsn 0, WAL replay rebuilds") — the variant exists for the cases that
    /// must NOT be swallowed, e.g. a schema that contradicts the caller's.
    #[error("corrupt snapshot: {0}")]
    CorruptSnapshot(String),

    #[error("snapshot schema does not match the schema supplied by the caller")]
    SchemaMismatch,
}

pub type Result<T> = std::result::Result<T, Error>;
