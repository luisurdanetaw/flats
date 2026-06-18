use thiserror::Error;

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
}

pub type Result<T> = std::result::Result<T, Error>;
