//! Flats: a tiny, embeddable vector database.
//!
//! v1 is brute-force flat search backed by mmap'd vectors and a write-ahead
//! log. See `CLAUDE.md` for design scope and rationale.
//!
//! Milestone 1 ships only the dot-product kernel. The public API surface
//! ([`Db`], `insert`, `search`) lands in later milestones.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod engine;
pub mod error;
pub mod index;
pub mod simd;
pub mod wal;

pub use crate::engine::{CollectionConfig, Db, DbOptions};
pub use crate::error::{Error, Result};
pub use simd::dot;
