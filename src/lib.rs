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
pub mod metadata;
pub mod sql;
pub mod vm;
pub mod compiler;
pub mod query;

pub use crate::engine::cursor::Cursor;
pub use crate::engine::{CollectionConfig, Db, DbOptions};
pub use crate::error::{Error, Result};
// The query pipeline's boundary types. `query::Error` is deliberately NOT
// re-exported unqualified — `Error` above is the ENGINE's, and two types with
// that name at the crate root would be exactly the confusion the stage-local
// error split exists to prevent (CLAUDE.md §5).
pub use crate::query::{Error as QueryError, RowStream};
pub use crate::metadata::{ColumnSpec, ColumnType, RangeOp, Row, Schema, Value};
pub use simd::dot;
