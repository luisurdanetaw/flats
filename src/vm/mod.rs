//! The virtual machine — the dispatch loop that executes a compiled
//! [`Program`](crate::compiler::Program) against a [`Db`](crate::engine::Db).
//!
//! Front-to-back the query layer is: **lexer → parser → binder → planner →
//! compiler → VM → optimizer**. The VM is the last stage before storage, and
//! the first one that touches the engine at all — everything above it is a pure
//! transformation of a statement into a program.
//!
//! # What lives here now
//!
//! The dispatch loop is not written yet. What is here are the ADAPTERS between
//! the two vocabularies it has to reconcile, each one a place where the
//! bytecode's view of the world and the engine's do not line up:
//!
//! | | bytecode says | the engine wants |
//! |---|---|---|
//! | collection | a `String` name | a `u32` id ([`Db::collection_id`](crate::engine::Db::collection_id)) |
//! | a row | ONE packed record, declaration order | vector and row SEPARATE, storage `ColumnId`s ([`record`]) |
//! | a value | a [`Literal`](crate::sql::ast::Literal) | a [`Value`](crate::metadata::common::Value) ([`value`]) |
//!
//! They are built and tested ahead of the loop because each is a real
//! translation with a way to be silently wrong — not plumbing.

pub mod record;
pub mod value;

pub use record::{Record, SplitError, split_record};
pub use value::ValueError;
