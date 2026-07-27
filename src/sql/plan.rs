//! Logical plan — the relational-algebra IR (Phase 7f types / 7g logic).
//!
//! Front-to-back the query layer is: **lexer → parser → binder → PLANNER →
//! bytecode compiler → VM → optimizer**. This module is the planner's output.
//!
//! # Where this sits in the two-pass split
//!
//! The [binder](crate::sql::bind) already did the hard part: it resolved every
//! name to an ordinal, type-checked every value, and produced a
//! [`BoundStatement`](crate::sql::bind::BoundStatement) — a *resolved,
//! validated, statement-shaped* IR. The planner ([`super::planner`]) takes that
//! and changes only the **shape**: statement-shaped → relational-algebra tree
//! (`Project` wrapping `Scan`, etc.), carrying the binder's resolved fields
//! straight through. Because everything was already checked, planning is
//! **infallible** — see [`plan`](crate::sql::planner::plan).
//!
//! # Types
//!
//! This file is **types only** — no logic. The resolved support types
//! ([`Schema`], [`ColumnRef`], [`TypedValue`]) are REUSED from the binder, not
//! redefined here, so a plan and its originating bound statement speak in
//! exactly the same resolved vocabulary.

// Every node derives Debug + Clone + PartialEq: the planner tests assert whole
// plans with `assert_eq!`, so structural equality is load-bearing.

use crate::sql::bind::{ColumnRef, Schema, TypedValue};

/// One resolved query plan — the relational-algebra root the compiler consumes.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Read a collection's rows (leaf of a read tree).
    Scan(Scan),
    /// Project columns out of its input.
    Project(Project),
    /// A single, resolved row insert.
    Insert(Insert),
    /// Provision a new collection.
    CreateCollection(CreateCollection),
    /// Rank a collection's rows by similarity to a query vector (leaf of a
    /// ranked read tree).
    Knn(Knn),
    // EXTEND: Filter(Filter) for WHERE — a new variant plus a match arm in the
    // planner, without reworking the core.
}

/// Read every live row of a collection. The leaf of a read plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    /// The collection name (already confirmed to exist by the binder).
    pub collection: String,
    /// Its resolved schema — carried so downstream nodes share its ordinals.
    pub schema: Schema,
    // EXTEND: predicate: Option<Predicate>  // pushed-down WHERE / bitmap mask.
}

/// Rank a collection's rows by similarity to a query vector, nearest first.
///
/// The LEAF of a ranked read, sitting exactly where a [`Scan`] sits under the
/// same [`Project`]. That is the whole design: a `SEARCH` and a `SELECT` differ
/// only in WHERE THE ORDINALS COME FROM — `Scan` walks every live row, `Knn`
/// walks the top `k` in score order — and everything above is identical.
///
/// It carries no projection of its own for the same reason: the projection
/// belongs to the [`Project`] above it, so there is exactly one place that
/// decides which columns a read returns, shared by both statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Knn {
    /// The collection searched (already confirmed to exist by the binder).
    pub collection: String,
    /// Its resolved schema — carried so downstream nodes share its ordinals.
    pub schema: Schema,
    /// How many nearest rows to return (validated `>= 1` by the binder).
    pub k: u64,
    /// The query vector, already checked against the collection's dimension.
    pub query: Vec<f32>,
    // EXTEND: prefilter: Option<Predicate> — the WHERE-mask a `BitmapFrom`
    // would supply, narrowing the candidate set before ranking.
}

/// Project a column list out of `input`. Wraps a [`Scan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    /// The plan tree below this node — a [`LogicalPlan::Scan`] for a `SELECT`,
    /// a [`LogicalPlan::Knn`] for a `SEARCH`. Boxed because a plan is
    /// recursive.
    pub input: Box<LogicalPlan>,
    /// The projected columns, bound to schema ordinals — carried through from
    /// the bound projection.
    pub columns: Vec<ColumnRef>,
    /// Split-storage flag, carried straight from the binder: `true` iff the
    /// embedding must be fetched from the flat vector index.
    pub include_vector: bool,
}

/// A single resolved row insert.
#[derive(Debug, Clone, PartialEq)]
pub struct Insert {
    /// Target collection.
    pub collection: String,
    /// Its resolved schema.
    pub schema: Schema,
    /// One [`TypedValue`] per schema column, in schema order (the binder
    /// already reordered and type-checked it; the planner does not touch it).
    pub row: Vec<TypedValue>,
}

/// Provision a new collection with a resolved schema.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateCollection {
    /// New collection name.
    pub name: String,
    /// The resolved schema (ordinals + the single vector column identified).
    pub schema: Schema,
    /// Capacity from the `WITH (capacity = ...)` clause.
    pub capacity: u64,
}
