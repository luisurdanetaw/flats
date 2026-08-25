//! Query frontend — the binder / analyze pass (Phase 7e).
//!
//! Front-to-back the query layer is: **lexer → parser → BINDER → planner →
//! bytecode compiler → VM → optimizer**. This module is the third stage, and
//! the first that is NOT pure syntax.
//!
//! # Two-pass design (Postgres-style, not SQLite's fused pass)
//!
//! We deliberately split analysis from planning, because an optimizer and a
//! bytecode compiler are coming and want a clean, already-valid IR:
//!
//! * **Pass 1 — binder (this module).** [`Statement`] + catalog →
//!   [`BoundStatement`]. Resolves every name to a schema **ordinal**,
//!   type-checks every value, and runs *all* the semantic checks the parser
//!   deliberately skipped. This is **fallible** — the one and only layer that
//!   produces semantic errors ([`BindError`]).
//! * **Pass 2 — planner (later commits).** [`BoundStatement`] → logical plan.
//!   Pure structural construction, **infallible**. It consumes the resolved
//!   types re-exported from here and never touches the catalog.
//!
//! The binder's output is still *statement-shaped* ([`BoundSelect`] /
//! [`BoundInsert`] / [`BoundCreate`]) — building the algebra tree is the
//! planner's job, not this one's.
//!
//! # Three architecture-driven rules (this DB is not a normal SQL DB)
//!
//! The binder enforces all three:
//!
//! 1. **Split storage / [`BoundSelect::include_vector`].** The embedding lives
//!    in the flat vector index, separate from the scalar columns, and fetching
//!    it is expensive. `SELECT *` never returns it; naming the vector column
//!    explicitly is the only way to set the flag.
//! 2. **Vector dimension check.** An `INSERT` into a `VECTOR(768)` column
//!    requires a literal of exactly 768 elements — see
//!    [`BindError::DimensionMismatch`].
//! 3. **Exactly one vector column.** Every collection has one flat index, so a
//!    `CREATE` schema has exactly one vector column — see
//!    [`BindError::VectorColumnCount`].

use std::fmt;

use crate::sql::ast::{
    ColumnType, CompareOp, CreateStmt, Expr, InsertStmt, Literal, Projection, SearchStmt,
    SelectStmt, Statement,
};

// ---------------------------------------------------------------------------
// catalog read access
// ---------------------------------------------------------------------------

/// The read-only catalog access the binder needs: resolve a collection name to
/// its resolved [`Schema`]. Deliberately minimal — the binder depends on this
/// one method and nothing else, so it is trivially satisfied by a test double
/// and (in a later phase) by a thin adapter over the engine catalog.
///
/// ## Why a trait, and not the engine catalog directly (7e note)
///
/// The storage engine's persisted schema (`metadata::Schema`) holds only the
/// SCALAR columns and represents the embedding separately as
/// `CollectionConfig.dim`; it records neither the vector column's *name* nor
/// its *declaration ordinal*. So it cannot reconstruct this vector-INCLUSIVE
/// [`Schema`] (where the embedding is a real, ordinal-bearing column) without
/// an engine-side change — a breaking on-disk schema change, out of scope
/// here. Defining the minimal read trait keeps the binder decoupled and
/// unit-testable; only the binder ever touches the catalog.
///
/// EXTEND: add `impl Catalog for <engine adapter>` once the engine records the
/// vector column, mapping `CollectionConfig` → [`Schema`].
pub trait Catalog {
    /// The resolved schema for `name`, or `None` if no such collection exists.
    fn get_collection(&self, name: &str) -> Option<Schema>;
}

// ---------------------------------------------------------------------------
// resolved-schema support types (shared; re-exported for the planner)
// ---------------------------------------------------------------------------

/// A collection's resolved schema: its columns in declaration order, each
/// carrying the ordinal the VM will index by. Distinct from the storage
/// engine's `metadata::Schema` — this one is vector-INCLUSIVE (the embedding is
/// column `@0` in the bootstrap schema) and uses the syntactic [`ColumnType`].
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    /// Columns in declaration order; `columns[i].ordinal == i`.
    pub columns: Vec<ColumnSchema>,
}

impl Schema {
    /// The column named `name`, or `None`. A trivial by-name lookup — the
    /// schema is tiny, so a linear scan is fine.
    pub fn column(&self, name: &str) -> Option<&ColumnSchema> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// One resolved column: its name, type, ordinal (the VM's `Column` index), and
/// whether it is the vector column.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    /// Column name (source case preserved).
    pub name: String,
    /// Declared type (carries the dimension for [`ColumnType::Vector`]).
    pub ty: ColumnType,
    /// Position in the schema — the ordinal a [`ColumnRef`] binds to.
    pub ordinal: usize,
    /// `true` for the single vector column (equivalently `matches!(ty,
    /// ColumnType::Vector(_))`, stored explicitly so consumers need not match).
    pub is_vector: bool,
}

/// A resolved reference to a column: the bound `ordinal` is the whole product
/// of binding; `name` is retained for diagnostics and RETURNING labels.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnRef {
    /// The referenced column's name.
    pub name: String,
    /// Its schema ordinal — the VM's future `Column` index.
    pub ordinal: usize,
}

/// An `INSERT` value that has passed the type check for its target column.
/// Pairs the (possibly coerced) [`Literal`] with the column type it satisfied.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedValue {
    /// The literal value (an `INT` bound to a `FLOAT` column is coerced here).
    pub value: Literal,
    /// The target column's type it type-checked against.
    pub ty: ColumnType,
}

/// A resolved, validated `WHERE` predicate.
///
/// The narrow counterpart to the parser's permissive [`Expr`]: every leaf is
/// `column <op> literal` with the column bound to an ordinal and the literal
/// type-checked against it. The shapes `Expr` allows but V-SQL does not —
/// column-to-column, literal-to-literal, a bare column as a truth value — do
/// not exist in this type, so no later stage has to handle them.
///
/// # Why this shape, and not an expression tree
///
/// Every leaf here is exactly one call to the metadata index
/// ([`lookup_eq`](crate::metadata::index::Reader::lookup_eq) /
/// [`lookup_range`](crate::metadata::index::Reader::lookup_range)), and `And` /
/// `Or` are bitmap intersection and union. That is why the VM needs no
/// comparison opcodes and no per-row predicate evaluation: a `WHERE` clause is
/// answered as set algebra over roaring bitmaps, and the result is an ordinal
/// source a [`Cursor`](crate::engine::cursor::Cursor) opens over directly.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundPredicate {
    /// `column <op> value`, always in that order — [`bind_comparison`]
    /// normalizes a flipped comparison before it gets here.
    Compare {
        /// The column, bound to its schema ordinal.
        column: ColumnRef,
        /// The comparison.
        op: CompareOp,
        /// The comparand, type-checked against the column.
        value: TypedValue,
    },
    /// Both sides must hold — bitmap intersection.
    And(Box<BoundPredicate>, Box<BoundPredicate>),
    /// Either side may hold — bitmap union.
    Or(Box<BoundPredicate>, Box<BoundPredicate>),
}

// ---------------------------------------------------------------------------
// bound statements
// ---------------------------------------------------------------------------

/// A fully resolved, fully validated, statement-shaped IR — the binder's
/// output. Every name is bound to an ordinal and every value type-checked.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundStatement {
    /// A resolved `SELECT`.
    Select(BoundSelect),
    /// A resolved `INSERT`.
    Insert(BoundInsert),
    /// A resolved `CREATE COLLECTION`.
    CreateCollection(BoundCreate),
    /// A resolved `SEARCH`.
    Search(BoundSearch),
    // EXTEND: Delete(BoundDelete), Update(BoundUpdate).
}

/// A resolved `SELECT projection FROM from`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundSelect {
    /// The collection scanned (confirmed to exist).
    pub from: String,
    /// Its resolved schema.
    pub schema: Schema,
    /// The projected columns, in *source* order, each bound to a schema
    /// ordinal. `SELECT *` expands to every non-vector column.
    pub projection: Vec<ColumnRef>,
    /// Split-storage rule (A): `true` iff the embedding must be fetched. `*`
    /// leaves it `false`; naming the vector column sets it `true`.
    pub include_vector: bool,
    /// The resolved `WHERE` predicate, or `None` for an unfiltered scan.
    pub filter: Option<BoundPredicate>,
}

/// A resolved `INSERT` — the row type-checked and reordered to schema order.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundInsert {
    /// Target collection (confirmed to exist).
    pub collection: String,
    /// Its resolved schema.
    pub schema: Schema,
    /// One [`TypedValue`] per supplied column, **in schema order** (the user's
    /// `(cols) VALUES (...)` list may be in any order; the three stores want
    /// canonical order).
    pub row: Vec<TypedValue>,
}

/// One resolved item of a `SEARCH`'s `RETURNING` list.
///
/// A SEARCH can project two things a SELECT cannot, because a ranked read
/// produces them and a plain scan does not: the row's ordinal and its
/// similarity score. Neither is stored, so neither can be a [`ColumnRef`] —
/// hence the enum rather than a widened column list.
#[derive(Debug, Clone, PartialEq)]
pub enum Projected {
    /// A stored scalar column, bound to its schema ordinal like any `SELECT`
    /// column.
    Column(ColumnRef),
    /// A computed pseudo-column.
    Pseudo(Pseudo),
}

/// A value a `SEARCH` computes rather than stores.
///
/// # These names are reserved inside a `RETURNING`
///
/// `id` and `score` are matched case-insensitively and win over any stored
/// column of the same name, so the meaning of `RETURNING score` does not depend
/// on which collection it is aimed at.
///
/// `id` is already reserved engine-wide (`create_collection` rejects a column
/// called `id`, since it would collide with the ordinal SEARCH returns).
/// **`score` is NOT** — a collection may currently declare a `score` column, and
/// that column then becomes unreachable through `RETURNING`. Closing that means
/// adding `score` to the engine's reserved list, which is a storage-layer change
/// and deliberately not made here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pseudo {
    /// The row's ordinal — its stable id within the collection.
    Id,
    /// The similarity the kernel computed for this row against the query.
    Score,
}

impl Pseudo {
    /// The canonical label, for a result header.
    pub fn name(self) -> &'static str {
        match self {
            Pseudo::Id => "id",
            Pseudo::Score => "score",
        }
    }

    /// Recognize a pseudo-column by name, case-insensitively. `None` means the
    /// name must resolve against the schema like an ordinary column.
    fn from_name(name: &str) -> Option<Pseudo> {
        if name.eq_ignore_ascii_case("id") {
            Some(Pseudo::Id)
        } else if name.eq_ignore_ascii_case("score") {
            Some(Pseudo::Score)
        } else {
            None
        }
    }
}

/// A resolved bare `SEARCH` — the ranked read.
///
/// Structurally a `SELECT *` with a different ROW SOURCE: same collection, same
/// schema, same all-scalars projection. Only `k` and `query` are new, and they
/// decide *which* ordinals the read walks and *in what order* — nothing below
/// the row source changes.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundSearch {
    /// The collection searched (confirmed to exist).
    pub from: String,
    /// Its resolved schema.
    pub schema: Schema,
    /// How many nearest rows to return. Validated `>= 1` here, so no later
    /// stage has to wonder — `Db::search` has its own opinion about `k`, and it
    /// should never be the one to discover a meaningless one.
    pub k: u64,
    /// The query vector, already checked against the collection's declared
    /// dimension. A bare `Vec<f32>` rather than a `Literal`: the binder has
    /// proved it is a vector of the right length, so carrying a type that could
    /// still be a string would throw that proof away — and the compiler interns
    /// it as `Const::Vector` unchanged.
    pub query: Vec<f32>,
    /// What the query returns, resolved: stored columns bound to their ordinals,
    /// `id`/`score` as [`Pseudo`] items, in source order.
    ///
    /// When `RETURNING` is absent this holds the DEFAULT — every scalar column
    /// in schema order, the embedding excluded — so a consumer never re-derives
    /// it and bare `SEARCH` needs no special case. One field, because the
    /// compiler now emits from exactly this list; the split that existed while
    /// `RETURNING` was recorded-but-not-executed is gone.
    pub projection: Vec<Projected>,
    /// The resolved `WHERE` predicate, or `None` for an unfiltered search.
    ///
    /// This is a PREFILTER: the ranking runs over the rows the predicate
    /// admits, so `TOP 5 ... WHERE z < 2` returns the five nearest rows *among
    /// those with `z < 2`*. Ranking first and filtering after would return
    /// fewer than five rows and look entirely plausible.
    pub filter: Option<BoundPredicate>,
}

/// A resolved `CREATE COLLECTION`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCreate {
    /// New collection name (confirmed not to already exist).
    pub name: String,
    /// The schema built from the column definitions, with ordinals and the
    /// single vector column identified.
    pub schema: Schema,
    /// Capacity from the `WITH (capacity = ...)` clause.
    pub capacity: u64,
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// A semantic (binding) error — the checks the parser deliberately skipped.
/// The binder is the only layer that produces these.
#[derive(Debug, Clone, PartialEq)]
pub enum BindError {
    /// A referenced collection does not exist in the catalog.
    CollectionNotFound(String),
    /// A `CREATE COLLECTION` names a collection that already exists.
    CollectionExists(String),
    /// A projected/inserted column name is not in the collection's schema.
    ColumnNotFound(String),
    /// An inserted value's type does not match its target column.
    TypeMismatch {
        /// The offending column's name.
        column: String,
        /// The column's declared type.
        expected: ColumnType,
        /// The type of the supplied literal.
        found: ColumnType,
    },
    /// An `INSERT` supplied a different number of values than columns.
    ArityMismatch {
        /// Columns named in the insert list.
        expected: usize,
        /// Values supplied.
        found: usize,
    },
    /// A vector literal's length does not match the column's declared dimension.
    DimensionMismatch {
        /// The vector column's name.
        column: String,
        /// The declared dimension.
        expected: usize,
        /// The supplied literal's length.
        found: usize,
    },
    /// A `CREATE COLLECTION` schema did not have exactly one vector column.
    VectorColumnCount {
        /// How many vector columns were declared (valid schemas have exactly 1).
        found: usize,
    },
    /// A `SEARCH`'s `TOP k` is not a positive count.
    InvalidTopK {
        /// The requested count.
        k: i64,
    },
    /// A `WHERE` expression is well-formed but outside V-SQL's predicate
    /// subset — comparing two columns, ordering a TEXT column, filtering on
    /// the embedding.
    UnsupportedPredicate {
        /// What was attempted, phrased for the message.
        what: String,
    },
    /// A `WHERE` expression is not a predicate at all: a bare column or
    /// literal, with no comparison. V-SQL has no boolean columns and no
    /// truthiness, so there is nothing to interpret.
    NotAPredicate {
        /// What appeared where a predicate was expected.
        what: String,
    },
    /// The statement is valid V-SQL, but this layer does not bind it yet.
    ///
    /// Present so a newly-parseable statement cannot reach an unhandled match
    /// arm: the grammar for a feature lands before its semantics do, and in
    /// that window `Db::execute` must report a clean error rather than panic on
    /// input a user can type. Remove the arm when the binding lands.
    Unbound {
        /// Which statement kind.
        statement: &'static str,
    },
    // EXTEND: DuplicateColumn, UnknownOption, ... as later phases need.
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindError::CollectionNotFound(name) => write!(f, "unknown collection {name:?}"),
            BindError::CollectionExists(name) => write!(f, "collection {name:?} already exists"),
            BindError::ColumnNotFound(name) => write!(f, "unknown column {name:?}"),
            BindError::TypeMismatch {
                column,
                expected,
                found,
            } => write!(
                f,
                "type mismatch for column {column:?}: expected {expected:?}, found {found:?}"
            ),
            BindError::ArityMismatch { expected, found } => write!(
                f,
                "wrong number of values: expected {expected}, found {found}"
            ),
            BindError::DimensionMismatch {
                column,
                expected,
                found,
            } => write!(
                f,
                "vector dimension mismatch for column {column:?}: expected {expected}, found {found}"
            ),
            BindError::VectorColumnCount { found } => write!(
                f,
                "a collection must have exactly one VECTOR column, found {found}"
            ),
            BindError::InvalidTopK { k } => {
                write!(f, "TOP must be at least 1, found {k}")
            }
            BindError::UnsupportedPredicate { what } => {
                write!(f, "unsupported predicate: {what}")
            }
            BindError::NotAPredicate { what } => {
                write!(f, "expected a comparison in WHERE, found {what}")
            }
            BindError::Unbound { statement } => {
                write!(f, "{statement} is parsed but not implemented yet")
            }
        }
    }
}

impl std::error::Error for BindError {}

// ---------------------------------------------------------------------------
// the binder
// ---------------------------------------------------------------------------

/// Analyze `stmt` against `catalog`: resolve names to ordinals, type-check
/// values, and run every semantic check, producing a [`BoundStatement`].
pub fn analyze(stmt: Statement, catalog: &impl Catalog) -> Result<BoundStatement, BindError> {
    match stmt {
        Statement::Select(s) => bind_select(s, catalog).map(BoundStatement::Select),
        Statement::Insert(i) => bind_insert(i, catalog).map(BoundStatement::Insert),
        Statement::CreateCollection(c) => {
            bind_create(c, catalog).map(BoundStatement::CreateCollection)
        }
        Statement::Search(s) => bind_search(s, catalog).map(BoundStatement::Search),
        // EXTEND: Delete/Update dispatch here as those statements land.
    }
}

/// `SELECT projection FROM from`. Resolves the collection, binds each projected
/// column to its schema ordinal, and sets `include_vector` per rule (A).
/// Bind a bare `SEARCH`.
///
/// Three checks, ordered by how fundamental the failure is — the same shape
/// `bind_select` follows:
///
///  1. the collection exists (same lookup, same error as a `SELECT`);
///  2. `k >= 1` — the parser deliberately let `TOP 0` through as syntax, and
///     this is where it stops;
///  3. the query vector's length matches the collection's declared dimension —
///     the same rule, and the same error, an `INSERT`'s embedding gets.
///
/// The projection is not bound from source, because bare `SEARCH` has none: it
/// defaults to every scalar column, exactly as `SELECT *` expands.
fn bind_search(stmt: SearchStmt, catalog: &impl Catalog) -> Result<BoundSearch, BindError> {
    // 1. Collection must exist.
    let schema = catalog
        .get_collection(&stmt.collection)
        .ok_or_else(|| BindError::CollectionNotFound(stmt.collection.clone()))?;

    // 2. `TOP k` must be a real count. `k <= 0` is meaningless, and handing it
    //    to `Db::search` would just move the discovery somewhere less helpful.
    if stmt.k < 1 {
        return Err(BindError::InvalidTopK { k: stmt.k });
    }
    let k = stmt.k as u64;

    // 3. The query must match the collection's embedding dimension. Checked
    //    HERE so a wrong-length query never reaches the SIMD kernel.
    let vector = schema
        .columns
        .iter()
        .find(|c| c.is_vector)
        // Every collection has exactly one vector column by construction.
        .ok_or(BindError::VectorColumnCount { found: 0 })?;
    let dim = match vector.ty {
        ColumnType::Vector(dim) => dim,
        // `is_vector` mirrors `ty`, so this is unreachable; erroring rather than
        // panicking keeps the invariant checked instead of assumed.
        _ => return Err(BindError::VectorColumnCount { found: 0 }),
    };
    // The parser only ever produces `Literal::Vector` here (it requires a `[`),
    // so the other arms are unreachable — but a bind must not panic.
    let query = match stmt.query {
        Literal::Vector(v) => v,
        _ => {
            return Err(BindError::TypeMismatch {
                column: vector.name.clone(),
                expected: ColumnType::Vector(dim),
                found: ColumnType::Text,
            });
        }
    };
    if query.len() != dim {
        return Err(BindError::DimensionMismatch {
            column: vector.name.clone(),
            expected: dim,
            found: query.len(),
        });
    }

    // Default projection: every scalar, in schema order — `SELECT *`'s rule (A).
    let projection: Vec<ColumnRef> = schema
        .columns
        .iter()
        .filter(|c| !c.is_vector)
        .map(|c| ColumnRef {
            name: c.name.clone(),
            ordinal: c.ordinal,
        })
        .collect();

    // 4. Resolve `RETURNING`, if any. Each item is either a reserved
    //    pseudo-column or an ordinary schema lookup — the SAME lookup and the
    //    SAME error a `SELECT` column gets, so an unknown name reads
    //    identically whichever statement it appears in.
    let returning = match stmt.projection {
        None => projection.iter().cloned().map(Projected::Column).collect(),
        Some(names) => {
            let mut items = Vec::with_capacity(names.len());
            for name in names {
                // Pseudo-columns are checked FIRST, so `RETURNING score` means
                // the same thing against every collection (see `Pseudo`).
                if let Some(pseudo) = Pseudo::from_name(&name) {
                    items.push(Projected::Pseudo(pseudo));
                    continue;
                }
                let col = schema
                    .column(&name)
                    .ok_or_else(|| BindError::ColumnNotFound(name.clone()))?;
                // The embedding is not projectable: it lives in the flat index,
                // and `RETURNING vector` would need a `VectorFetch` the ranked
                // read does not emit. Rejected as "no such column" rather than
                // silently dropped.
                if col.is_vector {
                    return Err(BindError::ColumnNotFound(name));
                }
                items.push(Projected::Column(ColumnRef {
                    name: col.name.clone(),
                    ordinal: col.ordinal,
                }));
            }
            items
        }
    };

    // 5. The predicate — a PREFILTER, narrowing the candidate set the ranking
    //    runs over. See `BoundSearch::filter`.
    let filter = stmt
        .filter
        .map(|expr| bind_predicate(expr, &schema))
        .transpose()?;

    Ok(BoundSearch {
        from: stmt.collection,
        schema,
        k,
        query,
        projection: returning,
        filter,
    })
}

fn bind_select(stmt: SelectStmt, catalog: &impl Catalog) -> Result<BoundSelect, BindError> {
    // 1. Collection must exist (the most fundamental failure comes first).
    let schema = catalog
        .get_collection(&stmt.from)
        .ok_or_else(|| BindError::CollectionNotFound(stmt.from.clone()))?;

    // 2/3. Resolve the projection; ordinals come from the SCHEMA, not source
    // order. `include_vector` is set only when the embedding is named.
    let (projection, include_vector) = match stmt.projection {
        Projection::Star => {
            // Rule (A): `*` returns every NON-vector column, never the embedding.
            let cols = schema
                .columns
                .iter()
                .filter(|c| !c.is_vector)
                .map(|c| ColumnRef {
                    name: c.name.clone(),
                    ordinal: c.ordinal,
                })
                .collect();
            (cols, false)
        }
        Projection::Columns(names) => {
            let mut refs = Vec::with_capacity(names.len());
            let mut include_vector = false;
            for name in names {
                let col = schema
                    .column(&name)
                    .ok_or_else(|| BindError::ColumnNotFound(name.clone()))?;
                include_vector |= col.is_vector;
                refs.push(ColumnRef {
                    name: col.name.clone(),
                    ordinal: col.ordinal,
                });
            }
            (refs, include_vector)
        }
    };

    // 4. The predicate, resolved against the SAME schema the projection used.
    let filter = stmt
        .filter
        .map(|expr| bind_predicate(expr, &schema))
        .transpose()?;

    Ok(BoundSelect {
        from: stmt.from,
        schema,
        projection,
        include_vector,
        filter,
    })
}

/// `INSERT INTO collection (cols) VALUES (vals)`. Checks arity, resolves each
/// column, type-checks (and dimension-checks) each value, then reorders the row
/// to schema order.
fn bind_insert(stmt: InsertStmt, catalog: &impl Catalog) -> Result<BoundInsert, BindError> {
    // 1. Collection must exist.
    let schema = catalog
        .get_collection(&stmt.collection)
        .ok_or_else(|| BindError::CollectionNotFound(stmt.collection.clone()))?;

    // 2. Arity: one value per named column. (This PARSES fine in 7d; it is a
    //    semantic error, caught here.)
    if stmt.columns.len() != stmt.values.len() {
        return Err(BindError::ArityMismatch {
            expected: stmt.columns.len(),
            found: stmt.values.len(),
        });
    }

    // 3/4. Resolve + type-check each (column, value) pair, remembering the
    //      target ordinal so we can reorder afterwards.
    let mut placed: Vec<(usize, TypedValue)> = Vec::with_capacity(stmt.columns.len());
    for (name, literal) in stmt.columns.into_iter().zip(stmt.values) {
        let col = schema
            .column(&name)
            .ok_or_else(|| BindError::ColumnNotFound(name.clone()))?;
        let typed = typecheck(col, literal)?;
        placed.push((col.ordinal, typed));
    }

    // 5. Reorder into schema (canonical) order.
    placed.sort_by_key(|(ordinal, _)| *ordinal);
    let row = placed.into_iter().map(|(_, tv)| tv).collect();

    Ok(BoundInsert {
        collection: stmt.collection,
        schema,
        row,
    })
}

/// `CREATE COLLECTION name (cols) WITH (opts)`. Checks the name is free, builds
/// the resolved schema, enforces exactly-one-vector, and reads the capacity.
fn bind_create(stmt: CreateStmt, catalog: &impl Catalog) -> Result<BoundCreate, BindError> {
    // 1. Must not already exist.
    if catalog.get_collection(&stmt.name).is_some() {
        return Err(BindError::CollectionExists(stmt.name));
    }

    // 2. Build the schema: ordinals in declaration order, mark the vector.
    let columns: Vec<ColumnSchema> = stmt
        .columns
        .iter()
        .enumerate()
        .map(|(ordinal, def)| ColumnSchema {
            name: def.name.clone(),
            ty: def.ty,
            ordinal,
            is_vector: matches!(def.ty, ColumnType::Vector(_)),
        })
        .collect();

    // 3. Rule (C): exactly one vector column.
    let vector_count = columns.iter().filter(|c| c.is_vector).count();
    if vector_count != 1 {
        return Err(BindError::VectorColumnCount {
            found: vector_count,
        });
    }

    // 4. Capacity from `WITH (capacity = ...)`. A negative literal (nonsensical
    //    for a count) clamps to 0, which the engine rejects downstream.
    let capacity = stmt
        .options
        .iter()
        .find(|opt| opt.name.eq_ignore_ascii_case("capacity"))
        .map_or(0, |opt| u64::try_from(opt.value).unwrap_or(0));

    Ok(BoundCreate {
        name: stmt.name,
        schema: Schema { columns },
        capacity,
    })
}

/// Type-check `value` against `column`, producing a [`TypedValue`]. `INT`→
/// `FLOAT` coercion into a `FLOAT` column is allowed; a vector column requires
/// a vector literal of the exact declared dimension.
fn typecheck(column: &ColumnSchema, value: Literal) -> Result<TypedValue, BindError> {
    let mismatch = |found: ColumnType| BindError::TypeMismatch {
        column: column.name.clone(),
        expected: column.ty,
        found,
    };
    match column.ty {
        ColumnType::Vector(dim) => match value {
            Literal::Vector(v) => {
                if v.len() == dim {
                    Ok(TypedValue {
                        value: Literal::Vector(v),
                        ty: column.ty,
                    })
                } else {
                    Err(BindError::DimensionMismatch {
                        column: column.name.clone(),
                        expected: dim,
                        found: v.len(),
                    })
                }
            }
            other => Err(mismatch(literal_type(&other))),
        },
        ColumnType::Text => match value {
            Literal::Str(s) => Ok(TypedValue {
                value: Literal::Str(s),
                ty: ColumnType::Text,
            }),
            other => Err(mismatch(literal_type(&other))),
        },
        ColumnType::Int => match value {
            Literal::Int(n) => Ok(TypedValue {
                value: Literal::Int(n),
                ty: ColumnType::Int,
            }),
            other => Err(mismatch(literal_type(&other))),
        },
        ColumnType::Float => match value {
            Literal::Float(f) => Ok(TypedValue {
                value: Literal::Float(f),
                ty: ColumnType::Float,
            }),
            // INT → FLOAT coercion: the value is canonicalized to a float so
            // `value` and `ty` agree in the bound IR.
            Literal::Int(n) => Ok(TypedValue {
                value: Literal::Float(n as f64),
                ty: ColumnType::Float,
            }),
            other => Err(mismatch(literal_type(&other))),
        },
    }
}

// ---------------------------------------------------------------------------
// predicates
// ---------------------------------------------------------------------------

/// Bind a `WHERE` expression into the restricted [`BoundPredicate`] form.
///
/// This is where the parser's permissive [`Expr`] is narrowed to what the
/// metadata index can actually answer: `column <op> literal`, combined with
/// `AND`/`OR`. Everything else is rejected HERE, with an error naming what was
/// wrong, rather than surviving into a plan no executor can run.
fn bind_predicate(expr: Expr, schema: &Schema) -> Result<BoundPredicate, BindError> {
    match expr {
        Expr::And(l, r) => Ok(BoundPredicate::And(
            Box::new(bind_predicate(*l, schema)?),
            Box::new(bind_predicate(*r, schema)?),
        )),
        Expr::Or(l, r) => Ok(BoundPredicate::Or(
            Box::new(bind_predicate(*l, schema)?),
            Box::new(bind_predicate(*r, schema)?),
        )),
        Expr::Compare { left, op, right } => bind_comparison(*left, op, *right, schema),
        // A bare column or literal is not a predicate. V-SQL has no boolean
        // columns and no truthiness, so `WHERE author` cannot mean anything —
        // and guessing (non-empty? non-zero?) would be inventing semantics.
        Expr::Column(name) => Err(BindError::NotAPredicate {
            what: format!("column {name:?}"),
        }),
        Expr::Literal(_) => Err(BindError::NotAPredicate {
            what: "a literal".to_string(),
        }),
    }
}

/// Bind one `left <op> right` into a resolved comparison.
///
/// Accepts the comparison written either way round — `z < 2` and `2 > z` mean
/// the same thing — by NORMALIZING to `column <op> literal`. Normalizing here
/// is what lets every later stage assume the column is on the left, so the
/// compiler never has to think about operand order.
fn bind_comparison(
    left: Expr,
    op: CompareOp,
    right: Expr,
    schema: &Schema,
) -> Result<BoundPredicate, BindError> {
    let (name, op, literal) = match (left, right) {
        (Expr::Column(name), Expr::Literal(lit)) => (name, op, lit),
        // Flipped: `2 > z` becomes `z < 2`. The OPERATOR flips with the
        // operands — reusing it unchanged would silently invert the meaning.
        (Expr::Literal(lit), Expr::Column(name)) => (name, op.flipped(), lit),
        // Out of scope, and each says so specifically. A generic "invalid
        // predicate" would leave the user guessing which half is the problem.
        (Expr::Column(a), Expr::Column(b)) => {
            return Err(BindError::UnsupportedPredicate {
                what: format!("comparing two columns ({a:?} and {b:?})"),
            });
        }
        (Expr::Literal(_), Expr::Literal(_)) => {
            return Err(BindError::UnsupportedPredicate {
                what: "comparing two literals".to_string(),
            });
        }
        // A parenthesized AND/OR used as a comparison operand, e.g.
        // `(a = 1) < 2`.
        _ => {
            return Err(BindError::UnsupportedPredicate {
                what: "a compound expression as a comparison operand".to_string(),
            });
        }
    };

    let column = schema
        .column(&name)
        .ok_or_else(|| BindError::ColumnNotFound(name.clone()))?;

    // The embedding is not filterable: it lives in the flat index, which has no
    // per-value postings to consult. Similarity is what `SEARCH` is for.
    if column.is_vector {
        return Err(BindError::UnsupportedPredicate {
            what: format!("filtering on the vector column {name:?}"),
        });
    }

    // Ordered comparisons on TEXT are refused rather than answered.
    // `lookup_range` returns EMPTY for a text column by design, so accepting
    // `author < 'm'` would produce zero rows and look exactly like a
    // collection with no matching data — a wrong answer that reads as a right
    // one. Equality is fine: that is a dictionary lookup.
    if column.ty == ColumnType::Text && !matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return Err(BindError::UnsupportedPredicate {
            what: format!("ordered comparison on the TEXT column {name:?}"),
        });
    }

    // Reuses the INSERT type check, so `f < 2` against a FLOAT column coerces
    // exactly the way `INSERT ... VALUES (2)` into that column does. One rule,
    // one place.
    let value = typecheck(column, literal)?;

    Ok(BoundPredicate::Compare {
        column: ColumnRef {
            name: column.name.clone(),
            ordinal: column.ordinal,
        },
        op,
        value,
    })
}

/// The [`ColumnType`] a literal presents as, for a [`BindError::TypeMismatch`]
/// diagnostic. A vector literal reports its own length as the dimension.
fn literal_type(lit: &Literal) -> ColumnType {
    match lit {
        Literal::Vector(v) => ColumnType::Vector(v.len()),
        Literal::Str(_) => ColumnType::Text,
        Literal::Int(_) => ColumnType::Int,
        Literal::Float(_) => ColumnType::Float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::parser::parse;
    use std::collections::HashMap;

    // -- test catalog fixture ----------------------------------------------

    /// A hand-built catalog: an in-memory name → [`Schema`] map. This is what
    /// the binder reads; building it is most of the test setup.
    struct TestCatalog {
        schemas: HashMap<String, Schema>,
    }

    impl TestCatalog {
        fn new() -> Self {
            TestCatalog {
                schemas: HashMap::new(),
            }
        }

        fn with(mut self, name: &str, schema: Schema) -> Self {
            self.schemas.insert(name.to_string(), schema);
            self
        }
    }

    impl Catalog for TestCatalog {
        fn get_collection(&self, name: &str) -> Option<Schema> {
            self.schemas.get(name).cloned()
        }
    }

    fn col(name: &str, ty: ColumnType, ordinal: usize, is_vector: bool) -> ColumnSchema {
        ColumnSchema {
            name: name.to_string(),
            ty,
            ordinal,
            is_vector,
        }
    }

    /// The bootstrap `docs` schema: vector@0 (is_vector), author@1, title@2,
    /// published_at@3 — ordinals in declaration order.
    fn docs_schema() -> Schema {
        Schema {
            columns: vec![
                col("vector", ColumnType::Vector(768), 0, true),
                col("author", ColumnType::Text, 1, false),
                col("title", ColumnType::Text, 2, false),
                col("published_at", ColumnType::Int, 3, false),
            ],
        }
    }

    /// A tiny schema carrying a FLOAT column, for the int→float coercion case.
    fn nums_schema() -> Schema {
        Schema {
            columns: vec![
                col("vector", ColumnType::Vector(2), 0, true),
                col("x", ColumnType::Float, 1, false),
            ],
        }
    }

    fn docs_catalog() -> TestCatalog {
        TestCatalog::new().with("docs", docs_schema())
    }

    fn colref(name: &str, ordinal: usize) -> ColumnRef {
        ColumnRef {
            name: name.to_string(),
            ordinal,
        }
    }

    /// Bind a SEARCH against `docs`, returning the bound node.
    fn bound_search(src: &str) -> BoundSearch {
        match analyze(parse(src).expect("parse"), &docs_catalog()).expect("bind") {
            BoundStatement::Search(s) => s,
            other => panic!("expected Search, got {other:?}"),
        }
    }

    /// A vector literal of `n` `0.1` elements, as SQL source.
    fn vec_lit(n: usize) -> String {
        let elems = vec!["0.1"; n].join(", ");
        format!("[{elems}]")
    }

    /// The expected type-checked, schema-ordered row for the bootstrap insert.
    fn bootstrap_row() -> Vec<TypedValue> {
        vec![
            TypedValue {
                value: Literal::Vector(vec![0.1f32; 768]),
                ty: ColumnType::Vector(768),
            },
            TypedValue {
                value: Literal::Str("alice".to_string()),
                ty: ColumnType::Text,
            },
            TypedValue {
                value: Literal::Str("My doc".to_string()),
                ty: ColumnType::Text,
            },
            TypedValue {
                value: Literal::Int(1700000000),
                ty: ColumnType::Int,
            },
        ]
    }

    // -- WHERE / predicates -------------------------------------------------

    /// Bind a `SELECT ... WHERE` against `docs` and return the predicate.
    fn pred(where_clause: &str) -> BoundPredicate {
        let sql = format!("SELECT author FROM docs WHERE {where_clause};");
        match analyze_ok(&sql, &docs_catalog()) {
            BoundStatement::Select(s) => s.filter.expect("a WHERE clause binds to Some"),
            other => panic!("expected a SELECT, got {other:?}"),
        }
    }

    /// Bind a `SELECT ... WHERE` expecting failure.
    fn pred_err(where_clause: &str) -> BindError {
        let sql = format!("SELECT author FROM docs WHERE {where_clause};");
        analyze_err(&sql, &docs_catalog())
    }

    fn compare(name: &str, ordinal: usize, op: CompareOp, value: Literal) -> BoundPredicate {
        let ty = match value {
            Literal::Int(_) => ColumnType::Int,
            Literal::Float(_) => ColumnType::Float,
            Literal::Str(_) => ColumnType::Text,
            Literal::Vector(ref v) => ColumnType::Vector(v.len()),
        };
        BoundPredicate::Compare {
            column: colref(name, ordinal),
            op,
            value: TypedValue { value, ty },
        }
    }

    #[test]
    fn a_comparison_binds_its_column_to_an_ordinal() {
        assert_eq!(
            pred("published_at < 2"),
            compare("published_at", 3, CompareOp::Lt, Literal::Int(2))
        );
    }

    #[test]
    fn a_flipped_comparison_is_normalized_to_column_first() {
        // `2 > published_at` means `published_at < 2`. The OPERATOR flips with
        // the operands — carrying it through unchanged would invert the
        // meaning of every predicate written this way round.
        assert_eq!(pred("2 > published_at"), pred("published_at < 2"));
        assert_eq!(pred("2 >= published_at"), pred("published_at <= 2"));
        assert_eq!(pred("2 < published_at"), pred("published_at > 2"));
        assert_eq!(pred("2 <= published_at"), pred("published_at >= 2"));
        // Symmetric operators map to themselves.
        assert_eq!(pred("2 = published_at"), pred("published_at = 2"));
        assert_eq!(pred("2 != published_at"), pred("published_at != 2"));
    }

    #[test]
    fn and_or_bind_recursively() {
        assert_eq!(
            pred("published_at < 4 AND published_at = 2"),
            BoundPredicate::And(
                Box::new(compare("published_at", 3, CompareOp::Lt, Literal::Int(4))),
                Box::new(compare("published_at", 3, CompareOp::Eq, Literal::Int(2))),
            )
        );
        assert!(matches!(
            pred("published_at < 4 OR published_at = 9"),
            BoundPredicate::Or(..)
        ));
    }

    #[test]
    fn a_predicate_literal_is_type_checked_against_its_column() {
        // The same check an INSERT gets, so one rule governs both.
        assert_eq!(
            pred_err("published_at = 'alice'"),
            BindError::TypeMismatch {
                column: "published_at".to_string(),
                expected: ColumnType::Int,
                found: ColumnType::Text,
            }
        );
    }

    #[test]
    fn an_int_comparand_coerces_into_a_float_column() {
        // `x < 2` against a FLOAT column must work — people do not write `2.0`.
        // Reusing the INSERT typecheck is what gives this for free.
        let cat = TestCatalog::new().with("nums", nums_schema());
        let bound = match analyze_ok("SELECT x FROM nums WHERE x < 2;", &cat) {
            BoundStatement::Select(s) => s.filter.expect("bound"),
            other => panic!("expected a SELECT, got {other:?}"),
        };
        assert_eq!(
            bound,
            compare("x", 1, CompareOp::Lt, Literal::Float(2.0)),
            "the INT literal is canonicalized to a float"
        );
    }

    #[test]
    fn an_unknown_column_in_where_is_the_same_error_as_in_a_projection() {
        assert_eq!(pred_err("nope < 2"), BindError::ColumnNotFound("nope".into()));
    }

    #[test]
    fn ordered_comparison_on_text_is_refused_rather_than_answered() {
        // THE trap this check exists for: `lookup_range` returns EMPTY for a
        // TEXT column by design, so binding `author < 'm'` would produce zero
        // rows — indistinguishable from a collection with no matching data.
        // A wrong answer that reads as a right one is worse than an error.
        assert!(matches!(
            pred_err("author < 'm'"),
            BindError::UnsupportedPredicate { .. }
        ));
        // Equality and inequality on TEXT are fine — that is a dict lookup.
        assert_eq!(
            pred("author = 'alice'"),
            compare("author", 1, CompareOp::Eq, Literal::Str("alice".into()))
        );
        assert!(matches!(pred("author != 'alice'"), BoundPredicate::Compare { .. }));
    }

    #[test]
    fn filtering_on_the_embedding_is_refused() {
        // The flat index has no per-value postings to consult; similarity is
        // what SEARCH is for.
        assert!(matches!(
            pred_err("vector = 'x'"),
            BindError::UnsupportedPredicate { .. }
        ));
    }

    #[test]
    fn out_of_scope_predicate_shapes_are_rejected_specifically() {
        // Each names WHICH half is the problem, rather than a generic
        // "invalid predicate" that leaves the user guessing.
        assert!(matches!(
            pred_err("author = title"),
            BindError::UnsupportedPredicate { .. }
        ));
        assert!(matches!(
            pred_err("1 = 2"),
            BindError::UnsupportedPredicate { .. }
        ));
        // A bare column is not a predicate: V-SQL has no boolean columns and
        // no truthiness, so there is nothing to interpret.
        assert!(matches!(
            pred_err("author"),
            BindError::NotAPredicate { .. }
        ));
    }

    #[test]
    fn search_binds_its_filter_as_a_prefilter() {
        let search = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs WHERE published_at < 2 RETURNING id, score;",
            vec_lit(768)
        ));
        assert_eq!(
            search.filter,
            Some(compare("published_at", 3, CompareOp::Lt, Literal::Int(2)))
        );
        assert_eq!(search.k, 5, "the filter does not disturb TOP");
    }

    #[test]
    fn a_statement_without_where_binds_no_filter() {
        let sql = format!("SEARCH TOP 5 NEAREST TO {} FROM docs;", vec_lit(768));
        assert_eq!(bound_search(&sql).filter, None);
        match analyze_ok("SELECT author FROM docs;", &docs_catalog()) {
            BoundStatement::Select(s) => assert_eq!(s.filter, None),
            other => panic!("expected a SELECT, got {other:?}"),
        }
    }

    fn analyze_ok(src: &str, cat: &impl Catalog) -> BoundStatement {
        analyze(parse(src).expect("test SQL must parse"), cat).expect("expected a successful bind")
    }

    fn analyze_err(src: &str, cat: &impl Catalog) -> BindError {
        match analyze(parse(src).expect("test SQL must parse"), cat) {
            Ok(b) => panic!("expected a bind error, got {b:?}"),
            Err(e) => e,
        }
    }

    fn select(projection: Vec<ColumnRef>, include_vector: bool) -> BoundStatement {
        BoundStatement::Select(BoundSelect {
            from: "docs".to_string(),
            schema: docs_schema(),
            projection,
            include_vector,
            filter: None,
        })
    }

    // -- SELECT ------------------------------------------------------------

    #[test]
    fn select_columns_bind_to_ordinals() {
        let cat = docs_catalog();
        assert_eq!(
            analyze_ok("SELECT author, title FROM docs;", &cat),
            select(vec![colref("author", 1), colref("title", 2)], false),
        );
    }

    #[test]
    fn select_star_projects_non_vector_columns_and_excludes_embedding() {
        // Rule (A): SELECT * → every NON-vector column, include_vector = false.
        let cat = docs_catalog();
        assert_eq!(
            analyze_ok("SELECT * FROM docs;", &cat),
            select(
                vec![
                    colref("author", 1),
                    colref("title", 2),
                    colref("published_at", 3),
                ],
                false,
            ),
        );
    }

    #[test]
    fn select_vector_sets_include_vector() {
        let cat = docs_catalog();
        assert_eq!(
            analyze_ok("SELECT vector FROM docs;", &cat),
            select(vec![colref("vector", 0)], true),
        );
    }

    #[test]
    fn select_scalar_and_vector_includes_embedding() {
        let cat = docs_catalog();
        assert_eq!(
            analyze_ok("SELECT author, vector FROM docs;", &cat),
            select(vec![colref("author", 1), colref("vector", 0)], true),
        );
    }

    #[test]
    fn select_binds_ordinals_from_schema_not_projection_order() {
        // Reversed projection: ordinals still come from the schema (2 then 1).
        let cat = docs_catalog();
        assert_eq!(
            analyze_ok("SELECT title, author FROM docs;", &cat),
            select(vec![colref("title", 2), colref("author", 1)], false),
        );
    }

    #[test]
    fn select_unknown_column_is_column_not_found() {
        let cat = docs_catalog();
        assert_eq!(
            analyze_err("SELECT nope FROM docs;", &cat),
            BindError::ColumnNotFound("nope".to_string()),
        );
    }

    #[test]
    fn select_unknown_collection_is_collection_not_found() {
        let cat = docs_catalog();
        assert_eq!(
            analyze_err("SELECT x FROM ghosts;", &cat),
            BindError::CollectionNotFound("ghosts".to_string()),
        );
    }

    // -- INSERT ------------------------------------------------------------

    #[test]
    fn insert_full_valid_row_is_typechecked_and_ordered() {
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, title, published_at) \
             VALUES ({}, 'alice', 'My doc', 1700000000);",
            vec_lit(768)
        );
        assert_eq!(
            analyze_ok(&src, &cat),
            BoundStatement::Insert(BoundInsert {
                collection: "docs".to_string(),
                schema: docs_schema(),
                row: bootstrap_row(),
            }),
        );
    }

    #[test]
    fn insert_out_of_order_columns_reorder_to_schema_order() {
        // Same row, columns scrambled — must reorder to schema order, identical
        // to the in-order insert.
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (author, vector, published_at, title) \
             VALUES ('alice', {}, 1700000000, 'My doc');",
            vec_lit(768)
        );
        assert_eq!(
            analyze_ok(&src, &cat),
            BoundStatement::Insert(BoundInsert {
                collection: "docs".to_string(),
                schema: docs_schema(),
                row: bootstrap_row(),
            }),
        );
    }

    #[test]
    fn insert_arity_mismatch_is_caught_here_not_in_the_parser() {
        // 4 columns, 3 values — PARSES fine (7d), fails HERE.
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, title, published_at) \
             VALUES ({}, 'alice', 'My doc');",
            vec_lit(768)
        );
        assert_eq!(
            analyze_err(&src, &cat),
            BindError::ArityMismatch {
                expected: 4,
                found: 3,
            },
        );
    }

    #[test]
    fn insert_type_mismatch_string_into_int() {
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, title, published_at) \
             VALUES ({}, 'alice', 'My doc', 'not an int');",
            vec_lit(768)
        );
        assert_eq!(
            analyze_err(&src, &cat),
            BindError::TypeMismatch {
                column: "published_at".to_string(),
                expected: ColumnType::Int,
                found: ColumnType::Text,
            },
        );
    }

    #[test]
    fn insert_int_into_float_column_coerces() {
        // INT → FLOAT into a FLOAT column is allowed; value canonicalized.
        let cat = TestCatalog::new().with("nums", nums_schema());
        let src = "INSERT INTO nums (vector, x) VALUES ([0.1, 0.2], 5);";
        assert_eq!(
            analyze_ok(src, &cat),
            BoundStatement::Insert(BoundInsert {
                collection: "nums".to_string(),
                schema: nums_schema(),
                row: vec![
                    TypedValue {
                        value: Literal::Vector(vec![0.1f32, 0.2f32]),
                        ty: ColumnType::Vector(2),
                    },
                    TypedValue {
                        value: Literal::Float(5.0),
                        ty: ColumnType::Float,
                    },
                ],
            }),
        );
    }

    #[test]
    fn insert_dimension_mismatch() {
        // Rule (B): a 3-element literal into VECTOR(768).
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, title, published_at) \
             VALUES ({}, 'alice', 'My doc', 1);",
            vec_lit(3)
        );
        assert_eq!(
            analyze_err(&src, &cat),
            BindError::DimensionMismatch {
                column: "vector".to_string(),
                expected: 768,
                found: 3,
            },
        );
    }

    #[test]
    fn insert_unknown_column_is_column_not_found() {
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, nope, published_at) \
             VALUES ({}, 'alice', 'x', 1);",
            vec_lit(768)
        );
        assert_eq!(
            analyze_err(&src, &cat),
            BindError::ColumnNotFound("nope".to_string()),
        );
    }

    // -- CREATE ------------------------------------------------------------

    #[test]
    fn create_bootstrap_builds_resolved_schema() {
        // Empty catalog so `docs` is new. Ordinals + is_vector resolved,
        // capacity read from WITH.
        let cat = TestCatalog::new();
        let src = "CREATE COLLECTION docs (
            vector VECTOR(768),
            author TEXT,
            title TEXT,
            published_at INT
        ) WITH (capacity = 1000000);";
        assert_eq!(
            analyze_ok(src, &cat),
            BoundStatement::CreateCollection(BoundCreate {
                name: "docs".to_string(),
                schema: docs_schema(),
                capacity: 1_000_000,
            }),
        );
    }

    #[test]
    fn create_with_zero_vector_columns_is_rejected() {
        // Rule (C).
        let cat = TestCatalog::new();
        let src = "CREATE COLLECTION c (author TEXT, title TEXT) WITH (capacity = 1);";
        assert_eq!(
            analyze_err(src, &cat),
            BindError::VectorColumnCount { found: 0 },
        );
    }

    #[test]
    fn create_with_two_vector_columns_is_rejected() {
        let cat = TestCatalog::new();
        let src = "CREATE COLLECTION c (a VECTOR(4), b VECTOR(8)) WITH (capacity = 1);";
        assert_eq!(
            analyze_err(src, &cat),
            BindError::VectorColumnCount { found: 2 },
        );
    }

    #[test]
    fn create_existing_collection_is_collection_exists() {
        // `docs` already lives in the fixture.
        let cat = docs_catalog();
        let src = "CREATE COLLECTION docs (
            vector VECTOR(768),
            author TEXT,
            title TEXT,
            published_at INT
        ) WITH (capacity = 1000000);";
        assert_eq!(
            analyze_err(src, &cat),
            BindError::CollectionExists("docs".to_string()),
        );
    }

    // -- RETURNING ---------------------------------------------------------

    #[test]
    fn id_and_score_bind_as_pseudocolumns() {
        // Both are COMPUTED, not stored: `id` is the row's ordinal and `score`
        // is what the kernel produced. Neither is looked up in the schema —
        // `docs` has no column by either name, so a binder that resolved them
        // like ordinary columns would reject the statement outright.
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING id, score;",
            vec_lit(768)
        ));
        assert_eq!(
            bound.projection,
            vec![
                Projected::Pseudo(Pseudo::Id),
                Projected::Pseudo(Pseudo::Score),
            ]
        );
        // Recognized case-insensitively, like the engine's reserved `id` rule.
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING ID, Score;",
            vec_lit(768)
        ));
        assert_eq!(
            bound.projection,
            vec![
                Projected::Pseudo(Pseudo::Id),
                Projected::Pseudo(Pseudo::Score),
            ]
        );
    }

    #[test]
    fn returning_a_stored_column_resolves() {
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING title;",
            vec_lit(768)
        ));
        // Bound to the SCHEMA ordinal, exactly as a SELECT column would be —
        // `title` is declaration ordinal 2.
        assert_eq!(
            bound.projection,
            vec![Projected::Column(colref("title", 2))]
        );

        // Stored and computed items mix freely, in source order.
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING score, title, id, author;",
            vec_lit(768)
        ));
        assert_eq!(
            bound.projection,
            vec![
                Projected::Pseudo(Pseudo::Score),
                Projected::Column(colref("title", 2)),
                Projected::Pseudo(Pseudo::Id),
                Projected::Column(colref("author", 1)),
            ]
        );
    }

    #[test]
    fn returning_unknown_column_is_a_bind_error() {
        let src = format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING nope;",
            vec_lit(768)
        );
        assert!(matches!(
            analyze(parse(&src).expect("parse"), &docs_catalog()),
            Err(BindError::ColumnNotFound(name)) if name == "nope"
        ));

        // The embedding is not projectable either: it lives in the flat index,
        // and bare SEARCH returns stored columns.
        let src = format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING vector;",
            vec_lit(768)
        );
        assert!(matches!(
            analyze(parse(&src).expect("parse"), &docs_catalog()),
            Err(BindError::ColumnNotFound(name)) if name == "vector"
        ));
    }

    #[test]
    fn no_returning_keeps_the_default_projection() {
        // Commit 2's behaviour, untouched: every scalar column in schema order,
        // the embedding excluded.
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs;",
            vec_lit(768)
        ));
        let default = vec![
            colref("author", 1),
            colref("title", 2),
            colref("published_at", 3),
        ];
        assert_eq!(
            bound.projection,
            default
                .into_iter()
                .map(Projected::Column)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn returning_replaces_the_default_projection() {
        // A `RETURNING` clause REPLACES the default outright — it does not add
        // to it. `id, score` means two columns, not five.
        let bound = bound_search(&format!(
            "SEARCH TOP 5 NEAREST TO {} FROM docs RETURNING id, score;",
            vec_lit(768)
        ));
        assert_eq!(
            bound.projection,
            vec![
                Projected::Pseudo(Pseudo::Id),
                Projected::Pseudo(Pseudo::Score),
            ]
        );
    }
}
