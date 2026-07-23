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
    ColumnType, CreateStmt, InsertStmt, Literal, Projection, SelectStmt, Statement,
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
    // EXTEND: Search(BoundSearch), Delete(BoundDelete), Update(BoundUpdate).
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
    // EXTEND: pub filter: Option<BoundPredicate>,  // WHERE, later.
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
        // EXTEND: Search/Delete/Update dispatch here as those statements land.
    }
}

/// `SELECT projection FROM from`. Resolves the collection, binds each projected
/// column to its schema ordinal, and sets `include_vector` per rule (A).
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

    Ok(BoundSelect {
        from: stmt.from,
        schema,
        projection,
        include_vector,
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
}
