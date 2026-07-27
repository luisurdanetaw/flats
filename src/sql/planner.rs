//! Logical planner — Pass 2 of the two-pass frontend (Phase 7f skeleton /
//! 7g logic).
//!
//! Takes the [binder](crate::sql::bind)'s
//! [`BoundStatement`](crate::sql::bind::BoundStatement) and builds the
//! relational-algebra [`LogicalPlan`]:
//!
//! ```text
//! BoundSelect -> Project(Scan)     BoundInsert -> Insert
//! BoundCreate -> CreateCollection
//! ```
//!
//! # Infallible by construction
//!
//! [`plan`] returns [`LogicalPlan`], **not** `Result`, and takes **no catalog**.
//! Every semantic failure — unknown collection/column, arity, type, dimension,
//! one-vector — was already caught by the binder, which owns all resolution and
//! all errors. The planner only reshapes already-resolved, already-valid input;
//! it cannot fail and has nothing to look up. That infallible, catalog-free
//! signature is the whole point of splitting analysis from planning.
//!

use crate::sql::bind::{BoundCreate, BoundInsert, BoundSearch, BoundSelect, BoundStatement};
use crate::sql::plan::{CreateCollection, Insert, Knn, LogicalPlan, Project, Scan};

/// Build the [`LogicalPlan`] for an already-bound statement. Infallible: the
/// binder validated everything, so this only reshapes statement-shaped IR into
/// the algebra tree.
pub fn plan(bound: BoundStatement) -> LogicalPlan {
    match bound {
        BoundStatement::Select(s) => plan_select(s),
        BoundStatement::Insert(i) => plan_insert(i),
        BoundStatement::CreateCollection(c) => plan_create(c),
        BoundStatement::Search(s) => plan_search(s),
        // EXTEND: Delete/Update -> their own nodes.
    }
}

// Per-variant shape transforms. Each carries the binder's resolved fields
// through unchanged — no catalog, no re-checks, no reordering.

/// `BoundSelect` → `Project` wrapping a `Scan`, carrying the bound projection
/// and `include_vector` through unchanged.
fn plan_select(bound: BoundSelect) -> LogicalPlan {
    LogicalPlan::Project(Project {
        input: Box::new(LogicalPlan::Scan(Scan {
            collection: bound.from,
            schema: bound.schema,
        })),
        columns: bound.projection,
        include_vector: bound.include_vector,
    })
}

/// `BoundSearch` → `Project` over `Knn`.
///
/// The SAME shape `plan_select` produces, with `Knn` where `Scan` would be. That
/// is not a coincidence to be tidied away later — it is the point: a ranked read
/// and a full read differ only in their row source, so the compiler emits one
/// loop for both and only the prologue that opens the cursor differs.
///
/// `include_vector` is `false`: bare `SEARCH` returns the matched rows' stored
/// columns, never the embedding — the same rule as `SELECT *`. `RETURNING` is
/// what will let a caller ask for more.
fn plan_search(bound: BoundSearch) -> LogicalPlan {
    LogicalPlan::Project(Project {
        input: Box::new(LogicalPlan::Knn(Knn {
            collection: bound.from,
            schema: bound.schema,
            k: bound.k,
            query: bound.query,
        })),
        columns: bound.projection,
        include_vector: false,
    })
}

/// `BoundInsert` → `Insert`, carrying collection/schema/row through (the row is
/// already in schema order from the binder; the planner does not reorder).
fn plan_insert(bound: BoundInsert) -> LogicalPlan {
    LogicalPlan::Insert(Insert {
        collection: bound.collection,
        schema: bound.schema,
        row: bound.row,
    })
}

/// `BoundCreate` → `CreateCollection`, carrying name/schema/capacity through.
fn plan_create(bound: BoundCreate) -> LogicalPlan {
    LogicalPlan::CreateCollection(CreateCollection {
        name: bound.name,
        schema: bound.schema,
        capacity: bound.capacity,
    })
}

#[cfg(test)]
mod tests {
    use super::plan;
    use crate::sql::ast::{ColumnType, Literal};
    use crate::sql::bind::{
        BoundCreate, BoundInsert, BoundSearch, BoundSelect, BoundStatement, Catalog, ColumnRef,
        ColumnSchema, Schema, TypedValue, analyze,
    };
    use crate::sql::parser::parse;
    use crate::sql::plan::{CreateCollection, Insert, Knn, LogicalPlan, Project, Scan};
    use std::collections::HashMap;

    // -- resolved-value helpers (mirror the binder's docs fixture) ---------

    fn col(name: &str, ty: ColumnType, ordinal: usize, is_vector: bool) -> ColumnSchema {
        ColumnSchema {
            name: name.to_string(),
            ty,
            ordinal,
            is_vector,
        }
    }

    /// The bootstrap `docs` schema: vector@0 (is_vector), author@1, title@2,
    /// published_at@3.
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

    fn colref(name: &str, ordinal: usize) -> ColumnRef {
        ColumnRef {
            name: name.to_string(),
            ordinal,
        }
    }

    fn vec_lit(n: usize) -> String {
        let elems = vec!["0.1"; n].join(", ");
        format!("[{elems}]")
    }

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

    /// A `Project` over the `docs` scan, for terse expectations.
    fn docs_project(columns: Vec<ColumnRef>, include_vector: bool) -> LogicalPlan {
        LogicalPlan::Project(Project {
            input: Box::new(LogicalPlan::Scan(Scan {
                collection: "docs".to_string(),
                schema: docs_schema(),
            })),
            columns,
            include_vector,
        })
    }

    // -- catalog fixture for the end-to-end (analyze → plan) tests ----------

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

    fn docs_catalog() -> TestCatalog {
        TestCatalog::new().with("docs", docs_schema())
    }

    // -- unit: plan(bound) shape transform (Bound* built directly) ---------

    #[test]
    fn plans_select_projection_over_scan() {
        let bound = BoundStatement::Select(BoundSelect {
            from: "docs".to_string(),
            schema: docs_schema(),
            projection: vec![colref("author", 1), colref("title", 2)],
            include_vector: false,
        });
        assert_eq!(
            plan(bound),
            docs_project(vec![colref("author", 1), colref("title", 2)], false),
        );
    }

    #[test]
    fn plans_select_star_over_scan() {
        // Non-vector columns, include_vector false — carried from the binder.
        let bound = BoundStatement::Select(BoundSelect {
            from: "docs".to_string(),
            schema: docs_schema(),
            projection: vec![
                colref("author", 1),
                colref("title", 2),
                colref("published_at", 3),
            ],
            include_vector: false,
        });
        assert_eq!(
            plan(bound),
            docs_project(
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
    fn plans_select_with_explicit_vector_sets_include_vector() {
        let bound = BoundStatement::Select(BoundSelect {
            from: "docs".to_string(),
            schema: docs_schema(),
            projection: vec![colref("vector", 0)],
            include_vector: true,
        });
        assert_eq!(plan(bound), docs_project(vec![colref("vector", 0)], true),);
    }

    #[test]
    fn plans_search_projection_over_knn() {
        // A `SEARCH` plans to the SAME shape a `SELECT` does — `Project` over a
        // leaf — with `Knn` where `Scan` would be. That is the invariant the
        // compiler leans on: one read loop serves both, and only the leaf's
        // prologue differs.
        let bound = BoundStatement::Search(BoundSearch {
            from: "docs".to_string(),
            schema: docs_schema(),
            k: 5,
            query: vec![0.5; 768],
            projection: vec![colref("author", 1), colref("title", 2)],
        });
        assert_eq!(
            plan(bound),
            LogicalPlan::Project(Project {
                input: Box::new(LogicalPlan::Knn(Knn {
                    collection: "docs".to_string(),
                    schema: docs_schema(),
                    k: 5,
                    query: vec![0.5; 768],
                })),
                columns: vec![colref("author", 1), colref("title", 2)],
                // Bare SEARCH never returns the embedding — the same rule
                // `SELECT *` follows.
                include_vector: false,
            })
        );
    }

    #[test]
    fn plans_search_carries_k_and_query_unchanged() {
        // The planner only changes SHAPE. `k` and the query vector are carried
        // through byte for byte — the binder already validated both, and a
        // planner that rounded, truncated, or re-derived either would be
        // silently changing what the query means.
        for (k, query) in [
            (1u64, vec![0.5f32; 768]),
            (1000, vec![-0.25; 768]),
            (7, vec![0.0; 768]),
        ] {
            let bound = BoundStatement::Search(BoundSearch {
                from: "docs".to_string(),
                schema: docs_schema(),
                k,
                query: query.clone(),
                projection: vec![colref("author", 1)],
            });
            match plan(bound) {
                LogicalPlan::Project(p) => match *p.input {
                    LogicalPlan::Knn(knn) => {
                        assert_eq!(knn.k, k);
                        assert_eq!(knn.query, query);
                        assert_eq!(knn.collection, "docs");
                    }
                    other => panic!("expected Knn under the Project, got {other:?}"),
                },
                other => panic!("expected Project, got {other:?}"),
            }
        }
    }

    #[test]
    fn plans_insert_passthrough() {
        // The planner carries collection/schema/row through unchanged; the row
        // is already in schema order from the binder.
        let bound = BoundStatement::Insert(BoundInsert {
            collection: "docs".to_string(),
            schema: docs_schema(),
            row: bootstrap_row(),
        });
        assert_eq!(
            plan(bound),
            LogicalPlan::Insert(Insert {
                collection: "docs".to_string(),
                schema: docs_schema(),
                row: bootstrap_row(),
            }),
        );
    }

    #[test]
    fn plans_create_passthrough() {
        let bound = BoundStatement::CreateCollection(BoundCreate {
            name: "docs".to_string(),
            schema: docs_schema(),
            capacity: 1_000_000,
        });
        assert_eq!(
            plan(bound),
            LogicalPlan::CreateCollection(CreateCollection {
                name: "docs".to_string(),
                schema: docs_schema(),
                capacity: 1_000_000,
            }),
        );
    }

    // -- integration: analyze() then plan() for the bootstrap statements ---

    #[test]
    fn end_to_end_select() {
        let cat = docs_catalog();
        let bound = analyze(
            parse("SELECT author, title FROM docs;").expect("parse"),
            &cat,
        )
        .expect("bind");
        assert_eq!(
            plan(bound),
            docs_project(vec![colref("author", 1), colref("title", 2)], false),
        );
    }

    /// A query vector matching `docs_schema`'s `VECTOR(768)`, plus its SQL
    /// spelling. Every element is an exact binary fraction, so the literal
    /// round-trips through the parser's `f64 -> f32` coercion bit for bit and
    /// the comparison below is not a float-tolerance question.
    fn query_768() -> (Vec<f32>, String) {
        let mut query = vec![0.5f32; 768];
        query[0] = 0.25;
        query[767] = -0.125;
        let literal = query
            .iter()
            .map(|x| format!("{x:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        (query, literal)
    }

    #[test]
    fn end_to_end_search() {
        let cat = docs_catalog();
        let (query, literal) = query_768();
        let bound = analyze(
            parse(&format!("SEARCH TOP 5 NEAREST TO [{literal}] FROM docs;")).expect("parse"),
            &cat,
        )
        .expect("bind");
        assert_eq!(
            plan(bound),
            LogicalPlan::Project(Project {
                input: Box::new(LogicalPlan::Knn(Knn {
                    collection: "docs".to_string(),
                    schema: docs_schema(),
                    k: 5,
                    query,
                })),
                // Bare SEARCH has no projection clause, so the binder supplies
                // every scalar in schema order — `SELECT *`'s expansion.
                columns: vec![
                    colref("author", 1),
                    colref("title", 2),
                    colref("published_at", 3),
                ],
                include_vector: false,
            })
        );
    }

    #[test]
    fn end_to_end_insert() {
        let cat = docs_catalog();
        let src = format!(
            "INSERT INTO docs (vector, author, title, published_at) \
             VALUES ({}, 'alice', 'My doc', 1700000000);",
            vec_lit(768)
        );
        let bound = analyze(parse(&src).expect("parse"), &cat).expect("bind");
        assert_eq!(
            plan(bound),
            LogicalPlan::Insert(Insert {
                collection: "docs".to_string(),
                schema: docs_schema(),
                row: bootstrap_row(),
            }),
        );
    }

    #[test]
    fn end_to_end_create() {
        // Empty catalog so `docs` is new (analyze would otherwise reject it).
        let cat = TestCatalog::new();
        let src = "CREATE COLLECTION docs (
            vector VECTOR(768),
            author TEXT,
            title TEXT,
            published_at INT
        ) WITH (capacity = 1000000);";
        let bound = analyze(parse(src).expect("parse"), &cat).expect("bind");
        assert_eq!(
            plan(bound),
            LogicalPlan::CreateCollection(CreateCollection {
                name: "docs".to_string(),
                schema: docs_schema(),
                capacity: 1_000_000,
            }),
        );
    }
}
