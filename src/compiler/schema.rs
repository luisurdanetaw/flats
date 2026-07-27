//! Schema lowering: the binder's vocabulary → storage's.
//!
//! The two schema types describe the same collection in different numbering
//! systems, and this is the forward bridge between them:
//!
//! | | [`bind::Schema`](crate::sql::bind::Schema) | [`metadata::Schema`](crate::metadata::common::Schema) |
//! |---|---|---|
//! | vector | a real, ordinal-bearing column | separate; has NO `ColumnId` |
//! | numbering | declaration ordinals (vector-inclusive) | scalar-only, dense `ColumnId`s |
//! | types | syntactic [`ast::ColumnType`](crate::sql::ast::ColumnType) | storage [`ColumnType`](crate::metadata::common::ColumnType) |
//!
//! The reverse (`engine::bind_schema`) already existed, because
//! `impl Catalog for Db` has to answer the binder from a persisted schema. This
//! direction is what `CREATE COLLECTION` needs: there is no catalog entry to
//! look up yet, so the storage schema can only come from the DDL itself.
//!
//! # Order is the whole thing
//!
//! [`Schema::from_columns`](crate::metadata::common::Schema::from_columns)
//! derives BOTH numberings from the order of the `ColumnSpec`s it is handed:
//! the declaration ordinal is a column's index in that list, and the scalar
//! `ColumnId` is its index among the scalars only. Hand it the same columns in
//! a different order and every id changes. So this function walks the bound
//! columns in declaration order and never sorts, dedups, or reorders — it is a
//! transcription, and the ordering is the payload.

use std::fmt;
use std::num::NonZeroUsize;

use crate::metadata::common::{ColumnSpec, ColumnType as MetaType, Schema as MetaSchema};
use crate::sql::ast::ColumnType as AstType;
use crate::sql::bind::Schema as BindSchema;

/// Lower a bound schema to its storage form.
///
/// Walks `bound.columns` in declaration order, mapping each to a
/// [`ColumnSpec`], then hands the list to
/// [`Schema::from_columns`](MetaSchema::from_columns) — which is what assigns
/// the ordinals and `ColumnId`s, and which enforces the storage invariants
/// (exactly one vector, no duplicate names) so an invalid schema is
/// unconstructable.
pub fn to_metadata_schema(bound: &BindSchema) -> Result<MetaSchema, SchemaError> {
    let mut specs = Vec::with_capacity(bound.columns.len());
    for (position, column) in bound.columns.iter().enumerate() {
        // `bind::Schema` documents `columns[i].ordinal == i`. Everything below
        // reads ordinals off POSITION, so a violated invariant would silently
        // renumber the collection — check it rather than trust it.
        if column.ordinal != position {
            return Err(SchemaError::OrdinalMismatch {
                column: column.name.clone(),
                position,
                ordinal: column.ordinal,
            });
        }
        // `ty` is authoritative; the sibling `is_vector` flag is a documented
        // convenience mirror of it, so matching here cannot disagree with it.
        // No wildcard arm: a new syntactic type must be handled explicitly or
        // this stops compiling, which is the point.
        let spec = match column.ty {
            AstType::Vector(dim) => ColumnSpec::Vector {
                name: column.name.clone(),
                // The one type that can fail to lower: storage encodes "a
                // vector has at least one dimension" in the type itself.
                dim: NonZeroUsize::new(dim).ok_or_else(|| SchemaError::ZeroDimension {
                    column: column.name.clone(),
                })?,
            },
            AstType::Int => ColumnSpec::Scalar {
                name: column.name.clone(),
                ty: MetaType::Int,
            },
            AstType::Float => ColumnSpec::Scalar {
                name: column.name.clone(),
                ty: MetaType::Float,
            },
            AstType::Text => ColumnSpec::Scalar {
                name: column.name.clone(),
                ty: MetaType::Text,
            },
        };
        specs.push(spec);
    }
    MetaSchema::from_columns(specs).map_err(SchemaError::Storage)
}

/// Why a bound schema could not be lowered to its storage form.
///
/// A stage-local error type (CLAUDE.md §5): lowering is a query-pipeline
/// concern, so it does not widen the engine's `Error` enum — it wraps one at
/// the boundary instead.
//
// No PartialEq: the `Storage` variant carries `crate::Error`, which holds an
// `io::Error`. Tests match on the variant.
#[derive(Debug)]
pub enum SchemaError {
    /// `VECTOR(0)` — storage requires at least one dimension.
    ZeroDimension {
        /// The offending column's name.
        column: String,
    },
    /// A bound column's `ordinal` did not equal its index, breaking
    /// `bind::Schema`'s documented invariant. A binder/planner bug, not a user
    /// error.
    OrdinalMismatch {
        /// The offending column's name.
        column: String,
        /// Its index in `columns`.
        position: usize,
        /// The ordinal it claimed instead.
        ordinal: usize,
    },
    /// A storage invariant rejected the schema — duplicate column names, or not
    /// exactly one vector column.
    //
    // NOTE: the binder checks the one-vector rule itself (rule C), so that half
    // is unreachable from a bound plan. It does NOT yet check duplicate names,
    // so `CREATE COLLECTION c (a INT, a TEXT, v VECTOR(4))` binds cleanly and
    // fails HERE. Correct, but late: the check belongs in the binder as a
    // `BindError::DuplicateColumn` with source position. Until then this is the
    // backstop that keeps a corrupt schema from ever being constructed.
    Storage(crate::Error),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::ZeroDimension { column } => {
                write!(
                    f,
                    "column {column:?} declares VECTOR(0); dimension must be >= 1"
                )
            }
            SchemaError::OrdinalMismatch {
                column,
                position,
                ordinal,
            } => write!(
                f,
                "bound column {column:?} at position {position} claims ordinal {ordinal}"
            ),
            SchemaError::Storage(e) => write!(f, "invalid schema: {e}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchemaError::Storage(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SchemaError, to_metadata_schema};
    use crate::engine::bind_schema;
    use crate::metadata::common::{ColumnLocation, DeclarationOrdinal};
    use crate::sql::ast::ColumnType as AstType;
    use crate::sql::bind::{ColumnSchema, Schema as BindSchema};

    /// Build a bound schema from `(name, type)` pairs in DECLARATION ORDER,
    /// assigning ordinals positionally — exactly what `bind_create` does.
    fn bound(columns: &[(&str, AstType)]) -> BindSchema {
        BindSchema {
            columns: columns
                .iter()
                .enumerate()
                .map(|(ordinal, (name, ty))| ColumnSchema {
                    name: (*name).to_string(),
                    ty: *ty,
                    ordinal,
                    is_vector: matches!(ty, AstType::Vector(_)),
                })
                .collect(),
        }
    }

    fn ord(n: usize) -> DeclarationOrdinal {
        DeclarationOrdinal::new(n)
    }

    #[test]
    fn roundtrip_bind_meta_bind() {
        // The vector sits in the MIDDLE, where the two numbering systems are
        // maximally out of step — a round-trip that only works with the vector
        // first would prove nothing.
        let original = bound(&[
            ("author", AstType::Text),
            ("vector", AstType::Vector(768)),
            ("published_at", AstType::Int),
            ("score", AstType::Float),
        ]);

        let storage = to_metadata_schema(&original).expect("lowers");
        let back = bind_schema(&storage);

        assert_eq!(
            back, original,
            "bind -> metadata -> bind must be the identity: same columns, same types, same order"
        );
    }

    #[test]
    fn column_order_fixes_ordinals_and_ids() {
        // Vector FIRST: the scalars' ColumnIds run one behind their ordinals.
        let a = to_metadata_schema(&bound(&[
            ("vector", AstType::Vector(4)),
            ("x", AstType::Int),
            ("y", AstType::Int),
        ]))
        .expect("lowers");

        assert_eq!(a.vector.ordinal, ord(0));
        assert_eq!(
            a.locate(ord(0)),
            Some(ColumnLocation::Vector { dim: a.vector.dim })
        );
        assert_eq!(a.locate(ord(1)), Some(ColumnLocation::Scalar(0)), "x");
        assert_eq!(a.locate(ord(2)), Some(ColumnLocation::Scalar(1)), "y");
        assert_eq!(a.by_name["x"], 0);
        assert_eq!(a.by_name["y"], 1);

        // SAME columns, SAME names, different declaration order. Every id moves
        // with the order — proving position is the source of truth, not the
        // name and not the type.
        let b = to_metadata_schema(&bound(&[
            ("y", AstType::Int),
            ("x", AstType::Int),
            ("vector", AstType::Vector(4)),
        ]))
        .expect("lowers");

        assert_eq!(b.vector.ordinal, ord(2), "the vector moved to the end");
        assert_eq!(
            b.locate(ord(0)),
            Some(ColumnLocation::Scalar(0)),
            "y is now first"
        );
        assert_eq!(
            b.locate(ord(1)),
            Some(ColumnLocation::Scalar(1)),
            "x is now second"
        );
        assert_eq!(b.by_name["y"], 0, "y took x's old id");
        assert_eq!(b.by_name["x"], 1, "x took y's old id");
    }

    #[test]
    fn type_mapping_is_exhaustive() {
        use crate::metadata::common::ColumnType as MetaType;

        // One case per syntactic type. The REAL guard against a new type
        // slipping through is that `to_metadata_schema`'s match has no wildcard
        // arm — adding an `ast::ColumnType` variant breaks the build. These
        // cases pin that each existing arm maps to the right storage type.
        let s = to_metadata_schema(&bound(&[
            ("i", AstType::Int),
            ("f", AstType::Float),
            ("t", AstType::Text),
            ("v", AstType::Vector(16)),
        ]))
        .expect("lowers");

        let by_id = |id: u32| &s.columns[id as usize];
        assert_eq!(by_id(s.by_name["i"]).ty, MetaType::Int);
        assert_eq!(by_id(s.by_name["f"]).ty, MetaType::Float);
        assert_eq!(by_id(s.by_name["t"]).ty, MetaType::Text);
        // The vector is NOT among the scalar columns — it has no ColumnId.
        assert_eq!(s.columns.len(), 3, "only scalars get ColumnIds");
        assert_eq!(s.vector.name, "v");
    }

    #[test]
    fn vector_column_dimension_preserved() {
        for dim in [1usize, 4, 768, 4096] {
            let s = to_metadata_schema(&bound(&[
                ("a", AstType::Int),
                ("embedding", AstType::Vector(dim)),
            ]))
            .expect("lowers");

            assert_eq!(s.vector.name, "embedding");
            assert_eq!(s.vector.dim.get(), dim, "dimension survives lowering");
            assert_eq!(s.vector.ordinal, ord(1));
            // …and it is reachable through `locate`, carrying the same dim —
            // the path the compiler uses to decide `VectorFetch` vs `Column`.
            assert_eq!(
                s.locate(ord(1)),
                Some(ColumnLocation::Vector { dim: s.vector.dim })
            );
        }
    }

    // ---- the failure modes reachable from a validly-BOUND schema -----------

    #[test]
    fn zero_dimension_is_rejected() {
        // The binder's rule (C) counts vector columns but never checks the
        // dimension, so `VECTOR(0)` binds cleanly and must be caught here.
        let err = to_metadata_schema(&bound(&[("v", AstType::Vector(0))])).unwrap_err();
        assert!(matches!(err, SchemaError::ZeroDimension { column } if column == "v"));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        // Likewise: the binder has no DuplicateColumn check yet, so this is the
        // backstop that stops a schema whose `by_name` map would silently lose
        // a column from ever being built.
        let err = to_metadata_schema(&bound(&[
            ("v", AstType::Vector(4)),
            ("a", AstType::Int),
            ("a", AstType::Text),
        ]))
        .unwrap_err();
        assert!(matches!(
            err,
            SchemaError::Storage(crate::Error::DuplicateColumn(ref name)) if name == "a"
        ));
    }

    #[test]
    fn non_positional_ordinals_are_rejected() {
        // A hand-built schema violating `columns[i].ordinal == i`. Renumbering
        // it silently would hand the collection a different ColumnId space than
        // the plan's ordinals refer to.
        let mut broken = bound(&[("v", AstType::Vector(4)), ("a", AstType::Int)]);
        broken.columns[1].ordinal = 7;
        let err = to_metadata_schema(&broken).unwrap_err();
        assert!(matches!(
            err,
            SchemaError::OrdinalMismatch {
                position: 1,
                ordinal: 7,
                ..
            }
        ));
    }
}
