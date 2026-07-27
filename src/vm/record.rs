//! Record splitting: one packed row → the two things
//! [`Db::insert`](crate::engine::Db::insert) wants.
//!
//! `Op::MakeRecord` packs a register run into ONE record, and `Op::Insert` hands
//! that single record to the engine. But
//! [`Db::insert`](crate::engine::Db::insert) takes the embedding and the
//! metadata row as SEPARATE arguments — because they go to separate stores. This
//! module is the seam between those two shapes.
//!
//! # Why the split cannot be positional
//!
//! A packed record is in DECLARATION ORDER: one literal per column, vector
//! included, exactly as the binder canonicalized it (see
//! [`plan::Insert::row`](crate::sql::plan::Insert)). Storage numbers the same
//! columns differently — scalars get a dense, vector-EXCLUDING
//! [`ColumnId`](crate::metadata::common::ColumnId), and the embedding gets none
//! at all. Declare the vector anywhere but last and the two numberings drift
//! apart:
//!
//! ```text
//!   CREATE COLLECTION docs (author TEXT, vector VECTOR(3), title TEXT)
//!
//!   packed position:   0        1         2          <- what the record has
//!   declaration ord:   0        1         2
//!   storage:      ColumnId 0  flat idx  ColumnId 1   <- what the engine wants
//!                                          ^^^^^^^^
//!                                  position 2, but ColumnId 1
//! ```
//!
//! So the split asks [`Schema::locate`] where each declaration ordinal actually
//! lives, and uses the `ColumnId` the schema STORED. It never counts scalars,
//! filters the vector out, or subtracts one — the same rule the compiler follows
//! for `Column` ops, for the same reason.
//!
//! # What this checks, and what it leaves to the engine
//!
//! STRUCTURE only — arity, which position is the embedding, and its dimension
//! (all three come straight from the schema's own answer). TYPES are
//! [`Schema::validate_row`](crate::metadata::common::Schema::validate_row)'s
//! job, called by `Db::insert` before anything is logged; duplicating the type
//! table here would give two places to keep in sync.
//!
//! `Db::insert` re-checks the dimension too. That is not redundant: its check
//! guards the DURABILITY boundary (a record that cannot apply must never reach
//! the WAL, or it fails on every replay forever), and it must hold for every
//! caller, not just this one. The check here exists to fail with the offending
//! record's *position* while that context still exists.

use std::fmt;

use crate::metadata::common::{ColumnLocation, DeclarationOrdinal, Row, Schema, Value};
use crate::sql::ast::Literal;
use crate::vm::value::ValueError;

/// One packed record — the payload `Op::MakeRecord` builds and `Op::Insert`
/// writes.
///
/// The values are in DECLARATION ORDER (vector included), one per column, which
/// is what makes position `i` mean `DeclarationOrdinal(i)` in
/// [`split_record`]. That ordering is inherited from the binder, which
/// canonicalizes a user's `(cols) VALUES (...)` list into schema order before
/// the compiler ever emits a register.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// One literal per column, in declaration order.
    pub values: Vec<Literal>,
}

/// Split `packed` into the `(vector, row)` pair
/// [`Db::insert`](crate::engine::Db::insert) takes.
///
/// Walks the record in declaration order, asking [`Schema::locate`] where each
/// position belongs: the one vector position becomes the embedding, every other
/// becomes a `(ColumnId, Value)` entry keyed by the id the SCHEMA assigned. See
/// the module header for why that indirection is mandatory.
///
/// The returned `Row` is in ascending `ColumnId` order — which is not
/// necessarily the record's order — though nothing downstream requires it
/// (`validate_row` accepts any order).
pub fn split_record(packed: &Record, schema: &Schema) -> Result<(Vec<f32>, Row), SplitError> {
    // Total declaration count = every scalar, plus the one vector column
    // `Schema::from_columns` guarantees exists (the vector is absent from
    // `columns` precisely because it has no `ColumnId`).
    let expected = schema.columns.len() + 1;
    if packed.values.len() != expected {
        return Err(SplitError::Arity {
            expected,
            got: packed.values.len(),
        });
    }

    let mut vector = None;
    let mut row: Row = Vec::with_capacity(schema.columns.len());

    for (position, literal) in packed.values.iter().enumerate() {
        match schema.locate(DeclarationOrdinal::new(position)) {
            Some(ColumnLocation::Vector { dim }) => {
                let embedding = match literal {
                    Literal::Vector(v) => v,
                    _ => return Err(SplitError::NotAVector { position }),
                };
                if embedding.len() != dim.get() {
                    return Err(SplitError::DimensionMismatch {
                        position,
                        expected: dim.get(),
                        got: embedding.len(),
                    });
                }
                vector = Some(embedding.clone());
            }
            Some(ColumnLocation::Scalar(id)) => {
                // The id comes from `locate`, never from `position`.
                let value =
                    Value::try_from(literal).map_err(|e| SplitError::NotAScalar { position, e })?;
                row.push((id, value));
            }
            // Arity already matched, so every position `0..expected` resolves —
            // unless the schema's declaration ordinals are not `0..N`, which
            // `from_columns` makes unconstructable.
            None => return Err(SplitError::UnknownOrdinal { position }),
        }
    }

    // Unreachable given the arity check plus "exactly one vector", but the
    // alternative is an `unwrap` in library code.
    let vector = vector.ok_or(SplitError::MissingVector)?;
    Ok((vector, row))
}

/// Why a packed record could not be split into `(vector, row)`.
///
/// A stage-local error type (CLAUDE.md §5): these are frontend-shaped failures —
/// every variant names the offending record POSITION, which is meaningless to
/// the engine and must not leak into [`crate::Error`].
///
/// Every variant is a "cannot happen if the binder did its job" case. They are
/// errors rather than panics because this is a boundary: the record comes from a
/// program, the schema from the catalog, and nothing in the type system pairs
/// the two.
#[derive(Debug, Clone, PartialEq)]
pub enum SplitError {
    /// The record has the wrong number of values for the schema.
    Arity {
        /// Columns the schema declares (scalars + the one vector).
        expected: usize,
        /// Values the record carries.
        got: usize,
    },
    /// The vector column's position holds something that is not a vector.
    NotAVector {
        /// Offending position in the record (its declaration ordinal).
        position: usize,
    },
    /// A scalar column's position holds a value that has no storage form (a
    /// vector literal — see [`ValueError`]).
    NotAScalar {
        /// Offending position in the record (its declaration ordinal).
        position: usize,
        /// Why the literal did not lower.
        e: ValueError,
    },
    /// The embedding is the wrong length for the schema's declared dimension.
    DimensionMismatch {
        /// The vector column's position in the record.
        position: usize,
        /// Dimension the schema declares.
        expected: usize,
        /// Length the record carries.
        got: usize,
    },
    /// A position does not resolve to any column in the schema.
    UnknownOrdinal {
        /// The unresolvable position.
        position: usize,
    },
    /// No position resolved to the vector column.
    MissingVector,
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitError::Arity { expected, got } => {
                write!(f, "record has {got} values, schema declares {expected}")
            }
            SplitError::NotAVector { position } => {
                write!(
                    f,
                    "value {position} is the vector column but is not a vector"
                )
            }
            SplitError::NotAScalar { position, e } => {
                write!(f, "value {position}: {e}")
            }
            SplitError::DimensionMismatch {
                position,
                expected,
                got,
            } => write!(
                f,
                "value {position}: vector has {got} dimensions, schema declares {expected}"
            ),
            SplitError::UnknownOrdinal { position } => {
                write!(
                    f,
                    "value {position} does not match any column in the schema"
                )
            }
            SplitError::MissingVector => write!(f, "record has no vector column"),
        }
    }
}

impl std::error::Error for SplitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SplitError::NotAScalar { e, .. } => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Record, SplitError, split_record};
    use crate::engine::{CollectionConfig, Db, DbOptions};
    use crate::metadata::common::{ColumnSpec, ColumnType, Row, Schema, Value};
    use crate::sql::ast::Literal;
    use crate::vm::value::ValueError;
    use std::num::NonZeroUsize;
    use std::time::Duration;

    const DIM: usize = 3;

    /// The KEY fixture: the vector sits in the MIDDLE, so declaration ordinals
    /// and storage `ColumnId`s are maximally out of step —
    ///
    ///   author@0 → ColumnId 0, vector@1 → flat index,
    ///   title@2  → ColumnId 1, published_at@3 → ColumnId 2
    ///
    /// A split that used the record position as the `ColumnId` would put `title`
    /// in `published_at`'s column and run off the end. Vector-first would hide
    /// that (every scalar would be off by a constant 1, still wrong but less
    /// visibly).
    fn schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(DIM).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "title".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "published_at".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap()
    }

    const EMBEDDING: [f32; DIM] = [0.5, -0.25, 0.125];

    /// What `Op::MakeRecord` would pack for
    /// `INSERT INTO docs (author, vector, title, published_at)
    ///  VALUES ('alice', [...], 'My doc', 1700000000)` — declaration order.
    fn packed() -> Record {
        Record {
            values: vec![
                Literal::Str("alice".into()),
                Literal::Vector(EMBEDDING.to_vec()),
                Literal::Str("My doc".into()),
                Literal::Int(1_700_000_000),
            ],
        }
    }

    fn opts() -> DbOptions {
        DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
        }
    }

    #[test]
    fn split_packed_record() {
        let schema = schema();
        let (vector, row) = split_record(&packed(), &schema).expect("a well-formed record splits");

        // The embedding comes out whole and unmodified.
        assert_eq!(vector, EMBEDDING.to_vec());

        // ...and every scalar is keyed by the id the SCHEMA assigned, not by its
        // position in the record: `title` is at position 2 but is ColumnId 1,
        // `published_at` at position 3 but ColumnId 2.
        let expected: Row = vec![
            (0, Value::Text("alice".into())),
            (1, Value::Text("My doc".into())),
            (2, Value::Int(1_700_000_000)),
        ];
        assert_eq!(row, expected);
        // The embedding is NOT also in the row — it lives in one store only.
        assert_eq!(row.len(), schema.columns.len());

        // The pair the split produced is exactly what `Db::insert` takes, and a
        // cursor reads the row back through the other two stores.
        let dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig {
            id: 0,
            name: "docs".into(),
            capacity: 16,
            schema: schema.clone(),
        };
        let db = Db::open(dir.path(), &[cfg], opts()).unwrap();

        let ordinal = db.insert(0, &vector, row).expect("the split row inserts");

        let mut cursor = db.scan(0).unwrap();
        assert!(cursor.seek_first().unwrap(), "the inserted row is live");
        assert_eq!(cursor.ordinal(), Some(ordinal));
        // `scan` projects every scalar in ColumnId order — so this is the
        // record's scalars, reordered by the same mapping the split applied.
        assert_eq!(
            cursor.row().expect("parked on a row"),
            &[
                Value::Text("alice".into()),
                Value::Text("My doc".into()),
                Value::Int(1_700_000_000),
            ]
        );
        assert!(!cursor.next().unwrap(), "exactly one row was inserted");

        // The embedding round-trips through the flat index, keyed by the same
        // ordinal — the two halves of the split rejoin.
        let reader = db.reader(0).expect("collection 0 exists");
        assert_eq!(reader.vector_at(ordinal), Some(&EMBEDDING[..]));

        db.close().unwrap();
    }

    // -- structural failures -------------------------------------------------
    //
    // Each is "cannot happen if the binder did its job", so each is a guard on a
    // boundary the type system does not cover.

    #[test]
    fn split_rejects_wrong_arity() {
        let schema = schema();

        let mut short = packed();
        short.values.pop();
        assert_eq!(
            split_record(&short, &schema),
            Err(SplitError::Arity {
                expected: 4,
                got: 3
            })
        );

        let mut long = packed();
        long.values.push(Literal::Int(1));
        assert_eq!(
            split_record(&long, &schema),
            Err(SplitError::Arity {
                expected: 4,
                got: 5
            })
        );
    }

    #[test]
    fn split_rejects_a_scalar_in_the_vector_position() {
        let mut record = packed();
        record.values[1] = Literal::Int(7);
        assert_eq!(
            split_record(&record, &schema()),
            Err(SplitError::NotAVector { position: 1 })
        );
    }

    #[test]
    fn split_rejects_a_vector_in_a_scalar_position() {
        // Position 2 is `title` (a TEXT column) — a vector there has nowhere to
        // go, since the collection has exactly one flat index and position 1
        // owns it.
        let mut record = packed();
        record.values[2] = Literal::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(
            split_record(&record, &schema()),
            Err(SplitError::NotAScalar {
                position: 2,
                e: ValueError::NotScalar,
            })
        );
    }

    #[test]
    fn split_rejects_a_wrong_dimension_embedding() {
        let mut record = packed();
        record.values[1] = Literal::Vector(vec![1.0, 2.0]);
        assert_eq!(
            split_record(&record, &schema()),
            Err(SplitError::DimensionMismatch {
                position: 1,
                expected: DIM,
                got: 2,
            })
        );
    }

    #[test]
    fn split_handles_a_vector_only_collection() {
        // No scalars at all: the row is empty and the record is one value long.
        // `Db::insert`'s documented vector-only case, reached through the split.
        let schema = Schema::from_columns(vec![ColumnSpec::Vector {
            name: "vector".into(),
            dim: NonZeroUsize::new(DIM).unwrap(),
        }])
        .unwrap();
        let record = Record {
            values: vec![Literal::Vector(EMBEDDING.to_vec())],
        };

        let (vector, row) = split_record(&record, &schema).expect("a lone vector splits");
        assert_eq!(vector, EMBEDDING.to_vec());
        assert!(row.is_empty());
    }

    #[test]
    fn split_handles_a_trailing_vector() {
        // The layout where position and ColumnId happen to AGREE for every
        // scalar. It must work too — the fixture above is chosen to catch
        // positional shortcuts, not because trailing vectors are unsupported.
        let schema = Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Scalar {
                name: "b".into(),
                ty: ColumnType::Float,
            },
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(DIM).unwrap(),
            },
        ])
        .unwrap();
        let record = Record {
            values: vec![
                Literal::Int(1),
                Literal::Float(2.5),
                Literal::Vector(EMBEDDING.to_vec()),
            ],
        };

        let (vector, row) = split_record(&record, &schema).expect("a trailing vector splits");
        assert_eq!(vector, EMBEDDING.to_vec());
        assert_eq!(row, vec![(0, Value::Int(1)), (1, Value::Float(2.5))]);
    }
}
