//! common.rs — shared primitives for Phases 4a/4b/4c.
//!
//! Everything here is deliberately dumb: plain data types with no behavior.
//! Both the MetadataIndex and TupleStore speak in these types, and Phase 4c
//! (the WAL applier) carries them inside Record::Insert. Keeping them in one
//! module is what makes "integrates with FlatIndex's ordinal space implicitly"
//! true — there is exactly ONE Ordinal type in the whole engine.
//!
//! REFACTOR NOTE (read before adding types here): `Ordinal` and `Lsn` already
//! exist as newtypes in `index::index` and `wal::wal`. We RE-EXPORT them
//! rather than declaring parallel aliases — two structurally-identical types
//! with the same name is exactly the bug class this module exists to prevent.
//! Dependency direction stays clean: this module pulls plain newtypes from
//! index/wal; neither of them imports anything from metadata (until Phase 4c,
//! where `Record::Insert` starts carrying `Value` — see docs/phase-4c).

use std::collections::HashMap;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Row identity within a collection — re-exported from the flat index so the
/// metadata index, tuple store, and vector index all share ONE ordinal space.
///
/// NOTE the width mismatch you will meet in Phase 4c: `Record::Insert` carries
/// `ordinal: u64`, but `Ordinal` wraps u32 (RoaringBitmap is a set of u32s, so
/// the metadata ordinal space is u32 by construction). The applier converts
/// with a checked `try_into` — capacity bounds mean overflow is a logic bug,
/// but it must surface as an error, not a truncation.
pub use crate::index::index::Ordinal;

/// Log sequence number — re-exported from the WAL, the one place LSNs are
/// minted. Monotonic, 1-based (0 = "nothing durable yet").
pub use crate::wal::wal::Lsn;

/// Stable identifier for a column within a collection's schema.
/// Using an id instead of a String on the hot path avoids hashing strings
/// on every insert. The schema owns the name→id mapping.
///
/// This is the SCALAR-ONLY numbering (dense `0..N`, vector excluded) that the
/// tuple store and metadata index address by — distinct from a
/// [`DeclarationOrdinal`], which is vector-inclusive.
pub type ColumnId = u32;

/// A column's position in the VECTOR-INCLUSIVE declaration numbering — the
/// numbering the query binder/plan speak, where the vector is "just a column."
///
/// A distinct newtype (not a bare `usize`) so a declaration ordinal can never be
/// silently passed where a scalar [`ColumnId`] is expected, or vice versa: the
/// two ordinal spaces do not interconvert by positional arithmetic (a vector
/// anywhere but last shifts them apart), so the type system must keep them
/// apart. Serde-transparent: a newtype struct serializes as its inner `usize`,
/// so persisting these fields is wire-identical to the old raw `usize` — no
/// format bump.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct DeclarationOrdinal(usize);

impl DeclarationOrdinal {
    /// Wrap a raw declaration-order index (e.g. a plan-supplied ordinal).
    pub fn new(ordinal: usize) -> Self {
        DeclarationOrdinal(ordinal)
    }

    /// The underlying declaration-order index.
    pub fn get(self) -> usize {
        self.0
    }
}

/// The three primitive column types Phase 4 supports.
/// NULL, arrays, etc. are explicit non-goals.
//
// Serde: these derive Serialize/Deserialize because (a) the snapshot files
// persist the schema, and (b) in Phase 4c `Record::Insert` carries values
// inside the bincode-encoded WAL frame. Same wire-format caveat as `Record`:
// bincode tags enum variants POSITIONALLY — append variants, never reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    Int,
    Float,
    Text,
}

/// A single metadata value. The Float variant may NEVER hold NaN once it is
/// inside the index — reject at the insert boundary (Phase 4a rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Int(i64),
    Float(f64),
    Text(String),
}

impl Value {
    /// Return the ColumnType this value belongs to.
    /// Used for schema validation on insert (type mismatch → Err).
    pub fn column_type(&self) -> ColumnType {
        match self {
            Value::Int(_) => ColumnType::Int,
            Value::Float(_) => ColumnType::Float,
            Value::Text(_) => ColumnType::Text,
        }
    }
}

/// A full metadata row: one value per column, keyed by ColumnId.
/// Phase 4 requires every column present (no NULLs) — see
/// [`Schema::validate_row`].
pub type Row = Vec<(ColumnId, Value)>;

/// Range comparison operators for lookup_range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// How a single collection is configured. Lives here (not in the engine)
/// because it is plain data that BOTH the catalog file and — since Phase 6 —
/// `Record::CreateCollection` carry; the WAL may only depend on this module.
//
// Serde: persisted inside catalog.snap AND inside the bincode-encoded WAL
// frame. Field order is part of the wire format (bincode is positional) —
// append fields, never reorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Stable numeric identity — what WAL records key on. Assigned once
    /// (open-time registration or `create_collection`) and never reused.
    pub id: u32,
    /// Human-facing name — what V-SQL will key on. Unique across the catalog.
    pub name: String,
    pub capacity: usize,
    /// The full schema — the single vector column (name + declaration ordinal +
    /// dim) AND every scalar. The vector's dim lives here (`schema.vector().dim`),
    /// the single source of truth; there is no separate `dim` field. Every
    /// insert's row is validated against the scalar columns BEFORE logging.
    pub schema: Schema,
}

/// The collection schema. Created once at collection-create time; Phase 4
/// treats it as immutable (schema evolution is a non-goal).
//
// PartialEq is derived so open() can check "snapshot schema == caller schema"
// (Error::SchemaMismatch). Serde is derived because both snapshot files
// persist the schema. `by_name` is redundant with `columns` — an alternative
// is to skip it (`#[serde(skip)]`) and rebuild it on deserialize; persisting
// it is simpler and the schema is tiny, so we just persist it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// The SCALAR columns. `ColumnId` is the index into this Vec — i.e.
    /// columns[3].id == 3 — the scalar-only numbering the tuple store and
    /// metadata index address by. The vector is NOT here (it has no `ColumnId`);
    /// it lives in `vector`.
    pub columns: Vec<ColumnDef>,
    /// The single vector column: name + declaration ordinal + dim. Every
    /// collection has exactly one (one flat index), guaranteed at construction.
    pub vector: VectorColumn,
    /// scalar name → ColumnId, built from `columns` at construction time.
    pub by_name: HashMap<String, ColumnId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Scalar-only ordinal — the tuple store / metadata index address (dense,
    /// 0-based, in scalar declaration order). Its index in [`Schema::columns`].
    pub id: ColumnId,
    pub name: String,
    pub ty: ColumnType,
    /// Declaration ordinal in the VECTOR-INCLUSIVE numbering — this column's
    /// position among ALL columns (vector included), the numbering the query
    /// binder/plan use. ADDITIONAL to `id`, never a replacement.
    pub ordinal: DeclarationOrdinal,
}

/// The single vector column's persisted layout. The embedding lives in the flat
/// vector index — not the tuple store / metadata index — so the vector has NO
/// `ColumnId` (the scalar-only numbering those stores address by). It still
/// records a NAME and a DECLARATION ORDINAL, so the vector-inclusive schema the
/// query binder needs is reconstructable from disk alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorColumn {
    /// Column name (e.g. `vector`).
    pub name: String,
    /// Declaration ordinal in the VECTOR-INCLUSIVE numbering.
    pub ordinal: DeclarationOrdinal,
    /// Embedding dimension (≥ 1 by construction).
    pub dim: NonZeroUsize,
}

/// One column of a [`Schema::from_columns`] declaration, in source order. A
/// column's declaration ordinal is simply its index in the list passed to the
/// constructor — the vector is "just a column that happens to be a vector."
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnSpec {
    /// A scalar column (gets a dense scalar `ColumnId`).
    Scalar {
        /// Column name.
        name: String,
        /// Scalar type.
        ty: ColumnType,
    },
    /// The vector column (gets NO `ColumnId`; the embedding is in the flat index).
    Vector {
        /// Column name.
        name: String,
        /// Embedding dimension.
        dim: NonZeroUsize,
    },
}

/// Where a [`DeclarationOrdinal`] resolves to in physical storage — the single,
/// indivisible answer to "is this column the vector, and if not, which
/// [`ColumnId`]?". Returned by [`Schema::locate`].
///
/// One enum (rather than paired `is_vector` / `column_id` accessors) so a caller
/// cannot check "is it the vector?" and forget "which ColumnId?" — the match
/// forces both to be handled together. The two arms are the two disjoint
/// storage locations: scalars live in the tuple store / metadata index (by
/// `ColumnId`); the embedding lives in the flat vector index (by `dim`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnLocation {
    /// A scalar column, addressed by this `ColumnId` in the tuple store and
    /// metadata index. Read straight from the schema — never recomputed.
    Scalar(ColumnId),
    /// The vector column, whose embedding lives in the flat vector index. Carries
    /// the dimension from the total [`Schema::vector`].
    Vector {
        /// Embedding dimension (≥ 1 by construction).
        dim: NonZeroUsize,
    },
}

impl Schema {
    /// Build a schema from its columns in DECLARATION ORDER. Derives both
    /// numberings in a single pass:
    ///
    ///  * **declaration ordinal** = the column's index in `cols`
    ///    (vector-inclusive; the numbering the query binder/plan use);
    ///  * **scalar `ColumnId`** = a dense counter that advances ONLY on scalar
    ///    columns, so scalars stay contiguous `0..N` and the vector receives no
    ///    `ColumnId` (the tuple store / metadata index numbering is unperturbed).
    ///
    /// The storage invariant is enforced HERE, so an invalid schema is
    /// unconstructable: exactly one vector column (else
    /// [`Error::VectorColumnCount`]) and no duplicate names (else
    /// [`Error::DuplicateColumn`]). `dim ≥ 1` is already guaranteed by
    /// `NonZeroUsize` in [`ColumnSpec::Vector`].
    pub fn from_columns(cols: Vec<ColumnSpec>) -> Result<Self> {
        let vector_count = cols
            .iter()
            .filter(|c| matches!(c, ColumnSpec::Vector { .. }))
            .count();
        if vector_count != 1 {
            return Err(Error::VectorColumnCount { found: vector_count });
        }

        let mut columns = Vec::new();
        let mut by_name = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        let mut vector = None;
        for (ordinal, spec) in cols.into_iter().enumerate() {
            let ordinal = DeclarationOrdinal::new(ordinal);
            match spec {
                ColumnSpec::Scalar { name, ty } => {
                    if !seen.insert(name.clone()) {
                        return Err(Error::DuplicateColumn(name));
                    }
                    // ColumnId counts scalars only → contiguous, vector excluded.
                    let id = columns.len() as ColumnId;
                    by_name.insert(name.clone(), id);
                    columns.push(ColumnDef {
                        id,
                        name,
                        ty,
                        ordinal,
                    });
                }
                ColumnSpec::Vector { name, dim } => {
                    if !seen.insert(name.clone()) {
                        return Err(Error::DuplicateColumn(name));
                    }
                    vector = Some(VectorColumn { name, ordinal, dim });
                }
            }
        }
        // `vector_count == 1` above guarantees this is Some.
        let vector = vector.ok_or(Error::VectorColumnCount { found: 0 })?;
        Ok(Schema {
            columns,
            vector,
            by_name,
        })
    }

    /// The single vector column. Total — construction guaranteed exactly one
    /// exists, so callers (e.g. the flat index reading `.dim`) never handle a
    /// missing case.
    pub fn vector(&self) -> &VectorColumn {
        &self.vector
    }

    /// Re-check the no-duplicate-names invariant on a schema that may have been
    /// hand-assembled (the fields are public). [`from_columns`](Self::from_columns)
    /// already guarantees it; `create_collection` calls this to reject a caller
    /// that bypassed the constructor.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        seen.insert(self.vector.name.as_str());
        for def in &self.columns {
            if !seen.insert(def.name.as_str()) {
                return Err(Error::DuplicateColumn(def.name.clone()));
            }
        }
        Ok(())
    }

    pub fn column(&self, id: ColumnId) -> Option<&ColumnDef> {
        // columns[i].id == i by construction, so the id IS the index.
        self.columns.get(id as usize)
    }

    /// Resolve a PLAN DECLARATION ORDINAL to its physical storage location.
    ///
    /// This is the compiler's one lookup from the vector-inclusive numbering the
    /// plan speaks to the physical addresses the engine speaks: a scalar's stored
    /// [`ColumnId`] (tuple store / metadata index) or the vector's `dim` (flat
    /// index). It only READS what [`from_columns`](Self::from_columns) already
    /// assigned — it does no counting, filtering, or positional arithmetic to
    /// derive a `ColumnId` (the vector may sit anywhere, so the two spaces do not
    /// interconvert positionally; the stored `id` is the sole source).
    ///
    /// Total over the valid domain: every declaration ordinal `0..N` resolves to
    /// exactly one column. An out-of-range or otherwise unknown ordinal returns
    /// `None` rather than panicking — the compiler feeds plan-supplied ordinals
    /// across a boundary and maps the `None` to its own error.
    pub fn locate(&self, ordinal: DeclarationOrdinal) -> Option<ColumnLocation> {
        if self.vector.ordinal == ordinal {
            return Some(ColumnLocation::Vector {
                dim: self.vector.dim,
            });
        }
        self.columns
            .iter()
            .find(|def| def.ordinal == ordinal)
            .map(|def| ColumnLocation::Scalar(def.id))
    }

    /// Validate a full row against this schema. Shared by MetadataIndex
    /// (insert_row) and TupleStore (write_row) so the two stores can never
    /// disagree about what a well-formed row is.
    ///
    /// Rules: length matches the schema (IncompleteRow), every id known
    /// (UnknownColumn), types match (TypeMismatch), no duplicate ids
    /// (IncompleteRow), no NaN floats (NaNRejected). Length + known ids +
    /// no duplicates together prove every column appears exactly once.
    ///
    /// IMPORTANT for callers: validate BEFORE mutating any state, so a bad row
    /// can never leave a half-inserted ordinal behind.
    pub fn validate_row(&self, row: &Row) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(Error::IncompleteRow);
        }
        let mut seen = vec![false; self.columns.len()];
        for (id, value) in row {
            let def = self
                .column(*id)
                .ok_or(Error::UnknownColumn { column: *id })?;
            let got = value.column_type();
            if got != def.ty {
                return Err(Error::TypeMismatch {
                    column: *id,
                    expected: def.ty,
                    got,
                });
            }
            if seen[*id as usize] {
                return Err(Error::IncompleteRow);
            }
            seen[*id as usize] = true;
            if let Value::Float(f) = value {
                if f.is_nan() {
                    return Err(Error::NaNRejected { column: *id });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "v".into(),
                dim: NonZeroUsize::new(4).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Scalar {
                name: "b".into(),
                ty: ColumnType::Float,
            },
            ColumnSpec::Scalar {
                name: "c".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap()
    }

    #[test]
    fn new_assigns_ids_in_declaration_order() {
        let s = schema();
        assert_eq!(s.columns.len(), 3);
        // Scalar ColumnId space unchanged: dense 0..N in scalar declaration order.
        for (i, def) in s.columns.iter().enumerate() {
            assert_eq!(def.id, i as ColumnId);
        }
        assert_eq!(s.by_name["b"], 1);
        assert_eq!(s.column(2).unwrap().name, "c");
        assert!(s.column(3).is_none());
        // Declaration ordinals are vector-inclusive: vector@0 shifts scalars to 1..
        assert_eq!(s.vector().name, "v");
        assert_eq!(s.vector().ordinal.get(), 0);
        assert_eq!(s.column(0).unwrap().ordinal.get(), 1); // a
        assert_eq!(s.column(1).unwrap().ordinal.get(), 2); // b
        assert_eq!(s.column(2).unwrap().ordinal.get(), 3); // c
    }

    #[test]
    fn new_rejects_duplicate_names() {
        let err = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "v".into(),
                dim: NonZeroUsize::new(2).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap_err();
        assert!(matches!(err, Error::DuplicateColumn(name) if name == "a"));
    }

    #[test]
    fn from_columns_requires_exactly_one_vector() {
        assert!(matches!(
            Schema::from_columns(vec![ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            }]),
            Err(Error::VectorColumnCount { found: 0 })
        ));
        assert!(matches!(
            Schema::from_columns(vec![
                ColumnSpec::Vector {
                    name: "u".into(),
                    dim: NonZeroUsize::new(2).unwrap(),
                },
                ColumnSpec::Vector {
                    name: "w".into(),
                    dim: NonZeroUsize::new(2).unwrap(),
                },
            ]),
            Err(Error::VectorColumnCount { found: 2 })
        ));
    }

    #[test]
    fn vector_after_scalars_shifts_only_later_ordinals() {
        // [author, vector, title]: scalar ColumnIds stay 0,1; declaration
        // ordinals are author@0, vector@1, title@2.
        let s = Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(8).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "title".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap();
        assert_eq!(s.vector().ordinal.get(), 1);
        assert_eq!(s.column(0).unwrap().name, "author");
        assert_eq!(s.column(0).unwrap().ordinal.get(), 0);
        assert_eq!(s.column(1).unwrap().name, "title");
        assert_eq!(s.column(1).unwrap().ordinal.get(), 2);
    }

    // -- locate(): plan declaration ordinal -> physical storage location ----

    fn ord(n: usize) -> DeclarationOrdinal {
        DeclarationOrdinal::new(n)
    }

    #[test]
    fn locate_maps_each_declaration_ordinal_to_its_stored_location() {
        // schema(): v@0 (dim 4), a@1 (id 0), b@2 (id 1), c@3 (id 2).
        let s = schema();
        let dim4 = NonZeroUsize::new(4).unwrap();
        // The vector's declaration ordinal returns the Vector variant WITH dim,
        // never a ColumnId.
        assert_eq!(s.locate(ord(0)), Some(ColumnLocation::Vector { dim: dim4 }));
        // Each scalar's declaration ordinal returns the ColumnId the schema
        // STORED (read from def.id, not recomputed).
        assert_eq!(s.locate(ord(1)), Some(ColumnLocation::Scalar(0)));
        assert_eq!(s.locate(ord(2)), Some(ColumnLocation::Scalar(1)));
        assert_eq!(s.locate(ord(3)), Some(ColumnLocation::Scalar(2)));
    }

    #[test]
    fn locate_skips_the_vector_when_it_is_not_first() {
        // [author, vector, title]: author id 0 @ decl 0, vector @ decl 1,
        // title id 1 @ decl 2. Scalar ColumnIds are contiguous 0..N over the
        // scalars ONLY and skip the vector; locate returns the right answer for
        // each declaration ordinal.
        let s = Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(8).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "title".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap();
        let dim8 = NonZeroUsize::new(8).unwrap();
        assert_eq!(s.locate(ord(0)), Some(ColumnLocation::Scalar(0))); // author
        assert_eq!(s.locate(ord(1)), Some(ColumnLocation::Vector { dim: dim8 }));
        assert_eq!(s.locate(ord(2)), Some(ColumnLocation::Scalar(1))); // title
    }

    #[test]
    fn locate_out_of_range_ordinal_is_none() {
        // schema() has declaration ordinals 0..=3; anything beyond is None —
        // no panic, no wrong column.
        let s = schema();
        assert_eq!(s.locate(ord(4)), None);
        assert_eq!(s.locate(ord(99)), None);
    }

    #[test]
    fn scalar_column_ids_are_dense_over_scalars_only() {
        // Contiguity guard: with the vector in the MIDDLE, scalar ColumnIds are
        // still dense 0..N over the scalars alone, the vector excluded.
        let s = Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Vector {
                name: "v".into(),
                dim: NonZeroUsize::new(3).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "b".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "c".into(),
                ty: ColumnType::Float,
            },
        ])
        .unwrap();
        let ids: Vec<ColumnId> = s.columns.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(s.columns.len(), 3); // vector is not among the scalar columns
    }

    #[test]
    fn validate_row_accepts_well_formed_row_in_any_order() {
        let s = schema();
        let row: Row = vec![
            (2, Value::Text("x".into())),
            (0, Value::Int(7)),
            (1, Value::Float(0.5)),
        ];
        assert!(s.validate_row(&row).is_ok());
    }

    #[test]
    fn validate_row_rejects_wrong_length() {
        let s = schema();
        let short: Row = vec![(0, Value::Int(1))];
        assert!(matches!(s.validate_row(&short), Err(Error::IncompleteRow)));
    }

    #[test]
    fn validate_row_rejects_unknown_column() {
        let s = schema();
        let row: Row = vec![
            (0, Value::Int(1)),
            (1, Value::Float(1.0)),
            (9, Value::Text("x".into())),
        ];
        assert!(matches!(
            s.validate_row(&row),
            Err(Error::UnknownColumn { column: 9 })
        ));
    }

    #[test]
    fn validate_row_rejects_type_mismatch() {
        let s = schema();
        let row: Row = vec![
            (0, Value::Float(1.0)),
            (1, Value::Float(1.0)),
            (2, Value::Text("x".into())),
        ];
        assert!(matches!(
            s.validate_row(&row),
            Err(Error::TypeMismatch {
                column: 0,
                expected: ColumnType::Int,
                got: ColumnType::Float,
            })
        ));
    }

    #[test]
    fn validate_row_rejects_duplicate_column_ids() {
        let s = schema();
        // Right length, all ids known + right types, but column 0 twice.
        let row: Row = vec![
            (0, Value::Int(1)),
            (0, Value::Int(2)),
            (2, Value::Text("x".into())),
        ];
        assert!(matches!(s.validate_row(&row), Err(Error::IncompleteRow)));
    }

    #[test]
    fn validate_row_rejects_nan() {
        let s = schema();
        let row: Row = vec![
            (0, Value::Int(1)),
            (1, Value::Float(f64::NAN)),
            (2, Value::Text("x".into())),
        ];
        assert!(matches!(
            s.validate_row(&row),
            Err(Error::NaNRejected { column: 1 })
        ));
        // Infinities are ordered fine — only NaN is rejected.
        let inf: Row = vec![
            (0, Value::Int(1)),
            (1, Value::Float(f64::INFINITY)),
            (2, Value::Text("x".into())),
        ];
        assert!(s.validate_row(&inf).is_ok());
    }
}
