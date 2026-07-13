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
pub type ColumnId = u32;

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
    pub dim: NonZeroUsize,
    pub capacity: usize,
    /// The metadata schema. Every insert's row is validated against it BEFORE
    /// logging. Vector-only collections use an empty schema
    /// (`Schema::new(vec![])`) and insert with an empty row.
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
    /// ColumnId is the index into this Vec — i.e. columns[3].id == 3.
    /// Keeping that invariant makes (de)serialization trivial.
    pub columns: Vec<ColumnDef>,
    /// name → id, built from `columns` at construction time.
    pub by_name: HashMap<String, ColumnId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    pub id: ColumnId,
    pub name: String,
    pub ty: ColumnType,
}

impl Schema {
    /// Assign ids 0..n in declaration order and build the name→id map.
    /// Duplicate names → Err(Error::DuplicateColumn).
    pub fn new(defs: Vec<(String, ColumnType)>) -> Result<Self> {
        let mut columns = Vec::with_capacity(defs.len());
        let mut by_name = HashMap::with_capacity(defs.len());
        for (id, (name, ty)) in defs.into_iter().enumerate() {
            let id = id as ColumnId;
            if by_name.insert(name.clone(), id).is_some() {
                return Err(Error::DuplicateColumn(name));
            }
            columns.push(ColumnDef { id, name, ty });
        }
        Ok(Schema { columns, by_name })
    }

    pub fn column(&self, id: ColumnId) -> Option<&ColumnDef> {
        // columns[i].id == i by construction, so the id IS the index.
        self.columns.get(id as usize)
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

    fn schema() -> Schema {
        Schema::new(vec![
            ("a".into(), ColumnType::Int),
            ("b".into(), ColumnType::Float),
            ("c".into(), ColumnType::Text),
        ])
        .unwrap()
    }

    #[test]
    fn new_assigns_ids_in_declaration_order() {
        let s = schema();
        assert_eq!(s.columns.len(), 3);
        for (i, def) in s.columns.iter().enumerate() {
            assert_eq!(def.id, i as ColumnId);
        }
        assert_eq!(s.by_name["b"], 1);
        assert_eq!(s.column(2).unwrap().name, "c");
        assert!(s.column(3).is_none());
    }

    #[test]
    fn new_rejects_duplicate_names() {
        let err = Schema::new(vec![
            ("a".into(), ColumnType::Int),
            ("a".into(), ColumnType::Text),
        ])
        .unwrap_err();
        assert!(matches!(err, Error::DuplicateColumn(name) if name == "a"));
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
