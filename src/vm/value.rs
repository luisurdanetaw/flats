//! The value bridge: V-SQL [`Literal`] ↔ storage [`Value`].
//!
//! Two vocabularies for the same data. The frontend produces `Literal`s (the
//! parser's output, carried through the binder and interned by the compiler);
//! the three stores speak [`metadata::Value`](crate::metadata::common::Value).
//! The VM sits between them and needs BOTH directions — `Literal → Value` to
//! build the row an `Op::Insert` writes, `Value → Literal` to render the row a
//! `ResultRow` emits.
//!
//! # The two directions are not symmetric
//!
//! `Literal` has FOUR variants; `Value` has three. The extra one is
//! [`Literal::Vector`], and it has no `Value` counterpart *by design*: the
//! embedding does not live in the tuple store or the metadata index at all — it
//! lives in the flat vector index, addressed by ordinal. So:
//!
//!   * **`Value → Literal` is total.** Every stored scalar renders.
//!   * **`Literal → Value` is PARTIAL.** A vector literal is not a metadata
//!     value, and forcing one into `Value` would mean inventing a fourth variant
//!     in storage for something storage deliberately keeps elsewhere.
//!
//! That asymmetry is why the forward direction is a `TryFrom` and the reverse a
//! plain `From`. In practice the fallible arm is unreachable on the insert path:
//! [`split_record`](super::record::split_record) asks the schema which position
//! IS the vector and routes it to the flat index before any scalar conversion
//! happens, so the only `Literal` this ever sees is one the schema already
//! called a scalar. The error exists because the type system cannot express
//! that — not because the VM is expected to hit it.
//!
//! # No coercion here
//!
//! An `INT` literal bound to a `FLOAT` column is canonicalized to
//! `Literal::Float` by the BINDER (see
//! [`TypedValue`](crate::sql::bind::TypedValue)). This bridge is a
//! transcription: `Int → Int`, `Float → Float`, `Str → Text`. If it coerced too
//! there would be two places that decide what an `INT` in a `FLOAT` column
//! means, and they would drift.

use std::fmt;

use crate::metadata::common::Value;
use crate::sql::ast::Literal;
use crate::vm::exec::RegValue;

/// Why a [`Literal`] could not be lowered to a storage [`Value`].
///
/// One cause today, and structurally the only one possible: the literal is a
/// vector, which storage keeps in the flat index rather than as a metadata
/// value. See the module header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    /// A vector literal was handed to the scalar bridge. The embedding belongs
    /// in the flat vector index — route it there via
    /// [`split_record`](super::record::split_record) instead.
    NotScalar,
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueError::NotScalar => write!(
                f,
                "a vector literal is not a metadata value (the embedding lives in the flat index)"
            ),
        }
    }
}

impl std::error::Error for ValueError {}

/// Lower a scalar literal to its storage value.
///
/// Partial: [`Literal::Vector`] has no `Value` counterpart (see the module
/// header). Takes `&Literal` because the caller — the VM's insert path — is
/// walking a packed record it does not own.
impl TryFrom<&Literal> for Value {
    type Error = ValueError;

    fn try_from(literal: &Literal) -> Result<Self, ValueError> {
        // No wildcard arm: a new `Literal` variant must be mapped explicitly or
        // this stops compiling, which is exactly the review we want.
        match literal {
            Literal::Int(n) => Ok(Value::Int(*n)),
            Literal::Float(f) => Ok(Value::Float(*f)),
            Literal::Str(s) => Ok(Value::Text(s.clone())),
            Literal::Vector(_) => Err(ValueError::NotScalar),
        }
    }
}

/// Render a stored value back as a literal. Total — every `Value` variant is a
/// scalar, and every scalar has a `Literal` form.
impl From<&Value> for Literal {
    fn from(value: &Value) -> Literal {
        match value {
            Value::Int(n) => Literal::Int(*n),
            Value::Float(f) => Literal::Float(*f),
            Value::Text(s) => Literal::Str(s.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// The register end of the same bridge
//
// The VM adds a third vocabulary: a REGISTER holds a `RegValue`. The two
// conversions below are its edges, and they are the reason the whole insert path
// reads `register → Literal → Value`:
//
//   Op::String / VectorConst / Integer / Real   pool or operand → RegValue
//   Op::MakeRecord                              RegValue → Literal  (here)
//   split_record                                Literal  → Value    (Prompt 4)
//
// and the read path is the mirror: `Op::Column` turns a stored `Value` straight
// into a `RegValue` (below), skipping `Literal` because there is no packed
// record on the way out.
// ---------------------------------------------------------------------------

/// A register's contents as a [`Literal`], for packing into a
/// [`Record`](super::record::Record).
///
/// `None` for [`RegValue::Unset`] and [`RegValue::Record`] — a register that was
/// never written, and a record nested inside a record. Both are emitter bugs
/// with precise [`ExecError`](super::exec::ExecError) variants, so this returns
/// `Option` rather than inventing a third error type for cases the caller can
/// already name better.
///
/// The vector arm COPIES: `RegValue` holds an `Arc<[f32]>` but `Literal::Vector`
/// owns a `Vec<f32>`. That is the one place the two representations disagree,
/// and it costs one copy per inserted row — acceptable because `Db::insert`
/// copies again into the WAL record anyway. Making `Record` hold `RegValue`
/// directly would remove the hop; it is not worth rewriting `split_record` for.
pub fn to_literal(value: &RegValue) -> Option<Literal> {
    // No wildcard: a new `RegValue` variant must be classified here.
    match value {
        RegValue::Int(n) => Some(Literal::Int(*n)),
        RegValue::Real(f) => Some(Literal::Float(*f)),
        RegValue::Str(s) => Some(Literal::Str(s.clone())),
        RegValue::Vector(v) => Some(Literal::Vector(v.to_vec())),
        // A bitmap is a WHERE intermediate, never a stored value.
        RegValue::Unset | RegValue::Record(_) | RegValue::Bitmap(_) => None,
    }
}

/// A stored value as a register's contents, for [`Op::Column`](crate::compiler::Op).
/// Total — every stored scalar has a register form.
impl From<&Value> for RegValue {
    fn from(value: &Value) -> RegValue {
        match value {
            Value::Int(n) => RegValue::Int(*n),
            Value::Float(f) => RegValue::Real(*f),
            Value::Text(s) => RegValue::Str(s.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ValueError;
    use crate::metadata::common::Value;
    use crate::sql::ast::Literal;

    /// One representative of EVERY [`Literal`] variant.
    ///
    /// The `match` below is the guard, not decoration: it has no wildcard arm,
    /// so adding a variant to `Literal` breaks this file until the variant is
    /// given a case here — and `literal_to_value_roundtrip` then has to say
    /// whether it lowers or not.
    fn every_literal() -> Vec<Literal> {
        let all = vec![
            Literal::Int(-1_700_000_000),
            // Exactly representable, so the round-trip compares bit-for-bit.
            Literal::Float(-0.25),
            Literal::Str("alice".into()),
            Literal::Vector(vec![0.5, -0.25, 0.125]),
        ];
        for literal in &all {
            match literal {
                Literal::Int(_) => {}
                Literal::Float(_) => {}
                Literal::Str(_) => {}
                Literal::Vector(_) => {}
            }
        }
        all
    }

    /// Likewise for [`Value`] — every stored variant has a representative.
    fn every_value() -> Vec<Value> {
        let all = vec![Value::Int(42), Value::Float(1.5), Value::Text("bob".into())];
        for value in &all {
            match value {
                Value::Int(_) => {}
                Value::Float(_) => {}
                Value::Text(_) => {}
            }
        }
        all
    }

    #[test]
    fn literal_to_value_roundtrip() {
        for literal in every_literal() {
            // Exhaustive, no wildcard: a new variant lands here as a compile
            // error rather than slipping through unmapped.
            match &literal {
                Literal::Int(_) | Literal::Float(_) | Literal::Str(_) => {
                    let value = Value::try_from(&literal).expect("a scalar literal lowers");
                    assert_eq!(
                        Literal::from(&value),
                        literal,
                        "{literal:?} did not survive the round trip"
                    );
                }
                // The asymmetry, asserted rather than assumed: the embedding is
                // not a metadata value, so this arm has no round trip to make.
                Literal::Vector(_) => {
                    assert_eq!(Value::try_from(&literal), Err(ValueError::NotScalar));
                }
            }
        }
    }

    #[test]
    fn value_to_literal_is_total() {
        // The other direction, which IS total: every stored scalar renders as a
        // literal and comes back unchanged.
        for value in every_value() {
            let literal = Literal::from(&value);
            assert_eq!(
                Value::try_from(&literal).expect("a rendered scalar re-lowers"),
                value,
                "{value:?} did not survive the round trip"
            );
        }
    }

    #[test]
    fn text_and_str_are_the_same_variant_under_two_names() {
        // The one pairing whose names differ (`Str` vs `Text`) — easy to
        // mis-wire, and nothing else in the mapping would notice.
        assert_eq!(
            Value::try_from(&Literal::Str("x".into())),
            Ok(Value::Text("x".into()))
        );
        assert_eq!(
            Literal::from(&Value::Text("x".into())),
            Literal::Str("x".into())
        );
    }
}
