//! Statement parameters: the values behind `?` placeholders.
//!
//! A parameter is substituted AFTER lexing, as the literal tokens of its value
//! (see [`parse_with_params`](crate::sql::parser::parse_with_params)). It is
//! never source text, so nothing inside one — a quote, a `;`, a keyword — can
//! be read as syntax. That, not escaping, is what makes a parameterized
//! statement immune to injection.

/// One value for one `?` placeholder, in placeholder order.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// An `INT` value. Also valid where a `FLOAT` is expected, exactly as an
    /// integer literal is.
    Int(i64),
    /// A `FLOAT` value. Must be finite.
    Float(f64),
    /// A `TEXT` value, any UTF-8 — quotes and all.
    Text(String),
    /// An embedding: for `NEAREST TO ?`, an INSERT's vector column, or
    /// `SET vector = ?`. Must be non-empty and finite; stored bit for bit.
    Vector(Vec<f32>),
}

impl From<i64> for Param {
    fn from(n: i64) -> Param {
        Param::Int(n)
    }
}

impl From<f64> for Param {
    fn from(x: f64) -> Param {
        Param::Float(x)
    }
}

impl From<&str> for Param {
    fn from(s: &str) -> Param {
        Param::Text(s.to_string())
    }
}

impl From<String> for Param {
    fn from(s: String) -> Param {
        Param::Text(s)
    }
}

impl From<Vec<f32>> for Param {
    fn from(v: Vec<f32>) -> Param {
        Param::Vector(v)
    }
}

impl From<&[f32]> for Param {
    fn from(v: &[f32]) -> Param {
        Param::Vector(v.to_vec())
    }
}
