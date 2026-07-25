//! Constant pool — the side table a [`Program`](super::bytecode::Program) carries
//! for values too large to live in a register.
//!
//! A register holds only a small tagged value or a *handle*; anything big — a
//! vector literal, a string, the schema a `CREATE COLLECTION` provisions — lives
//! here and is referenced by a [`ConstId`]. The compiler interns a value with
//! [`ConstPool::add`] (which returns the handle) and threads the handle through
//! the instruction stream; the VM reads the payload back with [`ConstPool::get`].
//!
//! This is plain infrastructure — no compilation logic lives here.

use crate::metadata::common::Schema;

/// A handle into a [`ConstPool`]. Instructions carry this, never the payload.
///
/// A newtype (not a bare `u32`) so a constant handle can never be confused with
/// a register index, cursor slot, or jump target — the operand spaces stay
/// type-distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(pub u32);

/// A pooled constant — a value too large to sit in a register.
///
/// The three payload shapes the core instruction set needs: a vector literal
/// (`VectorConst`), a string (`String`), and a collection schema
/// (`CreateCollection`). Integers and floats are NOT here — they load inline via
/// `Integer` / `Real`, small enough to be instruction operands.
#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    /// A vector literal — `f32` to match the flat index element type.
    Vector(Vec<f32>),
    /// A string literal.
    Str(String),
    /// The resolved storage schema a `CREATE COLLECTION` provisions.
    Schema(Schema),
}

/// The append-only pool of a program's constants. Handles are dense indices
/// assigned in insertion order, so a program and its pool round-trip by value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConstPool {
    consts: Vec<Const>,
}

impl ConstPool {
    /// An empty pool.
    pub fn new() -> Self {
        ConstPool { consts: Vec::new() }
    }

    /// Intern `value`, returning its handle. Handles are assigned densely in
    /// insertion order (`0, 1, 2, …`); no de-duplication — two equal values get
    /// two handles, which keeps `add` a pure push.
    pub fn add(&mut self, value: Const) -> ConstId {
        let id = ConstId(self.consts.len() as u32);
        self.consts.push(value);
        id
    }

    /// The payload behind `id`, or `None` if the handle is out of range.
    pub fn get(&self, id: ConstId) -> Option<&Const> {
        self.consts.get(id.0 as usize)
    }

    /// How many constants are interned — also the value the next [`add`](Self::add)
    /// will hand out.
    pub fn len(&self) -> usize {
        self.consts.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.consts.is_empty()
    }
}
