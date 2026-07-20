//! V-SQL abstract syntax tree.
//!
//! Pure type definitions — no logic, no parsing, no knowledge of `Token`.
//! The parser builds these; the logical planner consumes them. Nothing here
//! depends on the lexer, and nothing here knows how a statement is executed.
//!
//! Scope (deliberately minimal): exactly the three bootstrap statements —
//! CREATE COLLECTION, SELECT, INSERT. Extension points are marked `EXTEND:`.
//!
//! Grammar this mirrors (see parser.rs):
//!
//!   statement    := (select | insert | create) ';'
//!   select       := SELECT projection FROM ident
//!   projection   := '*' | ident (',' ident)*
//!   create       := CREATE COLLECTION ident '(' col_def (',' col_def)* ')'
//!                   WITH '(' opt (',' opt)* ')'
//!   col_def      := ident type
//!   type         := VECTOR '(' int_lit ')' | TEXT | INT | FLOAT
//!   opt          := ident '=' int_lit
//!   insert       := INSERT INTO ident '(' ident (',' ident)* ')'
//!                   VALUES '(' literal (',' literal)* ')'
//!   literal      := vector_lit | str_lit | number
//!   vector_lit   := '[' number (',' number)* ']'
//!   number       := '-'? (int_lit | float_lit)

// Every node derives Debug + Clone + PartialEq: the parser tests assert whole
// ASTs with `assert_eq!`, so structural equality is load-bearing. No methods —
// these are inert data the parser fills and the planner reads.

/// The AST root: one parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `SELECT …`
    Select(SelectStmt),
    /// `INSERT INTO …`
    Insert(InsertStmt),
    /// `CREATE COLLECTION …`
    CreateCollection(CreateStmt),
    // EXTEND: Search(SearchStmt), Delete(DeleteStmt), Update(UpdateStmt).
}

/// `SELECT projection FROM ident`.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStmt {
    /// What to return.
    pub projection: Projection,
    /// The collection name after `FROM`.
    pub from: String,
    // EXTEND: pub filter: Option<Expr>,  // WHERE — arrives with Expr, later.
}

/// The `SELECT` projection list.
#[derive(Debug, Clone, PartialEq)]
pub enum Projection {
    /// `*` — left UNEXPANDED here; expanding it needs the catalog (and must
    /// honor "SELECT * does not return the embedding"). The planner's job.
    Star,
    /// An explicit column list, in source order.
    Columns(Vec<String>),
}

/// `CREATE COLLECTION name ( columns ) WITH ( options )`.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateStmt {
    /// Collection name.
    pub name: String,
    /// Column definitions, in source order.
    pub columns: Vec<ColumnDef>,
    /// `WITH (...)` options, in source order.
    pub options: Vec<CollectionOption>,
}

/// A single `name type` column definition inside `CREATE COLLECTION`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// Column name (source case preserved).
    pub name: String,
    /// Its declared type.
    pub ty: ColumnType,
}

/// A column's declared type. Distinct from the storage engine's
/// `metadata::ColumnType` — this one is the *syntactic* type and carries the
/// vector dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// `VECTOR(dim)`.
    Vector(usize),
    /// `TEXT`.
    Text,
    /// `INT`.
    Int,
    /// `FLOAT`.
    Float,
}

/// One `name = value` entry from the `WITH (...)` clause (e.g. `capacity = 1000000`).
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionOption {
    /// Option name.
    pub name: String,
    /// Its integer value.
    pub value: i64,
}

/// `INSERT INTO collection ( columns ) VALUES ( values )`.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStmt {
    /// Target collection.
    pub collection: String,
    /// Target columns, in source order.
    pub columns: Vec<String>,
    /// One value per source position. The parser does NOT check that
    /// `values.len() == columns.len()` — count/type validation is the planner's.
    pub values: Vec<Literal>,
}

/// A literal value in an `INSERT ... VALUES (...)` list. Carries no type
/// checking and no schema knowledge: the parser cannot know whether a bare
/// `1700000000` targets an INT or FLOAT column. Coercion is the planner's job.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// `[…]` — a vector literal. `f32` to match the flat index element type;
    /// integer elements are coerced to `f32` by the parser.
    Vector(Vec<f32>),
    /// A single-quoted string.
    Str(String),
    /// A (possibly negated) integer.
    Int(i64),
    /// A (possibly negated) float.
    Float(f64),
}
