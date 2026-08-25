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
//!   select       := SELECT projection FROM ident [where]
//!   where        := WHERE expr
//!   expr         := or_expr
//!   or_expr      := and_expr (OR and_expr)*
//!   and_expr     := cmp_expr (AND cmp_expr)*
//!   cmp_expr     := primary [('<'|'<='|'>'|'>='|'='|'!=') primary]
//!   primary      := ident | literal | '(' expr ')'
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
    /// `SEARCH TOP … NEAREST TO … FROM …`
    Search(SearchStmt),
    // EXTEND: Delete(DeleteStmt), Update(UpdateStmt).
}

/// `SELECT projection FROM ident [WHERE expr]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStmt {
    /// What to return.
    pub projection: Projection,
    /// The collection name after `FROM`.
    pub from: String,
    /// The `WHERE` predicate, or `None` when the clause was omitted.
    pub filter: Option<Expr>,
}

/// A `WHERE` expression, as WRITTEN — unresolved and unchecked.
///
/// Deliberately more general than V-SQL accepts: `Compare` holds two arbitrary
/// sub-expressions, so `1 = 2` and `a < b` both parse. That is not an
/// oversight. The parser does no semantics anywhere else in this crate (a
/// column name is just a string until [`bind`](crate::sql::bind) resolves it),
/// and a grammar that could only *spell* legal predicates would have to know
/// which identifiers are columns — which needs the catalog. So the shape is
/// permissive here and narrowed by the binder, which can say
/// "`a < b`: comparing two columns is not supported" instead of the parser
/// saying "expected literal, found identifier".
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A bare identifier — a column reference, once something can confirm it.
    Column(String),
    /// A literal value.
    Literal(Literal),
    /// `left <op> right`.
    Compare {
        /// Left operand.
        left: Box<Expr>,
        /// The comparison.
        op: CompareOp,
        /// Right operand.
        right: Box<Expr>,
    },
    /// `left AND right` — binds tighter than `OR`.
    And(Box<Expr>, Box<Expr>),
    /// `left OR right` — the loosest operator.
    Or(Box<Expr>, Box<Expr>),
}

/// The six comparisons V-SQL supports. No `LIKE`, no `IN`, no `BETWEEN` —
/// see CLAUDE.md §4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `=`
    Eq,
    /// `!=` (or `<>`)
    Ne,
}

impl CompareOp {
    /// The operator meaning the same thing with its operands swapped:
    /// `2 > z` is `z < 2`.
    ///
    /// Used by the binder to normalize a flipped comparison into the
    /// `column <op> literal` form the metadata index can answer. Equality and
    /// inequality are symmetric and map to themselves.
    pub fn flipped(self) -> CompareOp {
        match self {
            CompareOp::Lt => CompareOp::Gt,
            CompareOp::Le => CompareOp::Ge,
            CompareOp::Gt => CompareOp::Lt,
            CompareOp::Ge => CompareOp::Le,
            CompareOp::Eq => CompareOp::Eq,
            CompareOp::Ne => CompareOp::Ne,
        }
    }
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

/// `SEARCH TOP k NEAREST TO [query] FROM collection [WHERE expr]
/// [RETURNING …]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchStmt {
    /// The `TOP` count. Carried as the lexed integer, unvalidated: the grammar
    /// already excludes a negative (the lexer emits `-` as its own token), but
    /// `TOP 0` parses fine and is the BINDER's to reject — the parser does not
    /// do semantics.
    pub k: i64,
    /// The query vector.
    ///
    /// Typed as [`Literal`], not `Vec<f32>`, so it is the SAME vector-literal an
    /// `INSERT` carries and a later stage can intern it as a `Const::Vector`
    /// unchanged. The parser only ever produces [`Literal::Vector`] here — it
    /// requires a `[` after `TO` — so the other variants are unreachable by
    /// construction, at the cost of the type not saying so.
    pub query: Literal,
    /// The collection name after `FROM`.
    pub collection: String,
    /// The `RETURNING` list, in source order, or `None` when the clause was
    /// omitted.
    ///
    /// `None` is NOT the same as `Some(vec![])`: it means "the default
    /// projection" (every scalar column, as bare `SEARCH` has always returned),
    /// whereas an empty list would be a request for no columns at all.
    ///
    /// The items are plain identifiers because at PARSE TIME they are
    /// indistinguishable: `id` and `score` are computed pseudo-columns and
    /// `title` is stored, but telling them apart needs the schema. That
    /// classification is [`bind`](crate::sql::bind)'s. Same representation as
    /// [`Projection::Columns`], deliberately — two different spellings of "a
    /// list of projected names" in one AST would be a trap.
    pub projection: Option<Vec<String>>,
    /// The `WHERE` predicate, or `None` when the clause was omitted.
    ///
    /// Same type as [`SelectStmt::filter`] — a predicate means the same thing
    /// in both statements, and giving `SEARCH` its own would guarantee the two
    /// drift apart.
    pub filter: Option<Expr>,
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
