//! Query frontend — the parser (Phase 7c skeleton / 7d logic).
//!
//! Turns the lexer's `Vec<SpannedToken>` into a [`Statement`] AST by straight
//! recursive descent — one grammar rule, one `parse_*` function. The grammar
//! is LL(1): a single token of lookahead always picks the branch, so there is
//! no backtracking and no parser generator.
//!
//! # Grammar
//!
//! ```text
//! statement    := (select | insert | create) ';'
//! select       := SELECT projection FROM ident
//! projection   := '*' | ident (',' ident)*
//! create       := CREATE COLLECTION ident '(' col_def (',' col_def)* ')'
//!                 WITH '(' opt (',' opt)* ')'
//! col_def      := ident type
//! type         := VECTOR '(' int_lit ')' | TEXT | INT | FLOAT
//! opt          := ident '=' int_lit
//! insert       := INSERT INTO ident '(' ident (',' ident)* ')'
//!                 VALUES '(' literal (',' literal)* ')'
//! literal      := vector_lit | str_lit | number
//! vector_lit   := '[' number (',' number)* ']'
//! number       := '-'? (int_lit | float_lit)
//! ```
//!
//! Notation: lowercase = a rule (a function); UPPERCASE/'quoted' = a lexer
//! token; `|` = or; `(...)*` = zero or more; `?` = optional.

use std::fmt;

use crate::sql::ast::{
    ColumnDef, ColumnType, CollectionOption, CreateStmt, InsertStmt, Literal, Projection,
    SelectStmt, Statement,
};
use crate::sql::lexer::{LexError, Lexer, Span, SpannedToken, Token};

/// A recursive-descent parser over a lexed token stream. Construct via the
/// free [`parse`] function rather than directly — it wires the lexer in.
pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

/// A syntax error, with the source span it was detected at. Mirrors the
/// lexer's [`LexError`] shape (kind + location).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// What went wrong.
    pub kind: ParseErrorKind,
    /// Where, as a source byte span.
    pub span: Span,
}

/// The category of a [`ParseError`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// The next token was not what the grammar required here.
    UnexpectedToken {
        /// A human description of what was expected (e.g. `"FROM"`, `"identifier"`).
        expected: String,
        /// A human description of what was found.
        found: String,
    },
    /// A word in type position that names no known type (e.g. `BLOB`).
    UnknownType(String),
    /// Input remained after a complete statement + its `;`.
    TrailingTokens,
    /// Input ended while a rule still expected more tokens.
    UnexpectedEof,
    /// The lexer failed before parsing could begin. `parse` surfaces lexer
    /// errors through this variant so callers deal with one error type.
    Lex(LexError),
    // EXTEND: new variants (e.g. for WHERE/Expr) may be added in later phases.
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ParseErrorKind::UnexpectedToken { expected, found } => write!(
                f,
                "unexpected token at byte {}: expected {expected}, found {found}",
                self.span.start
            ),
            ParseErrorKind::UnknownType(t) => {
                write!(f, "unknown type {t:?} at byte {}", self.span.start)
            }
            ParseErrorKind::TrailingTokens => {
                write!(f, "unexpected trailing tokens at byte {}", self.span.start)
            }
            ParseErrorKind::UnexpectedEof => write!(f, "unexpected end of input"),
            ParseErrorKind::Lex(e) => write!(f, "lex error: {e}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse V-SQL `src` into a single [`Statement`]. Lexes and parses in one call
/// so callers never touch the lexer directly. Exactly one statement is
/// expected; anything after its `;` is a [`ParseErrorKind::TrailingTokens`].
pub fn parse(src: &str) -> Result<Statement, ParseError> {
    let tokens = Lexer::new(src).tokenize().map_err(|e| ParseError {
        span: Span {
            start: e.pos,
            end: e.pos,
        },
        kind: ParseErrorKind::Lex(e),
    })?;
    let mut parser = Parser { tokens, pos: 0 };
    let statement = parser.parse_statement()?;
    // `parse_statement` consumes the trailing ';' (per the grammar); only Eof
    // may remain. Any real token here is leftover input.
    if let Some(st) = parser.tokens.get(parser.pos)
        && st.token != Token::Eof
    {
        return Err(ParseError {
            kind: ParseErrorKind::TrailingTokens,
            span: st.span,
        });
    }
    Ok(statement)
}

// The cursor primitives and one-per-rule functions. Bodies land in phase 7d;
// `#[allow(dead_code)]` covers the helpers that nothing calls until then.
#[allow(dead_code)]
impl Parser {
    // -- cursor primitives -------------------------------------------------

    /// The current token without consuming it. At/after end of input this is
    /// `Token::Eof` (the stream always ends in one).
    fn peek(&self) -> &Token {
        unimplemented!("phase 7d: parser logic")
    }

    /// Consume and return the current spanned token, advancing the cursor.
    fn advance(&mut self) -> &SpannedToken {
        unimplemented!("phase 7d: parser logic")
    }

    /// Consume the current token, requiring it to equal `t`; error otherwise.
    fn expect(&mut self, t: Token) -> Result<&SpannedToken, ParseError> {
        unimplemented!("phase 7d: parser logic — expect {t:?}")
    }

    /// Consume an identifier token, returning its (source-case) text.
    fn expect_ident(&mut self) -> Result<String, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    // -- one function per grammar rule -------------------------------------

    /// `statement := (select | insert | create) ';'`
    pub fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `select := SELECT projection FROM ident`
    fn parse_select(&mut self) -> Result<SelectStmt, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `projection := '*' | ident (',' ident)*`
    fn parse_projection(&mut self) -> Result<Projection, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `create := CREATE COLLECTION ident '(' col_def (',' col_def)* ')' WITH '(' opt (',' opt)* ')'`
    fn parse_create(&mut self) -> Result<CreateStmt, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `col_def := ident type`
    fn parse_col_def(&mut self) -> Result<ColumnDef, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `type := VECTOR '(' int_lit ')' | TEXT | INT | FLOAT`.
    /// Type names arrive as `Ident` (not keywords); resolved here BY POSITION,
    /// case-insensitively. An unknown word => [`ParseErrorKind::UnknownType`].
    fn parse_type(&mut self) -> Result<ColumnType, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `opt := ident '=' int_lit`
    fn parse_opt(&mut self) -> Result<CollectionOption, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `insert := INSERT INTO ident '(' ident (',' ident)* ')' VALUES '(' literal (',' literal)* ')'`
    fn parse_insert(&mut self) -> Result<InsertStmt, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `literal := vector_lit | str_lit | number`
    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `vector_lit := '[' number (',' number)* ']'` — elements coerced to `f32`.
    fn parse_vector_lit(&mut self) -> Result<Vec<f32>, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    /// `number := '-'? (int_lit | float_lit)` — the parser applies the sign
    /// (the lexer emits `-` as a separate `Minus` token).
    fn parse_number(&mut self) -> Result<Literal, ParseError> {
        unimplemented!("phase 7d: parser logic")
    }

    // EXTEND: `fn parse_expr(&mut self) -> Result<Expr, ParseError>` (WHERE)
    // lands here in a later phase — with it, its own Pratt/precedence
    // machinery. None of the three bootstrap statements contain an expression,
    // so there is deliberately no Expr type or precedence logic yet.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `src`, expecting success.
    fn ok(src: &str) -> Statement {
        parse(src).expect("expected a successful parse")
    }

    /// Parse `src`, expecting failure; return the error.
    fn err(src: &str) -> ParseError {
        match parse(src) {
            Ok(s) => panic!("expected a parse error, got {s:?}"),
            Err(e) => e,
        }
    }

    fn cols(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    // -- SELECT ------------------------------------------------------------

    #[test]
    fn select_with_column_list() {
        assert_eq!(
            ok("SELECT x, y FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Columns(cols(&["x", "y"])),
                from: "docs".to_string(),
            })
        );
    }

    #[test]
    fn select_star_is_not_expanded() {
        assert_eq!(
            ok("SELECT * FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Star,
                from: "docs".to_string(),
            })
        );
    }

    #[test]
    fn single_column_projection_parses() {
        // the zero-repetition case of (',' ident)*
        assert_eq!(
            ok("SELECT x FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Columns(cols(&["x"])),
                from: "docs".to_string(),
            })
        );
    }

    #[test]
    fn keyword_case_insensitivity_survives_to_ast() {
        assert_eq!(ok("select x from docs;"), ok("SELECT x FROM docs;"));
    }

    #[test]
    fn identifiers_keep_source_case() {
        assert_eq!(
            ok("SELECT published_at FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Columns(cols(&["published_at"])),
                from: "docs".to_string(),
            })
        );
    }

    // -- CREATE / types ----------------------------------------------------

    fn create_columns(stmt: &Statement) -> Vec<ColumnDef> {
        match stmt {
            Statement::CreateCollection(c) => c.columns.clone(),
            other => panic!("expected CreateCollection, got {other:?}"),
        }
    }

    #[test]
    fn types_resolve_by_position() {
        let stmt = ok("CREATE COLLECTION c (a VECTOR(4), b TEXT, d INT, e FLOAT) WITH (capacity = 1);");
        assert_eq!(
            create_columns(&stmt),
            vec![
                ColumnDef { name: "a".to_string(), ty: ColumnType::Vector(4) },
                ColumnDef { name: "b".to_string(), ty: ColumnType::Text },
                ColumnDef { name: "d".to_string(), ty: ColumnType::Int },
                ColumnDef { name: "e".to_string(), ty: ColumnType::Float },
            ]
        );
    }

    #[test]
    fn type_keywords_are_case_insensitive() {
        let lower = ok("CREATE COLLECTION c (author text) WITH (capacity = 1);");
        let upper = ok("CREATE COLLECTION c (author TEXT) WITH (capacity = 1);");
        assert_eq!(lower, upper);
        assert_eq!(
            create_columns(&lower),
            vec![ColumnDef { name: "author".to_string(), ty: ColumnType::Text }]
        );
    }

    #[test]
    fn column_named_vector_does_not_collide_with_vector_type() {
        // THE collision test: a column literally named `vector`, typed VECTOR.
        let stmt = ok("CREATE COLLECTION docs (vector VECTOR(768)) WITH (capacity = 1);");
        assert_eq!(
            create_columns(&stmt),
            vec![ColumnDef { name: "vector".to_string(), ty: ColumnType::Vector(768) }]
        );
    }

    #[test]
    fn unknown_type_errors() {
        let e = err("CREATE COLLECTION c (author BLOB) WITH (capacity = 1);");
        assert_eq!(e.kind, ParseErrorKind::UnknownType("BLOB".to_string()));
    }

    #[test]
    fn with_option_parses() {
        let stmt = ok("CREATE COLLECTION c (a INT) WITH (capacity = 1000000);");
        match stmt {
            Statement::CreateCollection(c) => assert_eq!(
                c.options,
                vec![CollectionOption { name: "capacity".to_string(), value: 1000000 }]
            ),
            other => panic!("expected CreateCollection, got {other:?}"),
        }
    }

    // -- INSERT / literals -------------------------------------------------

    fn insert_values(stmt: &Statement) -> Vec<Literal> {
        match stmt {
            Statement::Insert(i) => i.values.clone(),
            other => panic!("expected Insert, got {other:?}"),
        }
    }

    #[test]
    fn vector_literal_parses() {
        let stmt = ok("INSERT INTO docs (v) VALUES ([0.1, 0.2, 0.3]);");
        assert_eq!(insert_values(&stmt), vec![Literal::Vector(vec![0.1, 0.2, 0.3])]);
    }

    #[test]
    fn negative_vector_literal_applies_signs() {
        // proves the lexer's separate-Minus decision cashes out end to end.
        let stmt = ok("INSERT INTO docs (v) VALUES ([-0.1, 0.2, -0.3]);");
        assert_eq!(insert_values(&stmt), vec![Literal::Vector(vec![-0.1, 0.2, -0.3])]);
    }

    #[test]
    fn integer_vector_elements_coerce_to_f32() {
        let stmt = ok("INSERT INTO docs (v) VALUES ([1, 0, 0]);");
        assert_eq!(insert_values(&stmt), vec![Literal::Vector(vec![1.0, 0.0, 0.0])]);
    }

    #[test]
    fn single_element_vector_parses() {
        let stmt = ok("INSERT INTO docs (v) VALUES ([0.5]);");
        assert_eq!(insert_values(&stmt), vec![Literal::Vector(vec![0.5])]);
    }

    #[test]
    fn string_and_big_int_literals() {
        let stmt = ok("INSERT INTO docs (a, n) VALUES ('alice', 1700000000);");
        assert_eq!(
            insert_values(&stmt),
            vec![Literal::Str("alice".to_string()), Literal::Int(1700000000)]
        );
    }

    #[test]
    fn insert_column_value_count_mismatch_still_parses() {
        // A count mismatch is NOT a syntax error — the planner catches it later.
        let stmt = ok("INSERT INTO docs (a, b) VALUES (1);");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.columns.len(), 2);
                assert_eq!(i.values.len(), 1);
            }
            other => panic!("expected Insert, got {other:?}"),
        }
    }

    // -- error cases -------------------------------------------------------

    #[test]
    fn missing_from_is_unexpected_token() {
        let e = err("SELECT x docs;");
        assert!(
            matches!(e.kind, ParseErrorKind::UnexpectedToken { .. }),
            "expected UnexpectedToken, got {:?}",
            e.kind
        );
    }

    #[test]
    fn missing_semicolon_errors() {
        // no assertion on kind — just that it does not parse cleanly.
        let _ = err("SELECT x FROM docs");
    }

    #[test]
    fn trailing_tokens_after_semicolon_error() {
        let e = err("SELECT x FROM docs; garbage");
        assert_eq!(e.kind, ParseErrorKind::TrailingTokens);
    }

    #[test]
    fn trailing_comma_in_projection_errors() {
        let e = err("SELECT x, FROM docs;");
        assert!(
            matches!(e.kind, ParseErrorKind::UnexpectedToken { .. }),
            "expected UnexpectedToken, got {:?}",
            e.kind
        );
    }

    #[test]
    fn empty_input_is_unexpected_eof() {
        let e = err("");
        assert_eq!(e.kind, ParseErrorKind::UnexpectedEof);
    }

    // -- integration: exact full statements --------------------------------

    #[test]
    fn integration_select() {
        assert_eq!(
            ok("SELECT x, y FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Columns(cols(&["x", "y"])),
                from: "docs".to_string(),
            })
        );
    }

    #[test]
    fn integration_create_collection() {
        let src = "CREATE COLLECTION docs (\n\
                   \x20   vector VECTOR(768),\n\
                   \x20   author TEXT,\n\
                   \x20   title TEXT,\n\
                   \x20   published_at INT\n\
                   ) WITH (capacity = 1000000);";
        assert_eq!(
            ok(src),
            Statement::CreateCollection(CreateStmt {
                name: "docs".to_string(),
                columns: vec![
                    ColumnDef { name: "vector".to_string(), ty: ColumnType::Vector(768) },
                    ColumnDef { name: "author".to_string(), ty: ColumnType::Text },
                    ColumnDef { name: "title".to_string(), ty: ColumnType::Text },
                    ColumnDef { name: "published_at".to_string(), ty: ColumnType::Int },
                ],
                options: vec![CollectionOption { name: "capacity".to_string(), value: 1000000 }],
            })
        );
    }

    #[test]
    fn integration_insert() {
        let src = "INSERT INTO docs (vector, author, title, published_at) \
                   VALUES ([0.1, 0.2, 0.3], 'alice', 'My doc', 1700000000);";
        assert_eq!(
            ok(src),
            Statement::Insert(InsertStmt {
                collection: "docs".to_string(),
                columns: cols(&["vector", "author", "title", "published_at"]),
                values: vec![
                    Literal::Vector(vec![0.1, 0.2, 0.3]),
                    Literal::Str("alice".to_string()),
                    Literal::Str("My doc".to_string()),
                    Literal::Int(1700000000),
                ],
            })
        );
    }
}
