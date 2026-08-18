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
    CollectionOption, ColumnDef, ColumnType, CompareOp, CreateStmt, Expr, InsertStmt, Literal,
    Projection, SearchStmt, SelectStmt, Statement,
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

impl Parser {
    // -- cursor primitives -------------------------------------------------

    /// The current token without consuming it. The stream always ends in
    /// `Token::Eof`, and the cursor never advances past it, so this is always
    /// a valid token.
    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    /// Consume and return the current spanned token, advancing the cursor
    /// (which stays pinned on the trailing `Eof` once reached).
    fn advance(&mut self) -> &SpannedToken {
        let i = self.pos;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        &self.tokens[i]
    }

    /// Build an error carrying the current token's span.
    fn error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            kind,
            span: self.tokens[self.pos].span,
        }
    }

    /// Consume the current token, requiring it to equal `t`. Reaching `Eof`
    /// while expecting something is [`ParseErrorKind::UnexpectedEof`]; a
    /// different token present is [`ParseErrorKind::UnexpectedToken`].
    fn expect(&mut self, t: Token) -> Result<&SpannedToken, ParseError> {
        if *self.peek() == t {
            Ok(self.advance())
        } else if *self.peek() == Token::Eof {
            Err(self.error(ParseErrorKind::UnexpectedEof))
        } else {
            let found = format!("{:?}", self.peek());
            Err(self.error(ParseErrorKind::UnexpectedToken {
                expected: format!("{t:?}"),
                found,
            }))
        }
    }

    /// Consume an identifier token, returning its (source-case) text.
    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let name = match self.peek() {
            Token::Ident(s) => s.clone(),
            Token::Eof => return Err(self.error(ParseErrorKind::UnexpectedEof)),
            other => {
                let found = format!("{other:?}");
                return Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found,
                }));
            }
        };
        self.advance();
        Ok(name)
    }

    /// Consume an integer-literal token, returning its value.
    fn expect_int(&mut self) -> Result<i64, ParseError> {
        let n = match self.peek() {
            Token::IntLit(n) => *n,
            Token::Eof => return Err(self.error(ParseErrorKind::UnexpectedEof)),
            other => {
                let found = format!("{other:?}");
                return Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "integer literal".to_string(),
                    found,
                }));
            }
        };
        self.advance();
        Ok(n)
    }

    /// Consume a string-literal token, returning its (unescaped) contents.
    fn expect_str(&mut self) -> Result<String, ParseError> {
        let s = match self.peek() {
            Token::StrLit(s) => s.clone(),
            Token::Eof => return Err(self.error(ParseErrorKind::UnexpectedEof)),
            other => {
                let found = format!("{other:?}");
                return Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "string literal".to_string(),
                    found,
                }));
            }
        };
        self.advance();
        Ok(s)
    }

    // -- one function per grammar rule -------------------------------------

    /// `statement := (select | insert | create | search) ';'`
    pub fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        let stmt = match self.peek() {
            Token::Select => Statement::Select(self.parse_select()?),
            Token::Insert => Statement::Insert(self.parse_insert()?),
            Token::Create => Statement::CreateCollection(self.parse_create()?),
            Token::Search => Statement::Search(self.parse_search()?),
            // EXTEND: dispatch Delete/Update on their leading keyword.
            Token::Eof => return Err(self.error(ParseErrorKind::UnexpectedEof)),
            _ => {
                let found = format!("{:?}", self.peek());
                return Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "SELECT, INSERT, CREATE, or SEARCH".to_string(),
                    found,
                }));
            }
        };
        self.expect(Token::Semicolon)?;
        Ok(stmt)
    }

    /// `select := SELECT projection FROM ident [WHERE expr]`
    fn parse_select(&mut self) -> Result<SelectStmt, ParseError> {
        self.expect(Token::Select)?;
        let projection = self.parse_projection()?;
        self.expect(Token::From)?;
        let from = self.expect_ident()?;
        let filter = self.parse_where()?;
        Ok(SelectStmt {
            projection,
            from,
            filter,
        })
    }

    /// `where := WHERE expr` — the whole clause, absent when the keyword is not
    /// there. Shared by every statement that takes a predicate, so `SELECT` and
    /// `SEARCH` cannot end up accepting different expression grammars.
    fn parse_where(&mut self) -> Result<Option<Expr>, ParseError> {
        if *self.peek() != Token::Where {
            return Ok(None);
        }
        self.advance();
        Ok(Some(self.parse_expr()?))
    }

    /// `expr := or_expr` — the entry point, named for what callers want.
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    /// `or_expr := and_expr (OR and_expr)*`
    ///
    /// `OR` is the LOOSEST operator, so it sits outermost: parsing it here
    /// means each side is a fully-parsed `AND` chain, which is what makes
    /// `a = 1 OR b = 2 AND c = 3` group as `a = 1 OR (b = 2 AND c = 3)`.
    /// Left-associative, like SQL.
    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while *self.peek() == Token::Or {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `and_expr := cmp_expr (AND cmp_expr)*`
    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_comparison()?;
        while *self.peek() == Token::And {
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `cmp_expr := primary [op primary]`
    ///
    /// NON-associative, deliberately: `a < b < c` is a chained comparison that
    /// means nothing here, and looping would silently accept it as
    /// `(a < b) < c`. One optional operator, then stop — a second one is left
    /// for the caller to choke on, which reports the error at the right token.
    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_primary()?;
        let op = match self.peek() {
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_primary()?;
        Ok(Expr::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    /// `primary := ident | literal | '(' expr ')'`
    ///
    /// An identifier becomes [`Expr::Column`] without checking that any such
    /// column exists — that needs the catalog, and it is the binder's.
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Token::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Ident(_) => Ok(Expr::Column(self.expect_ident()?)),
            // Every literal form the rest of the grammar knows, including
            // vectors — which no predicate can usefully compare, but rejecting
            // that HERE would mean the parser knowing what a comparison means.
            // The binder says so, in terms of the column's type.
            Token::StrLit(_) | Token::Minus | Token::IntLit(_) | Token::FloatLit(_)
            | Token::LBracket => Ok(Expr::Literal(self.parse_literal()?)),
            Token::Eof => Err(self.error(ParseErrorKind::UnexpectedEof)),
            other => {
                let found = format!("{other:?}");
                Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "column name, literal, or `(`".to_string(),
                    found,
                }))
            }
        }
    }

    /// `search := SEARCH TOP int_lit NEAREST TO vector_lit FROM ident
    ///            (WHERE expr)? (RETURNING ident (',' ident)*)?`
    ///
    /// Every keyword is required and positional, so a missing one is an
    /// `UnexpectedToken` naming what was wanted. The query reuses
    /// [`parse_vector_lit`](Self::parse_vector_lit) — the same production an
    /// `INSERT`'s embedding goes through — so the two can never drift.
    fn parse_search(&mut self) -> Result<SearchStmt, ParseError> {
        self.expect(Token::Search)?;
        self.expect(Token::Top)?;
        let k = self.expect_int()?;
        self.expect(Token::Nearest)?;
        self.expect(Token::To)?;
        // Requiring the `[` here is what makes `SearchStmt::query` always a
        // `Literal::Vector`: a string or a bare number after `TO` is a parse
        // error, not something a later stage has to re-check.
        let query = Literal::Vector(self.parse_vector_lit()?);
        self.expect(Token::From)?;
        let collection = self.expect_ident()?;
        // WHERE comes BEFORE RETURNING (CLAUDE.md §4): the predicate selects
        // rows, the projection describes what comes back, and reading them in
        // that order is what the spec's examples spell.
        let filter = self.parse_where()?;
        // OPTIONAL. Absent means the default projection — see
        // `SearchStmt::projection` for why that is `None` and not an empty list.
        let projection = if *self.peek() == Token::Returning {
            self.advance();
            let mut names = vec![self.expect_ident()?];
            while *self.peek() == Token::Comma {
                self.advance();
                names.push(self.expect_ident()?);
            }
            Some(names)
        } else {
            None
        };
        Ok(SearchStmt {
            k,
            query,
            collection,
            projection,
            filter,
        })
    }

    /// `projection := '*' | ident (',' ident)*`
    fn parse_projection(&mut self) -> Result<Projection, ParseError> {
        // `*` is left UNEXPANDED here — expanding it needs the catalog (and must
        // honor "SELECT * does not return the embedding"), which is the planner's
        // job, not the parser's.
        if *self.peek() == Token::Star {
            self.advance();
            return Ok(Projection::Star);
        }
        let mut columns = vec![self.expect_ident()?];
        while *self.peek() == Token::Comma {
            self.advance();
            columns.push(self.expect_ident()?);
        }
        Ok(Projection::Columns(columns))
    }

    /// `create := CREATE COLLECTION ident '(' col_def (',' col_def)* ')' WITH '(' opt (',' opt)* ')'`
    fn parse_create(&mut self) -> Result<CreateStmt, ParseError> {
        self.expect(Token::Create)?;
        self.expect(Token::Collection)?;
        let name = self.expect_ident()?;

        self.expect(Token::LParen)?;
        let mut columns = vec![self.parse_col_def()?];
        while *self.peek() == Token::Comma {
            self.advance();
            columns.push(self.parse_col_def()?);
        }
        self.expect(Token::RParen)?;

        self.expect(Token::With)?;
        self.expect(Token::LParen)?;
        let mut options = vec![self.parse_opt()?];
        while *self.peek() == Token::Comma {
            self.advance();
            options.push(self.parse_opt()?);
        }
        self.expect(Token::RParen)?;

        Ok(CreateStmt {
            name,
            columns,
            options,
        })
    }

    /// `col_def := ident type`
    fn parse_col_def(&mut self) -> Result<ColumnDef, ParseError> {
        let name = self.expect_ident()?;
        let ty = self.parse_type()?;
        Ok(ColumnDef { name, ty })
    }

    /// `type := VECTOR '(' int_lit ')' | TEXT | INT | FLOAT`.
    /// Type names arrive as `Ident` (not keywords); resolved here BY POSITION,
    /// case-insensitively. An unknown word => [`ParseErrorKind::UnknownType`].
    fn parse_type(&mut self) -> Result<ColumnType, ParseError> {
        // Capture the type word's span before consuming it, so an UnknownType
        // error points at the word rather than the token after it.
        let span = self.tokens[self.pos].span;
        let word = self.expect_ident()?;
        match word.to_ascii_uppercase().as_str() {
            "VECTOR" => {
                self.expect(Token::LParen)?;
                let dim = self.expect_int()?;
                self.expect(Token::RParen)?;
                match usize::try_from(dim) {
                    Ok(d) => Ok(ColumnType::Vector(d)),
                    Err(_) => Err(ParseError {
                        kind: ParseErrorKind::UnknownType(word),
                        span,
                    }),
                }
            }
            "TEXT" => Ok(ColumnType::Text),
            "INT" => Ok(ColumnType::Int),
            "FLOAT" => Ok(ColumnType::Float),
            _ => Err(ParseError {
                kind: ParseErrorKind::UnknownType(word),
                span,
            }),
        }
    }

    /// `opt := ident '=' int_lit`
    fn parse_opt(&mut self) -> Result<CollectionOption, ParseError> {
        let name = self.expect_ident()?;
        self.expect(Token::Eq)?;
        let value = self.expect_int()?;
        Ok(CollectionOption { name, value })
    }

    /// `insert := INSERT INTO ident '(' ident (',' ident)* ')' VALUES '(' literal (',' literal)* ')'`
    fn parse_insert(&mut self) -> Result<InsertStmt, ParseError> {
        self.expect(Token::Insert)?;
        self.expect(Token::Into)?;
        let collection = self.expect_ident()?;

        self.expect(Token::LParen)?;
        let mut columns = vec![self.expect_ident()?];
        while *self.peek() == Token::Comma {
            self.advance();
            columns.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;

        self.expect(Token::Values)?;
        self.expect(Token::LParen)?;
        let mut values = vec![self.parse_literal()?];
        while *self.peek() == Token::Comma {
            self.advance();
            values.push(self.parse_literal()?);
        }
        self.expect(Token::RParen)?;

        // NOTE: no check that columns.len() == values.len() — a count mismatch
        // is a semantic error the planner catches, not a syntax error.
        Ok(InsertStmt {
            collection,
            columns,
            values,
        })
    }

    /// `literal := vector_lit | str_lit | number`
    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        match self.peek() {
            Token::LBracket => Ok(Literal::Vector(self.parse_vector_lit()?)),
            Token::StrLit(_) => Ok(Literal::Str(self.expect_str()?)),
            Token::Minus | Token::IntLit(_) | Token::FloatLit(_) => self.parse_number(),
            Token::Eof => Err(self.error(ParseErrorKind::UnexpectedEof)),
            _ => {
                let found = format!("{:?}", self.peek());
                Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "literal".to_string(),
                    found,
                }))
            }
        }
    }

    /// `vector_lit := '[' number (',' number)* ']'` — elements coerced to `f32`.
    fn parse_vector_lit(&mut self) -> Result<Vec<f32>, ParseError> {
        self.expect(Token::LBracket)?;
        let mut elems = vec![self.parse_vector_elem()?];
        while *self.peek() == Token::Comma {
            self.advance();
            elems.push(self.parse_vector_elem()?);
        }
        self.expect(Token::RBracket)?;
        Ok(elems)
    }

    /// A single vector element: a `number` coerced to `f32`.
    fn parse_vector_elem(&mut self) -> Result<f32, ParseError> {
        match self.parse_number()? {
            Literal::Int(n) => Ok(n as f32),
            Literal::Float(f) => Ok(f as f32),
            // parse_number only ever yields Int or Float; stay exhaustive
            // without panicking.
            Literal::Vector(_) | Literal::Str(_) => {
                Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "number".to_string(),
                    found: "non-numeric literal".to_string(),
                }))
            }
        }
    }

    /// `number := '-'? (int_lit | float_lit)` — the parser applies the sign
    /// (the lexer emits `-` as a separate `Minus` token). The lexed magnitude
    /// always fits, so negation cannot overflow.
    fn parse_number(&mut self) -> Result<Literal, ParseError> {
        let negative = *self.peek() == Token::Minus;
        if negative {
            self.advance();
        }
        let lit = match self.peek() {
            Token::IntLit(n) => Literal::Int(if negative { -*n } else { *n }),
            Token::FloatLit(f) => Literal::Float(if negative { -*f } else { *f }),
            Token::Eof => return Err(self.error(ParseErrorKind::UnexpectedEof)),
            other => {
                let found = format!("{other:?}");
                return Err(self.error(ParseErrorKind::UnexpectedToken {
                    expected: "number".to_string(),
                    found,
                }));
            }
        };
        self.advance();
        Ok(lit)
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

    // -- SEARCH ------------------------------------------------------------
    //
    // `search := SEARCH TOP int_lit NEAREST TO vector_lit FROM ident`
    // Bare form only — no RETURNING, no WHERE.

    #[test]
    fn parses_bare_search() {
        assert_eq!(
            ok("SEARCH TOP 5 NEAREST TO [0.1, 0.2, 0.3] FROM docs;"),
            Statement::Search(SearchStmt {
                k: 5,
                query: Literal::Vector(vec![0.1, 0.2, 0.3]),
                collection: "docs".to_string(),
                projection: None,
                filter: None,
            })
        );
        // Keywords are case-insensitive, like every other statement.
        assert_eq!(
            ok("search top 5 nearest to [0.1, 0.2, 0.3] from docs;"),
            ok("SEARCH TOP 5 NEAREST TO [0.1, 0.2, 0.3] FROM docs;")
        );
        // ...and the collection name keeps its source case.
        match ok("SEARCH TOP 1 NEAREST TO [1.0] FROM MyDocs;") {
            Statement::Search(s) => assert_eq!(s.collection, "MyDocs"),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn search_k_is_the_top_integer() {
        // Several values, so a hard-coded or off-by-one `k` cannot pass.
        for n in [1, 5, 10, 1000] {
            match ok(&format!("SEARCH TOP {n} NEAREST TO [1.0, 2.0] FROM docs;")) {
                Statement::Search(s) => assert_eq!(s.k, n, "TOP {n}"),
                other => panic!("expected Search, got {other:?}"),
            }
        }
        // `TOP 0` PARSES. It is meaningless, but "k must be >= 1" is a semantic
        // rule, and the parser does not do semantics — the binder rejects it.
        match ok("SEARCH TOP 0 NEAREST TO [1.0] FROM docs;") {
            Statement::Search(s) => assert_eq!(s.k, 0),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn search_query_is_a_vector_literal() {
        // The query slot holds the SAME `Literal::Vector` an INSERT carries —
        // not a string, not a bare `Vec<f32>`, not a new type. That is what lets
        // a later stage intern it as `Const::Vector` with no conversion, exactly
        // as `INSERT` already does.
        match ok("SEARCH TOP 3 NEAREST TO [0.5, -0.25, 0.125] FROM docs;") {
            Statement::Search(s) => {
                assert_eq!(s.query, Literal::Vector(vec![0.5, -0.25, 0.125]));
                // Integer elements coerce to f32, the same rule vector literals
                // already follow everywhere else.
                match ok("SEARCH TOP 3 NEAREST TO [1, 2, 3] FROM docs;") {
                    Statement::Search(s) => {
                        assert_eq!(s.query, Literal::Vector(vec![1.0, 2.0, 3.0]))
                    }
                    other => panic!("expected Search, got {other:?}"),
                }
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn parses_returning_list() {
        // The list is recorded IN SOURCE ORDER and left unclassified: at parse
        // time `id`, `score` and `title` are all just identifiers. Deciding
        // which are pseudo-columns needs the schema, so it is the binder's.
        match ok("SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING id, score;") {
            Statement::Search(s) => {
                assert_eq!(s.projection, Some(cols(&["id", "score"])));
                // ...and the rest of the statement is unaffected.
                assert_eq!(s.k, 5);
                assert_eq!(s.collection, "docs");
            }
            other => panic!("expected Search, got {other:?}"),
        }
        // Order is preserved, not normalized.
        match ok("SEARCH TOP 1 NEAREST TO [1.0] FROM docs RETURNING score, title, id;") {
            Statement::Search(s) => assert_eq!(s.projection, Some(cols(&["score", "title", "id"]))),
            other => panic!("expected Search, got {other:?}"),
        }
        // A single item is a list of one, not a special case.
        match ok("SEARCH TOP 1 NEAREST TO [1.0] FROM docs RETURNING title;") {
            Statement::Search(s) => assert_eq!(s.projection, Some(cols(&["title"]))),
            other => panic!("expected Search, got {other:?}"),
        }
        // RETURNING is case-insensitive like every other keyword.
        assert_eq!(
            ok("search top 1 nearest to [1.0] from docs returning id;"),
            ok("SEARCH TOP 1 NEAREST TO [1.0] FROM docs RETURNING id;")
        );
    }

    #[test]
    fn no_returning_is_none() {
        // Absence is `None`, NOT an empty list — commit 2's default projection
        // (every scalar column) is what `None` means, and an empty `Some(vec![])`
        // would be a different statement entirely.
        match ok("SEARCH TOP 5 NEAREST TO [1.0] FROM docs;") {
            Statement::Search(s) => assert_eq!(s.projection, None),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn malformed_returning_is_a_parse_error() {
        for src in [
            // RETURNING with nothing after it
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING;",
            // trailing comma
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING id,;",
            // leading comma
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING , id;",
            // a literal where a name belongs
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING 5;",
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs RETURNING 'title';",
            // RETURNING before FROM
            "SEARCH TOP 5 NEAREST TO [1.0] RETURNING id FROM docs;",
        ] {
            let _ = err(src);
        }
    }

    #[test]
    fn malformed_search_is_a_parse_error() {
        // One case per token the grammar requires. Each is a clean `ParseError`
        // — `err` panics if any of these parses OR panics.
        for src in [
            // missing TOP
            "SEARCH 5 NEAREST TO [1.0] FROM docs;",
            // non-integer TOP
            "SEARCH TOP many NEAREST TO [1.0] FROM docs;",
            "SEARCH TOP 1.5 NEAREST TO [1.0] FROM docs;",
            "SEARCH TOP 'five' NEAREST TO [1.0] FROM docs;",
            // missing NEAREST
            "SEARCH TOP 5 TO [1.0] FROM docs;",
            // missing TO
            "SEARCH TOP 5 NEAREST [1.0] FROM docs;",
            // missing the query vector
            "SEARCH TOP 5 NEAREST TO FROM docs;",
            // a non-vector where the query belongs
            "SEARCH TOP 5 NEAREST TO 'not a vector' FROM docs;",
            "SEARCH TOP 5 NEAREST TO 42 FROM docs;",
            // unterminated vector
            "SEARCH TOP 5 NEAREST TO [1.0, 2.0 FROM docs;",
            // missing FROM
            "SEARCH TOP 5 NEAREST TO [1.0] docs;",
            // missing the collection
            "SEARCH TOP 5 NEAREST TO [1.0] FROM;",
            // missing the terminator
            "SEARCH TOP 5 NEAREST TO [1.0] FROM docs",
            // nothing after the keyword
            "SEARCH;",
            "SEARCH",
        ] {
            let _ = err(src);
        }
    }

    // -- SELECT ------------------------------------------------------------

    #[test]
    fn select_with_column_list() {
        assert_eq!(
            ok("SELECT x, y FROM docs;"),
            Statement::Select(SelectStmt {
                projection: Projection::Columns(cols(&["x", "y"])),
                from: "docs".to_string(),
                filter: None,
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
                filter: None,
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
                filter: None,
            })
        );
    }

    // -- WHERE / expressions -----------------------------------------------

    /// The filter of a parsed `SELECT`, for tests that only care about it.
    fn filter_of(sql: &str) -> Option<Expr> {
        match ok(sql) {
            Statement::Select(s) => s.filter,
            other => panic!("expected a SELECT, got {other:?}"),
        }
    }

    fn col(name: &str) -> Box<Expr> {
        Box::new(Expr::Column(name.to_string()))
    }

    fn int(n: i64) -> Box<Expr> {
        Box::new(Expr::Literal(Literal::Int(n)))
    }

    fn cmp(left: Box<Expr>, op: CompareOp, right: Box<Expr>) -> Box<Expr> {
        Box::new(Expr::Compare { left, op, right })
    }

    #[test]
    fn a_statement_without_where_has_no_filter() {
        assert_eq!(filter_of("SELECT x FROM docs;"), None);
    }

    #[test]
    fn a_simple_comparison_parses() {
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE z < 2;"),
            Some(*cmp(col("z"), CompareOp::Lt, int(2)))
        );
    }

    #[test]
    fn every_comparison_operator_parses() {
        for (sql, op) in [
            ("<", CompareOp::Lt),
            ("<=", CompareOp::Le),
            (">", CompareOp::Gt),
            (">=", CompareOp::Ge),
            ("=", CompareOp::Eq),
            ("!=", CompareOp::Ne),
            ("<>", CompareOp::Ne),
        ] {
            let stmt = format!("SELECT x FROM docs WHERE z {sql} 2;");
            assert_eq!(
                filter_of(&stmt),
                Some(*cmp(col("z"), op, int(2))),
                "{sql} did not parse as {op:?}"
            );
        }
    }

    #[test]
    fn and_binds_tighter_than_or() {
        // THE precedence test. `a = 1 OR b = 2 AND c = 3` must group as
        // `a = 1 OR (b = 2 AND c = 3)`. Getting this backwards changes which
        // rows come back, and silently — both readings are valid predicates.
        let expected = Expr::Or(
            cmp(col("a"), CompareOp::Eq, int(1)),
            Box::new(Expr::And(
                cmp(col("b"), CompareOp::Eq, int(2)),
                cmp(col("c"), CompareOp::Eq, int(3)),
            )),
        );
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE a = 1 OR b = 2 AND c = 3;"),
            Some(expected)
        );
    }

    #[test]
    fn parentheses_override_precedence() {
        let expected = Expr::And(
            Box::new(Expr::Or(
                cmp(col("a"), CompareOp::Eq, int(1)),
                cmp(col("b"), CompareOp::Eq, int(2)),
            )),
            cmp(col("c"), CompareOp::Eq, int(3)),
        );
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE (a = 1 OR b = 2) AND c = 3;"),
            Some(expected)
        );
    }

    #[test]
    fn and_and_or_are_left_associative() {
        // `a AND b AND c` is `(a AND b) AND c`. Right-association would give
        // the same rows for AND/OR, but the plan shape differs and the tests
        // downstream assert on tree structure.
        let expected = Expr::And(
            Box::new(Expr::And(
                cmp(col("a"), CompareOp::Eq, int(1)),
                cmp(col("b"), CompareOp::Eq, int(2)),
            )),
            cmp(col("c"), CompareOp::Eq, int(3)),
        );
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE a = 1 AND b = 2 AND c = 3;"),
            Some(expected)
        );
    }

    #[test]
    fn a_negative_comparand_parses_as_a_negative_literal() {
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE z < -2;"),
            Some(*cmp(col("z"), CompareOp::Lt, int(-2)))
        );
    }

    #[test]
    fn a_flipped_comparison_parses_as_written() {
        // `2 > z` is legal syntax; NORMALIZING it to `z < 2` is the binder's
        // job, because only the binder knows which side names a column.
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE 2 > z;"),
            Some(*cmp(int(2), CompareOp::Gt, col("z")))
        );
    }

    #[test]
    fn a_string_comparand_parses() {
        assert_eq!(
            filter_of("SELECT x FROM docs WHERE author = 'alice';"),
            Some(*cmp(
                col("author"),
                CompareOp::Eq,
                Box::new(Expr::Literal(Literal::Str("alice".to_string())))
            ))
        );
    }

    #[test]
    fn chained_comparisons_are_rejected() {
        // `a < b < c` means nothing in V-SQL. Parsing it as `(a < b) < c`
        // would accept it and produce a predicate no one asked for.
        // (`err` panics unless the parse fails, so calling it IS the assertion.)
        err("SELECT x FROM docs WHERE a < b < c;");
    }

    #[test]
    fn a_where_clause_missing_its_expression_is_an_error() {
        err("SELECT x FROM docs WHERE;");
        err("SELECT x FROM docs WHERE z <;");
        err("SELECT x FROM docs WHERE (z < 2;");
    }

    #[test]
    fn search_takes_where_before_returning() {
        // Clause ORDER is fixed by the spec (CLAUDE.md §4).
        let stmt = ok("SEARCH TOP 5 NEAREST TO [0.1, 0.2] FROM docs WHERE z < 2 RETURNING id, score;");
        let Statement::Search(search) = stmt else {
            panic!("expected a SEARCH");
        };
        assert_eq!(search.filter, Some(*cmp(col("z"), CompareOp::Lt, int(2))));
        assert_eq!(
            search.projection,
            Some(vec!["id".to_string(), "score".to_string()])
        );

        // WHERE without RETURNING, and the reverse order rejected.
        let stmt = ok("SEARCH TOP 5 NEAREST TO [0.1, 0.2] FROM docs WHERE z < 2;");
        let Statement::Search(search) = stmt else {
            panic!("expected a SEARCH");
        };
        assert!(search.filter.is_some() && search.projection.is_none());
        // RETURNING before WHERE is not the spec's order.
        err("SEARCH TOP 5 NEAREST TO [0.1, 0.2] FROM docs RETURNING id WHERE z < 2;");
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
                filter: None,
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
        let stmt =
            ok("CREATE COLLECTION c (a VECTOR(4), b TEXT, d INT, e FLOAT) WITH (capacity = 1);");
        assert_eq!(
            create_columns(&stmt),
            vec![
                ColumnDef {
                    name: "a".to_string(),
                    ty: ColumnType::Vector(4)
                },
                ColumnDef {
                    name: "b".to_string(),
                    ty: ColumnType::Text
                },
                ColumnDef {
                    name: "d".to_string(),
                    ty: ColumnType::Int
                },
                ColumnDef {
                    name: "e".to_string(),
                    ty: ColumnType::Float
                },
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
            vec![ColumnDef {
                name: "author".to_string(),
                ty: ColumnType::Text
            }]
        );
    }

    #[test]
    fn column_named_vector_does_not_collide_with_vector_type() {
        // THE collision test: a column literally named `vector`, typed VECTOR.
        let stmt = ok("CREATE COLLECTION docs (vector VECTOR(768)) WITH (capacity = 1);");
        assert_eq!(
            create_columns(&stmt),
            vec![ColumnDef {
                name: "vector".to_string(),
                ty: ColumnType::Vector(768)
            }]
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
                vec![CollectionOption {
                    name: "capacity".to_string(),
                    value: 1000000
                }]
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
        assert_eq!(
            insert_values(&stmt),
            vec![Literal::Vector(vec![0.1, 0.2, 0.3])]
        );
    }

    #[test]
    fn negative_vector_literal_applies_signs() {
        // proves the lexer's separate-Minus decision cashes out end to end.
        let stmt = ok("INSERT INTO docs (v) VALUES ([-0.1, 0.2, -0.3]);");
        assert_eq!(
            insert_values(&stmt),
            vec![Literal::Vector(vec![-0.1, 0.2, -0.3])]
        );
    }

    #[test]
    fn integer_vector_elements_coerce_to_f32() {
        let stmt = ok("INSERT INTO docs (v) VALUES ([1, 0, 0]);");
        assert_eq!(
            insert_values(&stmt),
            vec![Literal::Vector(vec![1.0, 0.0, 0.0])]
        );
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
                filter: None,
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
                    ColumnDef {
                        name: "vector".to_string(),
                        ty: ColumnType::Vector(768)
                    },
                    ColumnDef {
                        name: "author".to_string(),
                        ty: ColumnType::Text
                    },
                    ColumnDef {
                        name: "title".to_string(),
                        ty: ColumnType::Text
                    },
                    ColumnDef {
                        name: "published_at".to_string(),
                        ty: ColumnType::Int
                    },
                ],
                options: vec![CollectionOption {
                    name: "capacity".to_string(),
                    value: 1000000
                }],
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
