//! Query frontend — the lexer (Phase 7a).
//!
//! Front-to-back the query layer is: **lexer → parser → logical plan →
//! bytecode compiler → VM → optimizer**. This module is the first stage: it
//! turns a raw V-SQL string into a flat stream of [`SpannedToken`]s. It knows
//! nothing about grammar, statements, or types — that is the parser's job.
//!
//! # Scope (bootstrap)
//!
//! The token set covers exactly the three bootstrap statements the query layer
//! is built against and nothing more:
//!
//! ```text
//! CREATE COLLECTION docs (vector VECTOR(768), author TEXT, ...) WITH (capacity = 1000000);
//! SELECT x, y FROM docs;
//! INSERT INTO docs (vector, ...) VALUES ([0.1, 0.2, 0.3], 'alice', 'My doc', 1700000000);
//! ```
//!
//! It is *minimal in statement support*, not in engineering: the scanner is a
//! single character-driven loop with clear extension points. Adding `WHERE`,
//! `AND`/`OR`, comparison operators, or `SEARCH TOP k NEAREST` later is one new
//! keyword-table arm or one new punctuation match arm — never a rework. Those
//! spots are marked `EXTEND:`.
//!
//! # Two intentional design decisions
//!
//! 1. **Type names (`VECTOR`/`INT`/`TEXT`/`FLOAT`) lex as [`Token::Ident`], not
//!    keywords.** A column may legitimately be named `vector` (the bootstrap
//!    schema does exactly that); case-insensitive keyword matching would then
//!    collide the column with the `VECTOR` type. The parser resolves
//!    type-vs-name *by position* later, so reserving type names is a
//!    parser/semantic decision, not a lexer one.
//! 2. **`-` lexes as a standalone [`Token::Minus`].** Sign is applied by the
//!    parser, not folded into number literals. So `[-0.1, 0.2]` tokenizes as
//!    `Minus, FloatLit(0.1), Comma, FloatLit(0.2)`.
//!
//! Both decisions are pinned by the test suite below; do not "fix" them.

use std::fmt;
use std::ops::Index;

/// A single lexical token. Covers only what the three bootstrap statements
/// need (see the module docs). Literals carry their decoded value; identifiers
/// preserve their source spelling (and case).
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // -- reserved keywords (matched case-insensitively) ---------------------
    /// `CREATE`
    Create,
    /// `COLLECTION`
    Collection,
    /// `WITH`
    With,
    /// `SELECT`
    Select,
    /// `FROM`
    From,
    /// `INSERT`
    Insert,
    /// `INTO`
    Into,
    /// `VALUES`
    Values,
    // EXTEND: future reserved keywords (Where, And, Or, Search, Top, Nearest,
    // To, Delete, Update, Set, Returning, …) get a variant here AND an arm in
    // `keyword()` — nothing else changes.

    // -- identifiers & literals ---------------------------------------------
    /// An identifier. Also carries *type names* (`VECTOR`, `INT`, `TEXT`,
    /// `FLOAT`) — see design decision (1) in the module docs. Source case is
    /// preserved.
    Ident(String),
    /// An integer literal, e.g. `768`, that fits in `i64`.
    IntLit(i64),
    /// A floating-point literal, e.g. `0.1`. Always unsigned at this layer —
    /// see design decision (2).
    FloatLit(f64),
    /// A single-quoted string literal, quotes stripped and `''` unescaped to a
    /// single `'`.
    StrLit(String),

    // -- punctuation --------------------------------------------------------
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `=`
    Eq,
    /// `-` (a standalone token; sign is a parser concern — decision (2)).
    Minus,
    // EXTEND: future operators (Star, Lt, Gt, Le, Ge, Ne, …) get a variant
    // here AND a match arm in `next_token` — nothing else changes.

    /// End of input. Always the final token in a successful stream.
    Eof,
}

/// A half-open byte range `[start, end)` into the source string. Indexing a
/// `str` with a `Span` yields exactly the matched slice (see the `Index` impl).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset of the first byte of the token (inclusive).
    pub start: usize,
    /// Byte offset one past the last byte of the token (exclusive).
    pub end: usize,
}

/// A [`Token`] paired with the [`Span`] it was scanned from.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    /// The token.
    pub token: Token,
    /// Its source span.
    pub span: Span,
}

/// `&src[span]` yields the exact source slice the token was scanned from.
impl Index<Span> for str {
    type Output = str;

    fn index(&self, span: Span) -> &str {
        &self[span.start..span.end]
    }
}

/// What went wrong while lexing, and where.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    /// The category of failure.
    pub kind: LexErrorKind,
    /// Byte offset into the source at which the error was detected.
    pub pos: usize,
}

impl LexError {
    /// Resolve [`pos`](Self::pos) to a 1-based `(line, column)` for diagnostics.
    /// Column counts characters (not bytes) within the offending line.
    pub fn line_col(&self, src: &str) -> (usize, usize) {
        let mut line = 1;
        let mut col = 1;
        for (i, ch) in src.char_indices() {
            if i >= self.pos {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LexErrorKind::UnexpectedChar(c) => {
                write!(f, "unexpected character {c:?} at byte {}", self.pos)
            }
            LexErrorKind::UnterminatedString => {
                write!(f, "unterminated string literal at byte {}", self.pos)
            }
            LexErrorKind::InvalidNumber(s) => {
                write!(f, "invalid number literal {s:?} at byte {}", self.pos)
            }
        }
    }
}

impl std::error::Error for LexError {}

/// The category of a [`LexError`].
#[derive(Debug, Clone, PartialEq)]
pub enum LexErrorKind {
    /// A character that cannot begin any token (e.g. `@`).
    UnexpectedChar(char),
    /// A `'`-opened string that reached end-of-input with no closing quote.
    UnterminatedString,
    /// A numeric literal that could not be parsed (e.g. an integer that
    /// overflows `i64`). Carries the offending text.
    InvalidNumber(String),
}

/// Resolve a reserved keyword by its (case-insensitive) spelling. Returns
/// `None` for ordinary identifiers — including type names like `VECTOR`/`INT`,
/// which are deliberately NOT keywords (design decision (1)).
///
/// EXTEND: this table is the single place to add a future reserved keyword.
fn keyword(word: &str) -> Option<Token> {
    match word.to_ascii_lowercase().as_str() {
        "create" => Some(Token::Create),
        "collection" => Some(Token::Collection),
        "with" => Some(Token::With),
        "select" => Some(Token::Select),
        "from" => Some(Token::From),
        "insert" => Some(Token::Insert),
        "into" => Some(Token::Into),
        "values" => Some(Token::Values),
        // EXTEND: future reserved keywords here.
        _ => None,
    }
}

/// The lexer: a cursor over a source string that yields [`SpannedToken`]s.
///
/// Construct with [`Lexer::new`], then either drain the whole stream with
/// [`tokenize`](Self::tokenize) or pull one token at a time with
/// [`next_token`](Self::next_token).
pub struct Lexer<'a> {
    /// The full source, retained for slicing spans and reporting positions.
    src: &'a str,
    /// Byte offset of the next unconsumed character.
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Create a lexer positioned at the start of `src`.
    pub fn new(src: &'a str) -> Self {
        Lexer { src, pos: 0 }
    }

    /// Consume the whole input, returning every token up to and including a
    /// trailing [`Token::Eof`]. Errors on the first malformed token.
    pub fn tokenize(mut self) -> Result<Vec<SpannedToken>, LexError> {
        let mut out = Vec::new();
        loop {
            let st = self.next_token()?;
            let done = st.token == Token::Eof;
            out.push(st);
            if done {
                return Ok(out);
            }
        }
    }

    /// Scan and return the next token. Returns [`Token::Eof`] once the input is
    /// exhausted (with a zero-width span at end of input) and on every
    /// subsequent call.
    ///
    /// Scanning is byte-driven: every delimiter is single-byte ASCII and `'`
    /// (0x27) can never occur inside a UTF-8 multibyte sequence, so token
    /// boundaries are found by byte without ever splitting a character.
    pub fn next_token(&mut self) -> Result<SpannedToken, LexError> {
        self.skip_trivia();
        let start = self.pos;
        let b = match self.src.as_bytes().get(start) {
            None => {
                return Ok(SpannedToken {
                    token: Token::Eof,
                    span: Span { start, end: start },
                });
            }
            Some(&b) => b,
        };
        match b {
            b'(' => Ok(self.single(Token::LParen, start)),
            b')' => Ok(self.single(Token::RParen, start)),
            b'[' => Ok(self.single(Token::LBracket, start)),
            b']' => Ok(self.single(Token::RBracket, start)),
            b',' => Ok(self.single(Token::Comma, start)),
            b';' => Ok(self.single(Token::Semicolon, start)),
            b'=' => Ok(self.single(Token::Eq, start)),
            // A `-` that reaches here is a genuine Minus: maximal munch means a
            // `--` line comment was already eaten by `skip_trivia`. Sign is the
            // parser's job (design decision 2).
            b'-' => Ok(self.single(Token::Minus, start)),
            // EXTEND: new single-char operators (Star `*`, Lt `<`, Gt `>`, …)
            // get a match arm here plus a `Token` variant.
            b'\'' => self.scan_string(start),
            _ if b.is_ascii_digit() => self.scan_number(start),
            _ if b == b'_' || b.is_ascii_alphabetic() => Ok(self.scan_ident(start)),
            _ => {
                // Report the whole offending character, not a stray byte, so a
                // non-ASCII intruder is named correctly.
                let ch = self.src[start..].chars().next().unwrap_or(b as char);
                Err(LexError {
                    kind: LexErrorKind::UnexpectedChar(ch),
                    pos: start,
                })
            }
        }
    }

    // -- private scanner helpers -------------------------------------------

    /// Advance past a single-byte token and span it `[start, pos)`.
    fn single(&mut self, token: Token, start: usize) -> SpannedToken {
        self.pos += 1;
        SpannedToken {
            token,
            span: Span { start, end: self.pos },
        }
    }

    /// Skip whitespace and `-- …` line comments (to end of line). Loops so runs
    /// of mixed trivia are consumed in one call.
    fn skip_trivia(&mut self) {
        let bytes = self.src.as_bytes();
        loop {
            match bytes.get(self.pos) {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.pos += 1,
                // `--` beats a lone `-` (maximal munch): a line comment runs to
                // the newline (or EOF).
                Some(b'-') if bytes.get(self.pos + 1) == Some(&b'-') => {
                    self.pos += 2;
                    while let Some(&c) = bytes.get(self.pos) {
                        if c == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                    // EXTEND: other trivia (block comments, …) go here.
                }
                _ => break,
            }
        }
    }

    /// Scan `[A-Za-z_][A-Za-z0-9_]*`, then resolve it against the reserved
    /// keyword set case-insensitively. A miss is an [`Token::Ident`] carrying
    /// the original source case — so type names (`VECTOR`/`INT`/…) stay idents
    /// (design decision 1).
    fn scan_ident(&mut self, start: usize) -> SpannedToken {
        let bytes = self.src.as_bytes();
        while let Some(&c) = bytes.get(self.pos) {
            if c == b'_' || c.is_ascii_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        let token = keyword(text).unwrap_or_else(|| Token::Ident(text.to_string()));
        SpannedToken {
            token,
            span: Span { start, end: self.pos },
        }
    }

    /// Scan a maximal digit run; a `.` joins the number ONLY if a digit follows
    /// (so `768.` is `IntLit(768)` and leaves the dot). Parse failure — e.g. an
    /// `i64` overflow — becomes [`LexErrorKind::InvalidNumber`].
    fn scan_number(&mut self, start: usize) -> Result<SpannedToken, LexError> {
        let bytes = self.src.as_bytes();
        while matches!(bytes.get(self.pos), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        let mut is_float = false;
        if bytes.get(self.pos) == Some(&b'.')
            && matches!(bytes.get(self.pos + 1), Some(c) if c.is_ascii_digit())
        {
            is_float = true;
            self.pos += 1; // consume '.'
            while matches!(bytes.get(self.pos), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = &self.src[start..self.pos];
        // The two parses have different error types; normalize both to
        // InvalidNumber(text) here rather than trying to unify them.
        let invalid = || LexError {
            kind: LexErrorKind::InvalidNumber(text.to_string()),
            pos: start,
        };
        let token = if is_float {
            Token::FloatLit(text.parse::<f64>().map_err(|_| invalid())?)
        } else {
            Token::IntLit(text.parse::<i64>().map_err(|_| invalid())?)
        };
        Ok(SpannedToken {
            token,
            span: Span { start, end: self.pos },
        })
    }

    /// Scan a `'`-quoted string. A doubled `''` unescapes to one `'` and the
    /// string continues; EOF before the closing quote is
    /// [`LexErrorKind::UnterminatedString`]. Content is copied as whole
    /// byte-segments between quotes — each break is on the ASCII byte `'`, never
    /// inside a multibyte char, so UTF-8 round-trips.
    fn scan_string(&mut self, start: usize) -> Result<SpannedToken, LexError> {
        let bytes = self.src.as_bytes();
        self.pos += 1; // opening quote
        let mut value = String::new();
        let mut segment_start = self.pos;
        loop {
            match bytes.get(self.pos) {
                None => {
                    return Err(LexError {
                        kind: LexErrorKind::UnterminatedString,
                        pos: start,
                    });
                }
                Some(&b'\'') => {
                    value.push_str(&self.src[segment_start..self.pos]);
                    if bytes.get(self.pos + 1) == Some(&b'\'') {
                        value.push('\''); // escaped quote; string continues
                        self.pos += 2;
                        segment_start = self.pos;
                    } else {
                        self.pos += 1; // closing quote
                        return Ok(SpannedToken {
                            token: Token::StrLit(value),
                            span: Span { start, end: self.pos },
                        });
                    }
                }
                Some(_) => self.pos += 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lex `src`, assert the stream ends in `Eof`, drop the `Eof`, and return
    /// the bare `Token`s for readable assertions.
    fn lex(src: &str) -> Vec<Token> {
        let spanned = Lexer::new(src).tokenize().expect("expected a clean lex");
        let mut tokens: Vec<Token> = spanned.into_iter().map(|st| st.token).collect();
        assert_eq!(
            tokens.last(),
            Some(&Token::Eof),
            "token stream must end in Eof"
        );
        tokens.pop();
        tokens
    }

    /// Lex `src` expecting failure; return the error.
    fn lex_err(src: &str) -> LexError {
        match Lexer::new(src).tokenize() {
            Ok(toks) => panic!("expected a lex error, got {toks:?}"),
            Err(e) => e,
        }
    }

    fn ident(s: &str) -> Token {
        Token::Ident(s.to_string())
    }

    // -- unit tests ---------------------------------------------------------

    #[test]
    fn keywords_are_case_insensitive() {
        assert_eq!(lex("select"), vec![Token::Select]);
        assert_eq!(lex("SELECT"), vec![Token::Select]);
        assert_eq!(lex("SeLeCt"), vec![Token::Select]);
    }

    #[test]
    fn type_names_are_idents_not_keywords() {
        // Design decision (1): VECTOR/INT/TEXT/FLOAT are identifiers so a column
        // can be named `vector` without colliding with the type.
        assert_eq!(lex("vector"), vec![ident("vector")]);
        assert_eq!(lex("VECTOR"), vec![ident("VECTOR")]);
        assert_eq!(lex("INT"), vec![ident("INT")]);
        assert_eq!(lex("text"), vec![ident("text")]);
    }

    #[test]
    fn identifiers_preserve_case_and_allow_underscores() {
        assert_eq!(lex("published_at"), vec![ident("published_at")]);
        assert_eq!(lex("docs"), vec![ident("docs")]);
    }

    #[test]
    fn punctuation_maps_without_surrounding_whitespace() {
        assert_eq!(
            lex("VECTOR(768),"),
            vec![
                ident("VECTOR"),
                Token::LParen,
                Token::IntLit(768),
                Token::RParen,
                Token::Comma,
            ]
        );
    }

    #[test]
    fn integers_versus_floats() {
        assert_eq!(lex("768"), vec![Token::IntLit(768)]);
        assert_eq!(lex("1700000000"), vec![Token::IntLit(1700000000)]);
        assert_eq!(lex("0.1"), vec![Token::FloatLit(0.1)]);
    }

    #[test]
    fn trailing_dot_is_not_consumed_into_the_number() {
        // "768." is IntLit(768) followed by an UnexpectedChar('.') — the '.' is
        // only part of a float when a digit follows it.
        let mut lx = Lexer::new("768.");
        assert_eq!(
            lx.next_token().expect("first token").token,
            Token::IntLit(768)
        );
        match lx.next_token() {
            Err(e) => assert_eq!(e.kind, LexErrorKind::UnexpectedChar('.')),
            Ok(t) => panic!("expected UnexpectedChar('.'), got {t:?}"),
        }
    }

    #[test]
    fn negative_number_is_minus_then_number() {
        // Design decision (2): sign is a parser concern.
        assert_eq!(lex("-0.2"), vec![Token::Minus, Token::FloatLit(0.2)]);
    }

    #[test]
    fn negative_vector_literal() {
        assert_eq!(
            lex("[-0.1, 0.2, -0.3]"),
            vec![
                Token::LBracket,
                Token::Minus,
                Token::FloatLit(0.1),
                Token::Comma,
                Token::FloatLit(0.2),
                Token::Comma,
                Token::Minus,
                Token::FloatLit(0.3),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn integer_overflow_is_invalid_number() {
        let err = lex_err("99999999999999999999999"); // 23 digits, overflows i64
        assert!(
            matches!(err.kind, LexErrorKind::InvalidNumber(_)),
            "expected InvalidNumber, got {:?}",
            err.kind
        );
    }

    #[test]
    fn strings_basic_and_with_spaces() {
        assert_eq!(lex("'alice'"), vec![Token::StrLit("alice".to_string())]);
        assert_eq!(lex("'My doc'"), vec![Token::StrLit("My doc".to_string())]);
    }

    #[test]
    fn string_doubled_quote_escape() {
        assert_eq!(lex("'it''s'"), vec![Token::StrLit("it's".to_string())]);
    }

    #[test]
    fn string_round_trips_utf8() {
        assert_eq!(lex("'café ☕'"), vec![Token::StrLit("café ☕".to_string())]);
    }

    #[test]
    fn unterminated_string_errors() {
        let err = lex_err("'abc");
        assert_eq!(err.kind, LexErrorKind::UnterminatedString);
    }

    #[test]
    fn unexpected_char_reports_line_col() {
        let src = "SELECT @ FROM docs";
        let err = lex_err(src);
        assert_eq!(err.kind, LexErrorKind::UnexpectedChar('@'));
        assert_eq!(err.line_col(src), (1, 8));
    }

    #[test]
    fn spans_index_back_into_source() {
        let src = "SELECT x";
        let toks = Lexer::new(src).tokenize().expect("expected a clean lex");
        assert_eq!(toks[0].token, Token::Select);
        assert_eq!(toks[0].span, Span { start: 0, end: 6 });
        assert_eq!(toks[1].token, ident("x"));
        assert_eq!(toks[1].span, Span { start: 7, end: 8 });
        assert_eq!(&src[toks[1].span], "x");
    }

    // -- integration tests: exact full streams (minus Eof) ------------------

    #[test]
    fn select_statement() {
        assert_eq!(
            lex("SELECT x, y FROM docs;"),
            vec![
                Token::Select,
                ident("x"),
                Token::Comma,
                ident("y"),
                Token::From,
                ident("docs"),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn create_collection_statement() {
        let src = "CREATE COLLECTION docs (\n\
                   \x20   vector VECTOR(768),\n\
                   \x20   author TEXT,\n\
                   \x20   title TEXT,\n\
                   \x20   published_at INT\n\
                   ) WITH (capacity = 1000000);";
        assert_eq!(
            lex(src),
            vec![
                Token::Create,
                Token::Collection,
                ident("docs"),
                Token::LParen,
                // column `vector` and type `VECTOR` are BOTH Ident (decision 1)
                ident("vector"),
                ident("VECTOR"),
                Token::LParen,
                Token::IntLit(768),
                Token::RParen,
                Token::Comma,
                ident("author"),
                ident("TEXT"),
                Token::Comma,
                ident("title"),
                ident("TEXT"),
                Token::Comma,
                ident("published_at"),
                ident("INT"),
                Token::RParen,
                Token::With,
                Token::LParen,
                ident("capacity"),
                Token::Eq,
                Token::IntLit(1000000),
                Token::RParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn insert_statement_with_vector_literal() {
        let src = "INSERT INTO docs (vector, author, title, published_at) \
                   VALUES ([0.1, 0.2, 0.3], 'alice', 'My doc', 1700000000);";
        assert_eq!(
            lex(src),
            vec![
                Token::Insert,
                Token::Into,
                ident("docs"),
                Token::LParen,
                ident("vector"),
                Token::Comma,
                ident("author"),
                Token::Comma,
                ident("title"),
                Token::Comma,
                ident("published_at"),
                Token::RParen,
                Token::Values,
                Token::LParen,
                Token::LBracket,
                Token::FloatLit(0.1),
                Token::Comma,
                Token::FloatLit(0.2),
                Token::Comma,
                Token::FloatLit(0.3),
                Token::RBracket,
                Token::Comma,
                Token::StrLit("alice".to_string()),
                Token::Comma,
                Token::StrLit("My doc".to_string()),
                Token::Comma,
                Token::IntLit(1700000000),
                Token::RParen,
                Token::Semicolon,
            ]
        );
    }
}
