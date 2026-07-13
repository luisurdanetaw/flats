//! Query frontend (Phase 7).
//!
//! First stage of the query pipeline (lexer → parser → logical plan → bytecode
//! → VM → optimizer). [`lexer`] turns a V-SQL string into a token stream; the
//! public token/error types are re-exported here so later stages import them
//! from `crate::lexer` without reaching into the submodule.

pub mod lexer;

pub use lexer::{LexError, LexErrorKind, Lexer, Span, SpannedToken, Token};
