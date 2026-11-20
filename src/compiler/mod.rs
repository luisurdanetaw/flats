pub mod bytecode;
// `compiler::compiler` trips clippy::module_inception. Allowed rather than renamed:
// the path is public and imported by tests/benches; revisit when the public
// surface is locked down.
#[allow(clippy::module_inception)]
pub mod compiler;
pub mod constants;
pub mod schema;

pub use bytecode::{Addr, Cursor, Op, Program, Reg, ValidateError};
pub use compiler::{CompileError, compile};
pub use constants::{Const, ConstId, ConstPool};
pub use schema::{SchemaError, to_metadata_schema};
