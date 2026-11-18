//! Flats: a tiny, embeddable vector database.
//!
//! Exact (brute-force) similarity search over `mmap`'d vectors, a write-ahead
//! log for durability, metadata filters through roaring bitmaps, and a small
//! SQL dialect, V-SQL, to drive it all. A library, not a server: open a
//! directory and query it from the same process.
//!
//! # Quickstart
//!
//! ```
//! use flats::{Db, DbOptions, RegValue};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tmp = tempfile::tempdir()?;
//! # let path = tmp.path().join("my-db");
//! let db = Db::open(&path, &[], DbOptions::default())?;
//!
//! db.execute(
//!     "CREATE COLLECTION docs (
//!          id TEXT PRIMARY KEY,
//!          vector VECTOR(3),
//!          year INT
//!      ) WITH (capacity = 10000);",
//! )?;
//!
//! // One statement, one durable record: both rows land, or neither does.
//! db.execute(
//!     "INSERT INTO docs (id, vector, year) VALUES
//!          ('a', [0.1, 0.9, 0.0], 2023),
//!          ('b', [0.8, 0.1, 1e-5], 2024);",
//! )?;
//!
//! // Top-k by dot product, filtered before ranking, so `k` counts matches.
//! let hits = db.execute(
//!     "SEARCH TOP 1 NEAREST TO [1, 0, 0] FROM docs WHERE year >= 2024 RETURNING id, score;",
//! )?;
//! let rows = hits.collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(rows.len(), 1);
//! match rows[0].0.as_slice() {
//!     [RegValue::Str(id), RegValue::Real(score)] => println!("{id}: {score}"),
//!     other => panic!("unexpected row {other:?}"),
//! }
//! # assert!(matches!(rows[0].0.as_slice(), [RegValue::Str(id), _] if id == "b"));
//!
//! db.close()?; // checkpoints; dropping without it is crash-safe too
//! # Ok(())
//! # }
//! ```
//!
//! Bulk loads skip SQL text entirely with [`Db::insert_many`]; a consistent
//! copy is one [`Db::backup`] away. Python, Node.js and C bindings live in
//! `sdk/` and wrap the same engine through one C ABI.
//!
//! # The API
//!
//! The supported API is what this crate root re-exports: [`Db`] and its
//! options, [`Db::execute`] for V-SQL with the [`RowStream`] it returns, and
//! the schema and value types those speak. Every module behind them is
//! private (see the `internals` feature in `Cargo.toml`). Design scope and
//! guarantees are in `CLAUDE.md` and `ARCHITECTURE.md`.

#![forbid(unsafe_op_in_unsafe_fn)]
// No `unwrap()`/`expect()` in library code (CLAUDE.md §5): bubble a `Result`.
// Tests are exempt. The few sites that genuinely cannot fail carry a local
// `#[allow]` with the reason.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
// Without `internals`, the modules are private, so the module-level
// convenience re-exports and the few items only this crate's own tests and
// benches reach (they build WITH `internals`) look unused. That is the shape of
// the split, not dead code: before it, every item was public and none of these
// warnings could fire either.
#![cfg_attr(not(feature = "internals"), allow(unused_imports, dead_code))]
// CLAUDE.md §5: the public API is documented. Checked on the default build —
// the one a downstream crate sees.
#![cfg_attr(not(feature = "internals"), warn(missing_docs))]

// ---------------------------------------------------------------------------
// Modules: all internal.
//
// The supported API is the set of re-exports below and nothing else. Every
// module is private unless the `internals` feature is on, in which case it is
// public but `#[doc(hidden)]` — reachable for this crate's own tests and
// benches (they enable it through a dev-dependency on the crate itself), and
// exempt from semver. A downstream crate on default features cannot name
// `flats::wal::…` at all; `tests/public_api.rs` holds that line.
// ---------------------------------------------------------------------------

macro_rules! internal_modules {
    ($($name:ident),* $(,)?) => {$(
        #[cfg(feature = "internals")]
        #[doc(hidden)]
        pub mod $name;
        #[cfg(not(feature = "internals"))]
        mod $name;
    )*};
}

internal_modules!(
    compiler, engine, error, index, metadata, query, simd, sql, vm, wal
);

// ---------------------------------------------------------------------------
// The public API.
// ---------------------------------------------------------------------------

// The database and its options.
pub use crate::engine::cursor::Cursor;
pub use crate::engine::{
    CollectionConfig, CollectionId, CompactionOptions, Db, DbOptions, MAX_INSTALL_BYTES,
};
pub use crate::error::{Error, Result};
// What `Db::insert`, `Db::insert_many` and `Db::search` hand back.
pub use crate::index::index::{Ordinal, SearchResult};
// Schemas and the values rows are made of.
pub use crate::metadata::common::ColumnId;
pub use crate::metadata::{ColumnSpec, ColumnType, RangeOp, Row, Schema, Value};
// The query pipeline's boundary types. `query::Error` is deliberately NOT
// re-exported unqualified — `Error` above is the ENGINE's, and two types with
// that name at the crate root would be exactly the confusion the stage-local
// error split exists to prevent (CLAUDE.md §5).
pub use crate::query::{Error as QueryError, RowStream};
// One row of a `RowStream`, and its cells.
pub use crate::vm::exec::{OutputRow, RegValue};
pub use simd::dot;
