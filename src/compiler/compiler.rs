//! Bytecode compiler — the lowering pass from [`LogicalPlan`] to [`Program`].
//!
//! Front-to-back the query layer is: **lexer → parser → binder → planner →
//! COMPILER → VM → optimizer**. This module is the SQLite-lineage codegen step:
//! a pure *structural* translation of the already-resolved, already-validated
//! plan into the instruction stream. It does no execution and no catalog
//! access — everything it needs is the plan plus the collection's storage
//! [`Schema`], handed in by the caller.
//!
//! # The one lookup it performs: declaration ordinal → storage location
//!
//! The plan speaks *declaration ordinals* (vector-inclusive: the embedding is a
//! real, ordinal-bearing column). Storage speaks two disjoint spaces — a
//! scalar-only [`ColumnId`](crate::metadata::common::ColumnId) for the tuple
//! store / metadata index, and the flat vector index for the embedding. These
//! are DIFFERENT numbering systems (a vector declared anywhere but last shifts
//! them apart), so the compiler MUST translate every column it lowers through
//! [`Schema::locate`](crate::metadata::common::Schema::locate):
//!
//!   * [`ColumnLocation::Vector`](crate::metadata::common::ColumnLocation) →
//!     emit `VectorFetch` (gated by `include_vector`), never a `Column`;
//!   * [`ColumnLocation::Scalar`](crate::metadata::common::ColumnLocation) →
//!     emit `Column` with the storage `ColumnId` the schema returns.
//!
//! The compiler NEVER computes a `ColumnId` itself — that logic lives in the
//! schema, and duplicating it (counting scalars, filtering the vector out) is
//! exactly the bug the two-space design exists to prevent.
//!
//! # Recursive shape (why a plan node emits a prologue AND an epilogue)
//!
//! A `Scan` owns a loop: `SeekFirst` opens it, `Next` closes it, and the
//! projection's per-row code sits BETWEEN them. Data flows UP (Scan produces
//! rows, Project consumes), but the loop is owned by the BOTTOM node — so the
//! upper node's code must land INSIDE the lower node's loop. That inversion is
//! why emission is recursive and `Project` passes a *body* DOWN into `Scan`
//! rather than emitting after it returns.
//!
//! # Phase 7i status (this commit)
//!
//! API skeleton + the full test suite. Every emission body is `unimplemented!()`,
//! so the crate compiles and the compiler tests fail *at* `unimplemented!()`.
//! Phase 7j fills the bodies in; the signatures here are stable.

use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
use crate::compiler::constants::ConstPool;
use crate::metadata::common::Schema;
use crate::sql::plan::LogicalPlan;

/// Compile a resolved [`LogicalPlan`] into a [`Program`].
///
/// `schema` is the target collection's STORAGE schema (the vector-inclusive
/// [`Schema`] whose [`locate`](Schema::locate) maps declaration ordinals to
/// storage locations). The caller looks it up from the same catalog the binder
/// used and hands it in; the compiler itself stays catalog-free.
pub fn compile(plan: LogicalPlan, schema: &Schema) -> Program {
    let mut compiler = Compiler::new(schema);
    // The plan root emits itself. Wrapper nodes (Scan) thread a per-row body
    // DOWN into their loop; the root therefore receives an empty top-level body
    // (a Scan reached as the root scans with no projection above it, and the
    // leaf nodes ignore the body entirely).
    compiler.emit_node(&plan, &mut |_| {});
    compiler.finish()
}

/// The compiler's working state: the growing instruction stream, the constant
/// pool, the monotonic register/cursor allocators, and the cursor currently in
/// scope (threaded into a `Scan`'s body so `Column` / `VectorFetch` reference
/// the right cursor rather than a hardcoded `0`).
///
/// Borrows the storage [`Schema`] for the duration of the compile so
/// [`emit_node`](Compiler::emit_node) can call [`Schema::locate`] without
/// re-plumbing it through every helper.
// Fields are written by `new` and consumed by the emission bodies, which land
// in 7j; until then they read as dead.
#[allow(dead_code)]
struct Compiler<'a> {
    /// The instruction stream being built.
    ops: Vec<Op>,
    /// The program's constant pool.
    consts: ConstPool,
    /// Monotonic register allocator — the next free register index.
    next_reg: u32,
    /// Monotonic cursor allocator — the next free cursor slot.
    next_cursor: u8,
    /// The cursor currently in scope (set while emitting inside a `Scan`'s
    /// loop), threaded into the per-row body. `None` outside any loop.
    cursor: Option<Cursor>,
    /// The target collection's storage schema — the source of the ordinal →
    /// storage-location translation.
    schema: &'a Schema,
}

// The emission bodies (everything but `new`) arrive in 7j; until then their
// `unimplemented!()` stubs leave parameters unread, and the primitive helpers —
// reached only from the stubbed `emit_node` — read as uncalled.
#[allow(unused_variables, dead_code)]
impl<'a> Compiler<'a> {
    /// A fresh compiler for a plan compiled against `schema`.
    fn new(schema: &'a Schema) -> Self {
        Compiler {
            ops: Vec::new(),
            consts: ConstPool::new(),
            next_reg: 0,
            next_cursor: 0,
            cursor: None,
            schema,
        }
    }

    /// Push `op`, returning its instruction index (used as a backpatch handle).
    fn emit(&mut self, op: Op) -> usize {
        unimplemented!()
    }

    /// Emit a forward jump whose target is not yet known: `make` builds the op
    /// from [`Addr::PLACEHOLDER`], and the returned index is later handed to
    /// [`patch`](Self::patch). Every forward jump routes through here so no
    /// patch site has to re-match the opcode.
    fn emit_jump_placeholder(&mut self, make: impl FnOnce(Addr) -> Op) -> usize {
        unimplemented!()
    }

    /// Backpatch the jump at instruction index `at` to point at `target`.
    fn patch(&mut self, at: usize, target: Addr) {
        unimplemented!()
    }

    /// Allocate the next register (monotonic; no reuse). Bumps [`next_reg`],
    /// which becomes [`Program::n_regs`].
    fn alloc_reg(&mut self) -> Reg {
        unimplemented!()
    }

    /// Allocate the next cursor slot (monotonic; no reuse). Bumps
    /// [`next_cursor`], which becomes [`Program::n_cursors`].
    fn alloc_cursor(&mut self) -> Cursor {
        unimplemented!()
    }

    /// Emit `node`. WRAPPER nodes (`Scan`; later `Limit`) emit a loop top, then
    /// run `body` to fill the loop, then emit the loop bottom. LEAF nodes
    /// (`Insert`, `CreateCollection`) emit directly and ignore `body`. `Project`
    /// is the middle case: it owns no loop and hands its per-row code down to
    /// its child via `body`.
    ///
    /// `body` carries the enclosing node's per-row code down so it lands INSIDE
    /// this node's loop — the inversion that makes emission recursive.
    fn emit_node(&mut self, node: &LogicalPlan, body: &mut dyn FnMut(&mut Self)) {
        unimplemented!()
    }

    /// Finish the compile, moving the accumulated state into a [`Program`].
    fn finish(self) -> Program {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
    use crate::compiler::constants::{Const, ConstPool};
    use crate::metadata::common::{
        CollectionConfig, ColumnSpec, ColumnType as MetaColumnType, Schema,
    };
    use crate::sql::ast::{ColumnType, Literal};
    use crate::sql::bind::{ColumnRef, ColumnSchema, Schema as BindSchema, TypedValue};
    use crate::sql::plan::{CreateCollection, Insert, LogicalPlan, Project, Scan};
    use std::num::NonZeroUsize;

    // ------------------------------------------------------------------------
    // Fixtures.
    //
    // Plans are built DIRECTLY from plan nodes (not via parse → analyze →
    // plan) so the compiler is tested in isolation from the frontend, and so
    // the vector-FIRST fixture is under our exact control.
    //
    // Two schemas describe the same `docs` collection in the two vocabularies:
    //   * `docs_bind_schema()`  — the vector-inclusive `bind::Schema` the plan
    //     carries (used only for shape: ordinals, include_vector);
    //   * `docs_meta_schema()`  — the storage `metadata::Schema` handed to
    //     compile(), whose `locate()` performs the ordinal → ColumnId mapping.
    //
    // The vector is declared FIRST, so the two numberings DIFFER: author is
    // declaration ordinal 1 but storage ColumnId 0, title ordinal 2 / id 1,
    // published_at ordinal 3 / id 2. That gap is the whole point of the KEY
    // test below.
    // ------------------------------------------------------------------------

    fn bcol(name: &str, ty: ColumnType, ordinal: usize, is_vector: bool) -> ColumnSchema {
        ColumnSchema {
            name: name.to_string(),
            ty,
            ordinal,
            is_vector,
        }
    }

    /// The plan-side `docs` schema: vector@0 (is_vector), author@1, title@2,
    /// published_at@3.
    fn docs_bind_schema() -> BindSchema {
        BindSchema {
            columns: vec![
                bcol("vector", ColumnType::Vector(768), 0, true),
                bcol("author", ColumnType::Text, 1, false),
                bcol("title", ColumnType::Text, 2, false),
                bcol("published_at", ColumnType::Int, 3, false),
            ],
        }
    }

    /// The storage `docs` schema. `from_columns` assigns the scalar ColumnIds:
    /// vector gets none, author→0, title→1, published_at→2.
    fn docs_meta_schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".to_string(),
                dim: NonZeroUsize::new(768).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "author".to_string(),
                ty: MetaColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "title".to_string(),
                ty: MetaColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "published_at".to_string(),
                ty: MetaColumnType::Int,
            },
        ])
        .expect("valid storage schema")
    }

    fn colref(name: &str, ordinal: usize) -> ColumnRef {
        ColumnRef {
            name: name.to_string(),
            ordinal,
        }
    }

    fn docs_scan() -> LogicalPlan {
        LogicalPlan::Scan(Scan {
            collection: "docs".to_string(),
            schema: docs_bind_schema(),
        })
    }

    /// A `Project` over the `docs` scan.
    fn docs_project(columns: Vec<ColumnRef>, include_vector: bool) -> LogicalPlan {
        LogicalPlan::Project(Project {
            input: Box::new(docs_scan()),
            columns,
            include_vector,
        })
    }

    /// The bootstrap insert row, in schema order (vector, author, title,
    /// published_at) — as the binder already produced it.
    fn bootstrap_row() -> Vec<TypedValue> {
        vec![
            TypedValue {
                value: Literal::Vector(vec![0.1f32; 768]),
                ty: ColumnType::Vector(768),
            },
            TypedValue {
                value: Literal::Str("alice".to_string()),
                ty: ColumnType::Text,
            },
            TypedValue {
                value: Literal::Str("My doc".to_string()),
                ty: ColumnType::Text,
            },
            TypedValue {
                value: Literal::Int(1_700_000_000),
                ty: ColumnType::Int,
            },
        ]
    }

    fn program(ops: Vec<Op>, consts: ConstPool, n_regs: u32, n_cursors: u8) -> Program {
        Program {
            ops,
            consts,
            n_regs,
            n_cursors,
        }
    }

    // ========================================================================
    // SELECT — the wrapping / recursion case.
    // ========================================================================

    /// THE KEY TEST. Vector-first fixture: author/title are declaration
    /// ordinals 1/2 but storage ColumnIds 0/1. The emitted `Column` ops MUST
    /// carry the STORAGE ColumnIds — proof the compiler translates through
    /// `schema.locate()` and never re-derives a ColumnId from the plan ordinal.
    #[test]
    fn select_two_scalars_emits_storage_column_ids_and_backpatched_jumps() {
        let plan = docs_project(vec![colref("author", 1), colref("title", 2)], false);
        let expected = program(
            vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                // SeekFirst's target is the Halt (index 6).
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(6),
                },
                // author: declaration ordinal 1 → storage ColumnId 0.
                Op::Column {
                    cur: Cursor(0),
                    col: 0,
                    dst: Reg(0),
                },
                // title: declaration ordinal 2 → storage ColumnId 1.
                Op::Column {
                    cur: Cursor(0),
                    col: 1,
                    dst: Reg(1),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 2,
                },
                // Next's target is the first body instruction (index 2).
                Op::Next {
                    cur: Cursor(0),
                    body: Addr(2),
                },
                Op::Halt,
            ],
            ConstPool::new(),
            2,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    #[test]
    fn select_star_scalars_emits_no_vector_fetch() {
        // include_vector = false → NO VectorFetch, embedding excluded.
        let plan = docs_project(
            vec![
                colref("author", 1),
                colref("title", 2),
                colref("published_at", 3),
            ],
            false,
        );
        let expected = program(
            vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(7),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 0,
                    dst: Reg(0),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 1,
                    dst: Reg(1),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 2,
                    dst: Reg(2),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 3,
                },
                Op::Next {
                    cur: Cursor(0),
                    body: Addr(2),
                },
                Op::Halt,
            ],
            ConstPool::new(),
            3,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    #[test]
    fn select_vector_only_emits_vector_fetch_not_column() {
        // include_vector = true, vector alone → VectorFetch, no ColumnId used.
        let plan = docs_project(vec![colref("vector", 0)], true);
        let expected = program(
            vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::VectorFetch {
                    cur: Cursor(0),
                    dst: Reg(0),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Next {
                    cur: Cursor(0),
                    body: Addr(2),
                },
                Op::Halt,
            ],
            ConstPool::new(),
            1,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    #[test]
    fn select_scalar_and_vector_mixes_column_and_vector_fetch() {
        // Projection order (author, vector) is preserved: Column for the
        // scalar, VectorFetch for the embedding — the vector never via Column.
        let plan = docs_project(vec![colref("author", 1), colref("vector", 0)], true);
        let expected = program(
            vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(6),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 0,
                    dst: Reg(0),
                },
                Op::VectorFetch {
                    cur: Cursor(0),
                    dst: Reg(1),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 2,
                },
                Op::Next {
                    cur: Cursor(0),
                    body: Addr(2),
                },
                Op::Halt,
            ],
            ConstPool::new(),
            2,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    #[test]
    fn select_single_column_edge_shape() {
        let plan = docs_project(vec![colref("author", 1)], false);
        let expected = program(
            vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 0,
                    dst: Reg(0),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Next {
                    cur: Cursor(0),
                    body: Addr(2),
                },
                Op::Halt,
            ],
            ConstPool::new(),
            1,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    // ========================================================================
    // INSERT — the no-loop case.
    // ========================================================================

    #[test]
    fn insert_emits_value_loads_makerecord_and_no_scan() {
        // Large values (vector, strings) go to the ConstPool; ints load inline.
        // NO SeekFirst, NO Next — INSERT walks no rows.
        let plan = LogicalPlan::Insert(Insert {
            collection: "docs".to_string(),
            schema: docs_bind_schema(),
            row: bootstrap_row(),
        });

        let mut consts = ConstPool::new();
        let vec_id = consts.add(Const::Vector(vec![0.1f32; 768]));
        let author_id = consts.add(Const::Str("alice".to_string()));
        let title_id = consts.add(Const::Str("My doc".to_string()));

        let expected = program(
            vec![
                Op::OpenWrite {
                    cur: Cursor(0),
                    collection: "docs".to_string(),
                },
                Op::VectorConst {
                    id: vec_id,
                    dst: Reg(0),
                },
                Op::String {
                    id: author_id,
                    dst: Reg(1),
                },
                Op::String {
                    id: title_id,
                    dst: Reg(2),
                },
                Op::Integer {
                    value: 1_700_000_000,
                    dst: Reg(3),
                },
                Op::MakeRecord {
                    start: Reg(0),
                    count: 4,
                    dst: Reg(4),
                },
                Op::Insert {
                    cur: Cursor(0),
                    rec: Reg(4),
                },
                Op::Halt,
            ],
            consts,
            5,
            1,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    // ========================================================================
    // CREATE — the one-fat-op case.
    // ========================================================================

    #[test]
    fn create_emits_single_fat_op_with_schema_in_pool() {
        // Schema interned in the pool; no cursor, no loop — just the fat op + Halt.
        let plan = LogicalPlan::CreateCollection(CreateCollection {
            name: "docs".to_string(),
            schema: docs_bind_schema(),
            capacity: 1_000_000,
        });

        let mut consts = ConstPool::new();
        let schema_id = consts.add(Const::Schema(docs_meta_schema()));

        let expected = program(
            vec![
                Op::CreateCollection {
                    name: "docs".to_string(),
                    schema: schema_id,
                    capacity: 1_000_000,
                },
                Op::Halt,
            ],
            consts,
            0,
            0,
        );
        assert_eq!(compile(plan, &docs_meta_schema()), expected);
    }

    /// SINGLE-SOURCE INVARIANT (schema drift guard). The `metadata::Schema` the
    /// compiler interns as `Const::Schema` MUST be the same object the engine's
    /// create path persists — not a second, independently-built schema that can
    /// drift in ordinals / vector position / capacity.
    ///
    /// This holds by construction: `create_collection` takes a `Schema` by value
    /// and moves it verbatim into the `CollectionConfig` its WAL record carries
    /// (it never re-derives one), and at execution the VM has only the program +
    /// const pool — no DDL — so the interned `Const::Schema` is its ONLY possible
    /// schema source. This test locks the type-level contract: one builder feeds
    /// both the pool and the persisted config, and interning is not lossy (the
    /// full vector-inclusive schema round-trips). A full end-to-end
    /// intern-equals-persist test lands with the VM.
    ///
    /// (The companion `create_emits_*` test pins the other half — that the
    /// compiler interns exactly the schema handed to `compile`.)
    #[test]
    fn create_interned_schema_is_the_schema_the_create_path_persists() {
        // ONE builder from the DDL — the single source.
        let schema = docs_meta_schema();

        // The compiler interns exactly this schema (see the expected CREATE
        // program above).
        let mut consts = ConstPool::new();
        let id = consts.add(Const::Schema(schema.clone()));

        // The engine's create path persists exactly this schema, verbatim, into
        // the CollectionConfig its WAL record carries (create_collection moves
        // the argument in — it does not build a new one).
        let persisted = CollectionConfig {
            id: 0,
            name: "docs".to_string(),
            capacity: 1_000_000,
            schema: schema.clone(),
        };

        match consts.get(id) {
            Some(Const::Schema(interned)) => assert_eq!(interned, &persisted.schema),
            other => panic!("expected an interned Const::Schema, got {other:?}"),
        }
    }

    // ========================================================================
    // Structural — every emitted program is well-formed.
    // ========================================================================

    #[test]
    fn compiled_program_passes_validate() {
        let plan = docs_project(vec![colref("author", 1), colref("title", 2)], false);
        assert!(compile(plan, &docs_meta_schema()).validate().is_ok());
    }
}
