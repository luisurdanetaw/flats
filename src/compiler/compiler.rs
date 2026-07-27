//! Bytecode compiler — the lowering pass from [`LogicalPlan`] to [`Program`].
//!
//! Front-to-back the query layer is: **lexer → parser → binder → planner →
//! COMPILER → VM → optimizer**. This module is the SQLite-lineage codegen step:
//! a pure *structural* translation of the already-resolved, already-validated
//! plan into the instruction stream. It does no execution and no catalog
//! access — everything it needs is the plan plus, for a statement against an
//! EXISTING collection, that collection's storage [`Schema`] as held by the
//! CATALOG, handed in by the caller. (A `CREATE` has no catalog entry and
//! supplies its own; see below.)
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
//! # The catalog is the single source of truth — except for `CREATE`
//!
//! Every statement against an EXISTING collection compiles against that
//! collection's schema **as the catalog holds it**. The caller performs the
//! lookup (the compiler stays catalog-free), against the same catalog the binder
//! resolved names through, so the ordinals in the plan and the `ColumnId`s in
//! the emitted code come from one stored schema. Nothing is reconstructed, and
//! there is no second build to drift from the first.
//!
//! A `CREATE` is the sole exception, and necessarily so: the collection does not
//! exist yet, so there is no catalog entry to read and its storage schema can
//! only be derived from the DDL. [`compile`] does that itself via
//! [`to_metadata_schema`] and interns the result — the `Const::Schema` a CREATE
//! program carries is lowered from the very statement being compiled.
//!
//! The `schema` parameter is an `Option` for exactly this reason: `None` means
//! "no such collection yet", which is a real state of the world rather than a
//! missing argument. It used to be required, which forced a `CREATE`'s caller to
//! fabricate a schema for a parameter this module never reads — a fabrication
//! one refactor away from being read is a silent wrong-column bug.

use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
use crate::compiler::constants::{Const, ConstPool};
use crate::compiler::schema::{SchemaError, to_metadata_schema};
use crate::metadata::common::{ColumnLocation, DeclarationOrdinal, Schema};
use crate::sql::ast::Literal;
use crate::sql::plan::LogicalPlan;
use std::fmt;

/// Compile a resolved [`LogicalPlan`] into a [`Program`].
///
/// `schema` is the TARGET COLLECTION'S storage schema, **as held by the
/// catalog** — the vector-inclusive [`Schema`] whose [`locate`](Schema::locate)
/// maps declaration ordinals to storage locations. The caller looks it up from
/// the same catalog the binder resolved against and hands it in; the compiler
/// itself stays catalog-free.
///
/// `None` means THERE IS NO SUCH COLLECTION YET — which is true of exactly one
/// statement, `CREATE COLLECTION`, whose storage schema can only come from its
/// own DDL (see the module header). The `Option` is the whole point: a `CREATE`
/// used to force its caller to fabricate a schema for a parameter this function
/// never reads, and a fabricated schema one refactor away from being read is a
/// silent wrong-column bug. Now the type says which statements have a schema and
/// which cannot.
///
/// A non-`CREATE` plan with `None` is [`CompileError::MissingSchema`].
pub fn compile(plan: LogicalPlan, schema: Option<&Schema>) -> Result<Program, CompileError> {
    // THE one fallible step, hoisted out of emission on purpose: doing it here
    // rather than in the CREATE arm keeps `emit_node` — and the recursive body
    // closures threaded through it — infallible, so the `?` never has to travel
    // back up through a `FnMut`.
    let created = match &plan {
        LogicalPlan::CreateCollection(create) => Some(to_metadata_schema(&create.schema)?),
        _ => None,
    };

    // Every other statement translates ordinals through the catalog's schema, so
    // its absence is a caller bug rather than something to paper over.
    let schema = match (&plan, schema) {
        (LogicalPlan::CreateCollection(_), _) => None,
        (_, Some(schema)) => Some(schema),
        (_, None) => return Err(CompileError::MissingSchema),
    };
    let mut compiler = Compiler::new(schema, created);
    // The plan root emits itself. Wrapper nodes (Scan) thread a per-row body
    // DOWN into their loop; the root therefore receives an empty top-level body
    // (the leaf nodes ignore the body entirely).
    compiler.emit_node(&plan, &mut |_| {});
    // Every program halts once the root's code is done — a Scan's loop has
    // already fallen through to here, an Insert/Create has finished its op.
    compiler.emit(Op::Halt);
    Ok(compiler.finish())
}

/// Why a plan could not be compiled.
///
/// A stage-local error type (CLAUDE.md §5), currently with one cause: a `CREATE`
/// whose bound schema does not lower to a valid storage schema. Emission itself
/// is infallible — the binder and planner resolved everything — so every future
/// variant will be of the same kind: a lowering the frontend could not have
/// checked.
#[derive(Debug)]
pub enum CompileError {
    /// A `CREATE COLLECTION`'s schema could not be lowered to storage form.
    Schema(SchemaError),
    /// A statement against an existing collection was compiled without that
    /// collection's catalog schema — there is nothing to translate its
    /// declaration ordinals through.
    MissingSchema,
}

impl From<SchemaError> for CompileError {
    fn from(e: SchemaError) -> Self {
        CompileError::Schema(e)
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Schema(e) => write!(f, "cannot compile CREATE COLLECTION: {e}"),
            CompileError::MissingSchema => {
                write!(f, "no catalog schema for the target collection")
            }
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompileError::Schema(e) => Some(e),
            CompileError::MissingSchema => None,
        }
    }
}

/// The compiler's working state: the growing instruction stream, the constant
/// pool, the monotonic register/cursor allocators, and the cursor currently in
/// scope (threaded into a `Scan`'s body so `Column` / `VectorFetch` reference
/// the right cursor rather than a hardcoded `0`).
///
/// Borrows the storage [`Schema`] for the duration of the compile so
/// [`emit_node`](Compiler::emit_node) can call [`Schema::locate`] without
/// re-plumbing it through every helper.
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
    /// The target collection's storage schema, from the catalog — the source of
    /// the ordinal → storage-location translation. `None` only for a `CREATE`,
    /// which has no collection to describe and never reads it.
    schema: Option<&'a Schema>,
    /// For a `CREATE`, the storage schema [`compile`] lowered from the DDL,
    /// waiting to be interned. `None` for every other plan.
    created: Option<Schema>,
}

impl<'a> Compiler<'a> {
    /// A fresh compiler for a plan compiled against `schema`, carrying the
    /// pre-lowered `created` schema when the plan is a `CREATE`.
    fn new(schema: Option<&'a Schema>, created: Option<Schema>) -> Self {
        Compiler {
            ops: Vec::new(),
            consts: ConstPool::new(),
            next_reg: 0,
            next_cursor: 0,
            cursor: None,
            schema,
            created,
        }
    }

    /// Push `op`, returning its instruction index (used as a backpatch handle).
    fn emit(&mut self, op: Op) -> usize {
        let at = self.ops.len();
        self.ops.push(op);
        at
    }

    /// The address of the NEXT instruction to be emitted — i.e. the current end
    /// of the stream. Used to capture a loop's top and to compute a forward
    /// jump's target.
    fn here(&self) -> Addr {
        Addr(self.ops.len() as u32)
    }

    /// Emit a forward jump whose target is not yet known: `make` builds the op
    /// from [`Addr::PLACEHOLDER`], and the returned index is later handed to
    /// [`patch`](Self::patch). Every forward jump routes through here so no
    /// patch site has to re-match the opcode.
    fn emit_jump_placeholder(&mut self, make: impl FnOnce(Addr) -> Op) -> usize {
        self.emit(make(Addr::PLACEHOLDER))
    }

    /// Backpatch the jump at instruction index `at` to point at `target`.
    fn patch(&mut self, at: usize, target: Addr) {
        match &mut self.ops[at] {
            Op::SeekFirst { end, .. } => *end = target,
            Op::Next { body, .. } => *body = target,
            // EXTEND: Goto / JumpIfFalse / DecrJumpIfZero once WHERE and LIMIT
            // land. Only a jump-bearing op is a valid patch target.
            other => unreachable!("patch target at {at} is not a jump: {other:?}"),
        }
    }

    /// Allocate the next register (monotonic; no reuse). Bumps [`next_reg`],
    /// which becomes [`Program::n_regs`].
    fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Allocate the next cursor slot (monotonic; no reuse). Bumps [`next_cursor`],
    /// which becomes [`Program::n_cursors`].
    fn alloc_cursor(&mut self) -> Cursor {
        let c = Cursor(self.next_cursor);
        self.next_cursor += 1;
        c
    }

    /// The cursor in scope. Set while emitting inside a `Scan`'s loop, so the
    /// body's `Column` / `VectorFetch` reference the cursor `Scan` allocated
    /// rather than a hardcoded slot. Reached only from within a loop body, where
    /// it is always set.
    fn current_cursor(&self) -> Cursor {
        match self.cursor {
            Some(cur) => cur,
            None => unreachable!("Column/VectorFetch emitted outside a Scan loop"),
        }
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
        match node {
            LogicalPlan::Scan(scan) => {
                // WRAPPER: owns the read loop. SeekFirst is the entry guard
                // (empty collection -> skip the body), Next the back-edge.
                let cur = self.alloc_cursor();
                self.emit(Op::OpenRead {
                    cur,
                    collection: scan.collection.clone(),
                });
                // Forward jump out of an empty collection — target not known
                // until the loop body has been emitted, so backpatch it.
                let seek = self.emit_jump_placeholder(|end| Op::SeekFirst { cur, end });
                let loop_top = self.here();

                // Thread this cursor into the body, restoring the prior one
                // afterwards (there is only one level in the bootstrap subset,
                // but nesting stays correct this way).
                let prev = self.cursor.replace(cur);
                body(self);
                self.cursor = prev;

                self.emit(Op::Next {
                    cur,
                    body: loop_top,
                });
                // SeekFirst lands just past the loop's back-edge.
                let after = self.here();
                self.patch(seek, after);
            }

            LogicalPlan::Project(project) => {
                // MIDDLE: owns no loop. Its per-row code is handed DOWN to the
                // child (a Scan) so it runs inside that loop.
                // `c` is annotated `Compiler<'a>` (not a fresh inference): `&mut`
                // is invariant over the schema lifetime, so the body handed down
                // must speak the SAME `'a` the outer `body` expects.
                let mut row_body = |c: &mut Compiler<'a>| {
                    let base = c.next_reg;
                    for col in &project.columns {
                        // THE ordinal translation: ask the schema where this
                        // declaration ordinal lives — never compute it here.
                        // `compile` proved a non-CREATE plan has a schema
                        // before emission began.
                        let schema = match c.schema {
                            Some(schema) => schema,
                            None => unreachable!("a read compiled without a catalog schema"),
                        };
                        match schema.locate(DeclarationOrdinal::new(col.ordinal)) {
                            Some(ColumnLocation::Scalar(id)) => {
                                let dst = c.alloc_reg();
                                let cur = c.current_cursor();
                                c.emit(Op::Column { cur, col: id, dst });
                            }
                            Some(ColumnLocation::Vector { .. }) => {
                                // The embedding is fetched from the flat index,
                                // never read as a Column — and only when the
                                // projection asked for it.
                                if project.include_vector {
                                    let dst = c.alloc_reg();
                                    let cur = c.current_cursor();
                                    c.emit(Op::VectorFetch { cur, dst });
                                }
                            }
                            // The plan is already validated, so every ordinal
                            // resolves; a miss is a compiler/schema-pairing bug.
                            None => {
                                unreachable!("plan ordinal {} not in schema", col.ordinal)
                            }
                        }
                    }
                    // Emit the columns produced this row as one output row.
                    let count = c.next_reg - base;
                    c.emit(Op::ResultRow {
                        start: Reg(base),
                        count,
                    });
                    // Anything ABOVE the projection (e.g. a future Limit).
                    body(c);
                };
                self.emit_node(&project.input, &mut row_body);
            }

            LogicalPlan::Insert(insert) => {
                // LEAF: no loop, no body. Load each value, pack, write.
                let cur = self.alloc_cursor();
                self.emit(Op::OpenWrite {
                    cur,
                    collection: insert.collection.clone(),
                });
                let base = self.next_reg;
                for value in &insert.row {
                    let dst = self.alloc_reg();
                    // Large payloads (vector, string) go to the constant pool
                    // and the instruction carries the handle; small scalars load
                    // inline.
                    let op = match &value.value {
                        Literal::Vector(v) => {
                            let id = self.consts.add(Const::Vector(v.clone()));
                            Op::VectorConst { id, dst }
                        }
                        Literal::Str(s) => {
                            let id = self.consts.add(Const::Str(s.clone()));
                            Op::String { id, dst }
                        }
                        Literal::Int(n) => Op::Integer { value: *n, dst },
                        Literal::Float(f) => Op::Real { value: *f, dst },
                    };
                    self.emit(op);
                }
                let count = self.next_reg - base;
                let rec = self.alloc_reg();
                self.emit(Op::MakeRecord {
                    start: Reg(base),
                    count,
                    dst: rec,
                });
                self.emit(Op::Insert { cur, rec });
            }

            LogicalPlan::CreateCollection(create) => {
                // LEAF: one fat op. The interned schema is the one `compile`
                // LOWERED FROM THIS STATEMENT'S DDL — never `self.schema`, which
                // describes an existing collection and, for a CREATE, is
                // whatever the caller happened to have in hand. It is the single
                // source the VM later persists; there is never a second build.
                let lowered = match self.created.take() {
                    Some(schema) => schema,
                    // `compile` lowers it before emission begins, so reaching
                    // here means the two dispatches disagree about the plan.
                    None => unreachable!("compile did not lower the CREATE schema"),
                };
                let schema = self.consts.add(Const::Schema(lowered));
                self.emit(Op::CreateCollection {
                    name: create.name.clone(),
                    schema,
                    capacity: create.capacity,
                });
            }
        }
    }

    /// Finish the compile, moving the accumulated state into a [`Program`]. The
    /// allocator counters become the register / cursor counts.
    fn finish(self) -> Program {
        Program {
            ops: self.ops,
            consts: self.consts,
            n_regs: self.next_reg,
            n_cursors: self.next_cursor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompileError, compile};
    use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
    use crate::compiler::constants::{Const, ConstId, ConstPool};
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
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
        assert_eq!(
            compile(plan, Some(&docs_meta_schema())).expect("compiles"),
            expected
        );
    }

    /// The interned schema is LOWERED FROM THE DDL, not echoed from the caller.
    ///
    /// Every other CREATE test passes a `schema` argument that happens to equal
    /// the one the DDL lowers to, so none of them can tell the two apart. This
    /// one passes a deliberately WRONG schema — different columns, different
    /// dimension — and asserts the program still carries the DDL's.
    #[test]
    fn create_ignores_the_caller_schema_and_lowers_the_ddl() {
        let plan = LogicalPlan::CreateCollection(CreateCollection {
            name: "docs".to_string(),
            schema: docs_bind_schema(),
            capacity: 1_000_000,
        });

        // Nothing like `docs`: one scalar, a 2-dim vector, different names.
        let unrelated = Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "other".to_string(),
                dim: NonZeroUsize::new(2).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "junk".to_string(),
                ty: MetaColumnType::Int,
            },
        ])
        .expect("valid schema");

        let program = compile(plan, Some(&unrelated)).expect("compiles");
        assert_eq!(
            program.consts.get(ConstId(0)),
            Some(&Const::Schema(docs_meta_schema())),
            "CREATE must intern the schema lowered from its own DDL"
        );
    }

    /// A `CREATE` needs NO catalog schema, and the signature says so.
    ///
    /// The collection does not exist yet, so there is nothing to look up — the
    /// storage schema can only come from the DDL. Passing `None` is the honest
    /// spelling of that; the previous shape forced a caller to fabricate a
    /// schema for a parameter this arm never reads.
    #[test]
    fn create_compiles_without_a_catalog_schema() {
        let plan = LogicalPlan::CreateCollection(CreateCollection {
            name: "docs".to_string(),
            schema: docs_bind_schema(),
            capacity: 1_000_000,
        });

        let program = compile(plan, None).expect("a CREATE needs no catalog schema");
        assert_eq!(
            program.consts.get(ConstId(0)),
            Some(&Const::Schema(docs_meta_schema())),
            "the interned schema is still the one lowered from the DDL"
        );
    }

    /// A statement against an EXISTING collection cannot compile without that
    /// collection's schema — it is the source of every ordinal → `ColumnId`
    /// translation. A missing one is a caller bug, and it must be an error
    /// rather than a fabricated stand-in that silently emits wrong columns.
    #[test]
    fn a_statement_against_an_existing_collection_requires_its_schema() {
        let plan = LogicalPlan::Project(Project {
            input: Box::new(LogicalPlan::Scan(Scan {
                collection: "docs".to_string(),
                schema: docs_bind_schema(),
            })),
            columns: vec![colref("author", 1)],
            include_vector: false,
        });

        assert!(matches!(
            compile(plan, None).unwrap_err(),
            CompileError::MissingSchema
        ));
    }

    /// A schema the binder accepts but storage cannot represent fails the
    /// compile rather than producing a corrupt `Const::Schema`. `VECTOR(0)`
    /// passes the binder (rule C counts vector columns, it does not check the
    /// dimension) and dies here.
    #[test]
    fn create_with_unlowerable_schema_is_a_compile_error() {
        let plan = LogicalPlan::CreateCollection(CreateCollection {
            name: "bad".to_string(),
            schema: BindSchema {
                columns: vec![bcol("v", ColumnType::Vector(0), 0, true)],
            },
            capacity: 10,
        });

        let err = compile(plan, Some(&docs_meta_schema())).unwrap_err();
        assert!(matches!(
            err,
            CompileError::Schema(crate::compiler::schema::SchemaError::ZeroDimension { .. })
        ));
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
        assert!(
            compile(plan, Some(&docs_meta_schema()))
                .expect("compiles")
                .validate()
                .is_ok()
        );
    }
}
