//! The dispatch loop — a **resumable stepper**, not a run-to-completion loop.
//!
//! # Streaming is the shape of this file
//!
//! `SELECT * FROM docs` over a million rows must never materialize a million
//! rows. So, like SQLite's `step()`, [`Vm::step`] runs until it produces **one**
//! row, yields it, and **suspends the entire machine** — program counter,
//! register file, open cursors — inside `self`. The next call resumes exactly
//! where it stopped. That persistence *is* the stream; there is no buffer
//! anywhere.
//!
//! [`Op::ResultRow`] is therefore a **yield point**, not a push to a sink. This
//! composes with the [`Cursor`], which is already
//! pull-based: one `step` pumps one `Cursor::next` pumps one tuple read. A
//! caller that stops pulling stops the whole chain — backpressure for free.
//!
//! Retrofitting this onto a run-to-completion loop is a rewrite, not a patch,
//! which is why the resumable shape is here from the first line rather than
//! after the opcodes work.
//!
//! # No lifetime parameter
//!
//! [`Vm`] is `'static`. Cursor slots will hold owned `Cursor<'static>` values
//! (the cursor owns an `Arc` clone of the tuple reader rather than borrowing
//! it), so a read stream never borrows the [`Db`] while it
//! iterates — the caller can hold a live stream without pinning the database
//! handle in place.
//!
//! # What is here
//!
//! Fifteen of the sixteen opcodes: the chassis (loop, register file, loads),
//! `CREATE COLLECTION`, the `INSERT` path (`OpenWrite` → `MakeRecord` →
//! `Insert`), and the read loop (`OpenRead` → `SeekFirst` → `Column` →
//! `ResultRow` → `Next`).
//!
//! [`Op::VectorFetch`] is the one still unbuilt, so `SELECT vector FROM …`
//! reaches a `todo!`. It needs the flat index's `Reader`, which the cursor
//! does not carry, plus a decision about copying an embedding out of the mmap
//! into a `'static` register — see [`RegValue::Vector`]. It is a `todo!` rather
//! than a no-op on purpose: an arm that silently did nothing would make a
//! `SELECT` return zero rows and read as an empty collection, which is the most
//! expensive way to be wrong.
//!
//! # Three vocabularies meet here
//!
//! An inserted value crosses two boundaries on its way to storage and one on the
//! way back, and each hop is a real translation rather than a cast:
//!
//! ```text
//!   INSERT:  Const/operand → RegValue → Literal → Value        → tuple store
//!                                     (MakeRecord)  (split_record)
//!   SELECT:  tuple store    → Value    → RegValue  → OutputRow
//!                                     (Column)      (ResultRow)
//! ```
//!
//! The insert path is longer because a packed [`Record`] sits in the middle —
//! [`Op::MakeRecord`] builds one in DECLARATION order and
//! [`split_record`] cuts it into the embedding (flat index) and the
//! `ColumnId`-keyed row (tuple store), because those go to different stores.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::compiler::bytecode::{Cursor as BytecodeCursor, Op, Program, Reg, ValidateError};
use crate::compiler::constants::Const;
use crate::engine::cursor::Cursor;
use crate::engine::{CollectionId, Db};
use crate::error::Error;
use crate::metadata::common::{Ordinal, Schema, Value};
use crate::metadata::index as meta;
use crate::vm::record::{Record, SplitError, split_record};
use crate::vm::value;

/// A value in a register.
///
/// NAMING: this is the *contents* of a register; [`Reg`] is the *index* of one.
/// They are deliberately not both called `Reg` — this codebase keeps
/// structurally-different types with related meanings type- and name-distinct
/// (see the `DeclarationOrdinal` / `ColumnId` split), and a `Reg` that is
/// sometimes an operand and sometimes a payload is exactly that bug class.
///
/// Registers hold SMALL tagged values only. [`RegValue::Vector`] does not
/// violate that: an `Arc<[f32]>` is two words, and the 768 floats live on the
/// heap behind it — the register holds the handle, as the ISA requires.
#[derive(Debug, Clone, PartialEq)]
pub enum RegValue {
    /// Never written. NOT a SQL `NULL` — V-SQL has no nulls (see
    /// [`ColumnType`](crate::metadata::common::ColumnType)); this is the
    /// "the emitter never loaded this register" state, and reading it is an
    /// [`ExecError::UnsetRegister`].
    Unset,
    /// An `i64`, from [`Op::Integer`] or an `INT` column.
    Int(i64),
    /// An `f64`, from [`Op::Real`] or a `FLOAT` column.
    Real(f64),
    /// A string, from [`Op::String`] or a `TEXT` column.
    Str(String),
    /// An embedding — **seam (b)**. Produced today only by [`Op::VectorConst`];
    /// [`Op::VectorFetch`] joins it in 5b, and a `SEARCH … NEAREST TO [...]`
    /// query vector interns as a `Const` and lands here too.
    ///
    /// `Arc<[f32]>` rather than a pool handle because the two producers have
    /// different sources: a literal is in the constant pool, but a fetched
    /// embedding is in the mmap'd flat index, and a borrow of that cannot live
    /// in a `'static` register file. An `Arc` is the one representation both
    /// can produce and neither has to copy again.
    Vector(Arc<[f32]>),
    /// A packed row, built by [`Op::MakeRecord`] and consumed by
    /// [`Op::Insert`]. The ISA's `MakeRecord dst` is a register, so a register
    /// has to be able to hold one.
    Record(Record),
    /// A set of ordinals — the value a `WHERE` clause computes.
    ///
    /// Produced by the `Bitmap*` ops and consumed by `OpenRead`/`KnnScan` as
    /// their `filter` operand. `Arc` for the same reason [`RegValue::Vector`]
    /// has one: a bitmap over a large collection is a real payload, and the
    /// combining ops read their inputs without wanting to copy them.
    Bitmap(Arc<RoaringBitmap>),
}

/// One cursor slot: what a `cur` operand refers to between `OpenRead`/
/// `OpenWrite` and the end of the program.
///
/// Read and write cursors are different things, not one thing with a flag: a
/// read cursor walks rows and is asked for columns, a write cursor never
/// iterates and instead needs the target's id and schema to split a record
/// against. Keeping them as separate variants means `Column` cannot be pointed
/// at a write cursor, or `Insert` at a read one, without a named error.
enum Slot {
    /// Never opened, or `cur` was allocated and never used.
    Closed,
    /// A read cursor. `Cursor<'static>` — it owns an `Arc` clone of the tuple
    /// reader, so a suspended stream borrows nothing.
    ///
    /// `scores` is present only for a cursor opened by [`Op::KnnScan`]: the
    /// similarity the kernel computed, KEYED BY ORDINAL.
    ///
    /// Keyed, not positional, and that is the whole correctness argument. The
    /// obvious design — a parallel array plus an index bumped once per `Next` —
    /// assumes the cursor consumes exactly one ordinal per step. It does not:
    /// the cursor SKIPS any ordinal whose row is not materialized (a mid-apply
    /// `Missing`, or a `Deleted` that landed in the tuple store before the flat
    /// index), so one `Next` can swallow several ordinals and slide every
    /// later row's score onto the wrong id. A map cannot drift, at the cost of
    /// one small allocation per ranked query.
    Read {
        /// The cursor being walked.
        cursor: Box<Cursor<'static>>,
        /// ordinal → similarity, for a ranked read. `None` for a plain scan,
        /// which has no scores to report.
        scores: Option<HashMap<Ordinal, f32>>,
    },
    /// A write cursor: the resolved collection and the schema `split_record`
    /// needs. Resolved ONCE at open time rather than per row.
    Write {
        /// The resolved target.
        collection: CollectionId,
        /// Its storage schema — the source of the declaration-ordinal →
        /// storage-location mapping the split performs.
        schema: Schema,
    },
}

/// One row yielded by [`Op::ResultRow`].
///
/// Built from a RUN OF REGISTERS — **seam (c)** — never read straight off a
/// cursor. That is what will let `RETURNING id, score` mix a stored column
/// (loaded by `Column`) with a computed `f32` (parked in a register by a KNN
/// op) at no extra cost: by the time `ResultRow` runs, both are just registers.
///
/// NAMING: not `Row`, which is already [`metadata::Row`](crate::metadata::Row)
/// — a `Vec<(ColumnId, Value)>` addressed by storage column id. This is a
/// positional list of register values. Two different things.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputRow(pub Vec<RegValue>);

/// What one instruction did to control flow. Internal to the dispatch loop —
/// [`Vm::step`] collapses it into the `Option<OutputRow>` a caller sees.
enum Flow {
    /// [`Op::ResultRow`] ran: hand this row out and suspend.
    Yield(OutputRow),
    /// [`Op::Halt`] ran: the program is finished.
    Halt,
    /// Anything else: state changed, keep dispatching.
    Next,
}

/// Whether an op needs a [`Db`] to run.
///
/// The four that do are the ones that touch the catalog or the stores. Every
/// other op works off state the [`Vm`] already owns — which is exactly why a
/// read can detach after its cursor is open (see [`Vm::open_cursors`]).
///
/// `VectorFetch` is deliberately absent: its arm is still a `todo!`, so listing
/// it here would turn "not implemented" into the misleading "cannot detach".
fn needs_db(op: &Op) -> bool {
    matches!(
        op,
        Op::OpenRead { .. }
            | Op::OpenWrite { .. }
            | Op::KnnScan { .. }
            | Op::Insert { .. }
            | Op::CreateCollection { .. }
            // Every bitmap op reads the collection's metadata index.
            | Op::BitmapEq { .. }
            | Op::BitmapRange { .. }
            | Op::BitmapNot { .. }
    )
}

/// An op's mnemonic, for error messages.
fn mnemonic(op: &Op) -> &'static str {
    match op {
        Op::OpenRead { .. } => "OpenRead",
        Op::OpenWrite { .. } => "OpenWrite",
        Op::SeekFirst { .. } => "SeekFirst",
        Op::Next { .. } => "Next",
        Op::Column { .. } => "Column",
        Op::VectorFetch { .. } => "VectorFetch",
        Op::Integer { .. } => "Integer",
        Op::Real { .. } => "Real",
        Op::String { .. } => "String",
        Op::VectorConst { .. } => "VectorConst",
        Op::ResultRow { .. } => "ResultRow",
        Op::MakeRecord { .. } => "MakeRecord",
        Op::Insert { .. } => "Insert",
        Op::KnnScan { .. } => "KnnScan",
        Op::RowId { .. } => "RowId",
        Op::Score { .. } => "Score",
        Op::CreateCollection { .. } => "CreateCollection",
        Op::BitmapEq { .. } => "BitmapEq",
        Op::BitmapRange { .. } => "BitmapRange",
        Op::BitmapAnd { .. } => "BitmapAnd",
        Op::BitmapOr { .. } => "BitmapOr",
        Op::BitmapNot { .. } => "BitmapNot",
        Op::Halt => "Halt",
    }
}

/// The virtual machine: a compiled [`Program`] plus the state that persists
/// between [`step`](Vm::step) calls.
///
/// No lifetime parameter — see the module header.
pub struct Vm {
    /// The program and its constant pool. Owned, never mutated.
    program: Program,
    /// The program counter — the index of the NEXT instruction to run. Survives
    /// across `step` calls; that is what makes the machine resumable.
    pc: usize,
    /// The register file, `program.n_regs` wide, allocated once at construction.
    regs: Vec<RegValue>,
    /// The cursor slots, `program.n_cursors` wide. Part of the suspended state:
    /// a stream's open cursor survives across `step` calls, which is what lets
    /// the next call resume the scan instead of restarting it.
    cursors: Vec<Slot>,
}

impl Vm {
    /// Load a program.
    ///
    /// Runs [`Program::validate`] ONCE here, so the dispatch loop never has to
    /// re-prove what validation already covers: every register operand is in
    /// range, every jump target resolves, every constant handle is live, and the
    /// stream ends in [`Op::Halt`]. A program that fails is rejected before any
    /// state exists.
    pub fn new(program: Program) -> Result<Vm, ExecError> {
        program.validate().map_err(ExecError::Invalid)?;
        let regs = vec![RegValue::Unset; program.n_regs as usize];
        let cursors = (0..program.n_cursors).map(|_| Slot::Closed).collect();
        Ok(Vm {
            program,
            pc: 0,
            regs,
            cursors,
        })
    }

    /// A machine that is already finished: an empty program that only halts.
    ///
    /// For the caller that needs a `Vm`-shaped nothing — a mutation's result
    /// stream, which has already run and has no rows to produce.
    pub fn halted() -> Vm {
        Vm {
            program: Program {
                ops: vec![Op::Halt],
                consts: crate::compiler::constants::ConstPool::new(),
                n_regs: 0,
                n_cursors: 0,
            },
            pc: 0,
            regs: Vec::new(),
            cursors: Vec::new(),
        }
    }

    /// Run until the next row, and yield it.
    ///
    /// * `Ok(Some(row))` — [`Op::ResultRow`] executed. The machine is now
    ///   **suspended** with `pc` parked immediately after it; call again to
    ///   resume.
    /// * `Ok(None)` — [`Op::Halt`] reached. The stream is drained, and calling
    ///   again keeps returning `Ok(None)` (`pc` stays on the `Halt`), so a
    ///   `while let Some(row) = vm.step(&db)?` drain loop terminates cleanly rather
    ///   than restarting the program.
    ///
    /// Every other opcode mutates state and continues the loop without
    /// returning — the loop only ever exits at a yield, a halt, or an error.
    ///
    /// `db` is passed per call rather than held, which is what keeps [`Vm`]
    /// free of a lifetime parameter: the machine can be stored, returned, and
    /// moved without pinning the database in place. Cursors opened along the way
    /// are `Cursor<'static>` and borrow nothing, so a suspended stream holds no
    /// reference to `db` between steps.
    pub fn step(&mut self, db: &Db) -> Result<Option<OutputRow>, ExecError> {
        self.run(Some(db))
    }

    /// Run until the next row, **detached from the database**.
    ///
    /// Identical to [`step`](Self::step) except that no `Db` is available, so an
    /// op that needs one fails with [`ExecError::Detached`] instead of running.
    /// This is what a read stream pulls on: once
    /// [`open_cursors`](Self::open_cursors) has run, a compiled `SELECT` touches
    /// nothing but its already-open cursor, so the rest of the scan needs no
    /// database handle at all — and the stream that owns this `Vm` borrows
    /// nothing.
    pub fn resume(&mut self) -> Result<Option<OutputRow>, ExecError> {
        self.run(None)
    }

    /// Execute the leading cursor-opening ops, then DETACH.
    ///
    /// A compiled read opens its cursors first and then walks them, so running
    /// just that prologue leaves a machine that can finish on its own. Two
    /// things have to hold, and both are checked rather than assumed:
    ///
    ///  * the ops executed here are only the ones that need a `Db`;
    ///  * **no op after them needs one** — otherwise the stream would fail
    ///    mid-flight, having already handed out rows. That is
    ///    [`ExecError::CannotDetach`], raised HERE, while the caller can still
    ///    report it as a compile-time failure.
    ///
    /// The second check is what makes this safe for plan shapes that do not
    /// exist yet: a future statement that opens a cursor mid-stream is rejected
    /// at the boundary instead of surprising a consumer.
    pub fn open_cursors(&mut self, db: &Db) -> Result<(), ExecError> {
        // Run until NOTHING AHEAD needs the database — not merely until the
        // current op stops needing it. A `SEARCH`'s prologue is `VectorConst`
        // (which does not) followed by `KnnScan` (which does), so stopping at
        // the first self-sufficient op would detach one instruction too early
        // and strand the scan.
        while self.program.ops[self.pc..].iter().any(needs_db) {
            match self.exec_one(Some(db))? {
                Flow::Next => {}
                // Producing a row, or finishing, before the last storage-facing
                // op means the program interleaves reads with row production —
                // it cannot be detached, and finding that out HERE is the point:
                // no row has been handed out yet, so the caller can still report
                // it as a compile-time failure.
                Flow::Yield(_) | Flow::Halt => {
                    let (at, op) = self.program.ops[self.pc..]
                        .iter()
                        .enumerate()
                        .find(|(_, op)| needs_db(op))
                        // The loop condition just proved one exists.
                        .ok_or(ExecError::PcOutOfRange { pc: self.pc })?;
                    return Err(ExecError::CannotDetach {
                        op: mnemonic(op),
                        at: self.pc + at,
                    });
                }
            }
        }
        Ok(())
    }

    /// The dispatch loop. Exits only at a yield, a halt, or an error.
    fn run(&mut self, db: Option<&Db>) -> Result<Option<OutputRow>, ExecError> {
        loop {
            match self.exec_one(db)? {
                Flow::Yield(row) => return Ok(Some(row)),
                Flow::Halt => return Ok(None),
                Flow::Next => {}
            }
        }
    }

    /// Execute exactly ONE instruction and report what it did to control flow.
    ///
    /// Split out of the loop so [`open_cursors`](Self::open_cursors) can run a
    /// prologue instruction-by-instruction without duplicating a single arm.
    fn exec_one(&mut self, db: Option<&Db>) -> Result<Flow, ExecError> {
        {
            // NOTE `validate` guaranteed the stream ends in `Halt` and that every
            // jump lands inside it, so the pc cannot walk off the end — but a
            // Vm must not index-panic if that ever stops being true.
            let op = self
                .program
                .ops
                .get(self.pc)
                .ok_or(ExecError::PcOutOfRange { pc: self.pc })?;

            match op {
                // -- loads: small scalars inline, large payloads from the pool --
                Op::Integer { value, dst } => {
                    let (value, dst) = (*value, *dst);
                    self.store(dst, RegValue::Int(value))?;
                }
                Op::Real { value, dst } => {
                    let (value, dst) = (*value, *dst);
                    self.store(dst, RegValue::Real(value))?;
                }
                Op::String { id, dst } => {
                    let (id, dst) = (*id, *dst);
                    let value = match self.program.consts.get(id) {
                        Some(Const::Str(s)) => RegValue::Str(s.clone()),
                        other => return Err(ExecError::bad_const(id.0, "a string", other)),
                    };
                    self.store(dst, value)?;
                }
                Op::VectorConst { id, dst } => {
                    let (id, dst) = (*id, *dst);
                    let value = match self.program.consts.get(id) {
                        Some(Const::Vector(v)) => RegValue::Vector(Arc::from(v.as_slice())),
                        other => return Err(ExecError::bad_const(id.0, "a vector", other)),
                    };
                    self.store(dst, value)?;
                }

                // -- the yield point ------------------------------------------
                Op::ResultRow { start, count } => {
                    let (start, count) = (*start, *count);
                    let row = self.read_run(start, count)?;
                    // Advance BEFORE returning: resuming must re-enter after the
                    // yield, not replay it.
                    self.pc += 1;
                    return Ok(Flow::Yield(row));
                }

                // -- done ------------------------------------------------------
                // The pc deliberately does NOT advance, so a drained stream
                // stays drained however many times it is stepped.
                Op::Halt => return Ok(Flow::Halt),

                // -- WHERE: set algebra over the metadata index ---------------
                Op::BitmapEq {
                    collection,
                    col,
                    value,
                    dst,
                } => {
                    let (col, value, dst) = (*col, *value, *dst);
                    let db = db.ok_or(ExecError::Detached { op: "BitmapEq" })?;
                    let meta = self.metadata(db, collection)?;
                    let needle = self.value(value)?;
                    let found = meta.lookup_eq(col, &needle).map_err(ExecError::Engine)?;
                    self.store(dst, RegValue::Bitmap(Arc::new(found)))?;
                }
                Op::BitmapRange {
                    collection,
                    col,
                    op,
                    value,
                    dst,
                } => {
                    let (col, op, value, dst) = (*col, *op, *value, *dst);
                    let db = db.ok_or(ExecError::Detached { op: "BitmapRange" })?;
                    let meta = self.metadata(db, collection)?;
                    let needle = self.value(value)?;
                    let found = meta
                        .lookup_range(col, op, &needle)
                        .map_err(ExecError::Engine)?;
                    self.store(dst, RegValue::Bitmap(Arc::new(found)))?;
                }
                Op::BitmapAnd { a, b, dst } => {
                    let (a, b, dst) = (*a, *b, *dst);
                    let (left, right) = (self.bitmap(a)?, self.bitmap(b)?);
                    let out = left.as_ref() & right.as_ref();
                    self.store(dst, RegValue::Bitmap(Arc::new(out)))?;
                }
                Op::BitmapOr { a, b, dst } => {
                    let (a, b, dst) = (*a, *b, *dst);
                    let (left, right) = (self.bitmap(a)?, self.bitmap(b)?);
                    let out = left.as_ref() | right.as_ref();
                    self.store(dst, RegValue::Bitmap(Arc::new(out)))?;
                }
                Op::BitmapNot {
                    collection,
                    src,
                    dst,
                } => {
                    let (src, dst) = (*src, *dst);
                    let db = db.ok_or(ExecError::Detached { op: "BitmapNot" })?;
                    let meta = self.metadata(db, collection)?;
                    let inner = self.bitmap(src)?;
                    // Complement within the LIVE set, not the whole ordinal
                    // space: a deleted row satisfies no predicate, so negating
                    // one must not resurrect it.
                    let out = meta.live() - inner.as_ref();
                    self.store(dst, RegValue::Bitmap(Arc::new(out)))?;
                }

                // -- the read loop ---------------------------------------------
                Op::OpenRead {
                    cur,
                    collection,
                    filter,
                } => {
                    let (cur, filter) = (*cur, *filter);
                    let db = db.ok_or(ExecError::Detached { op: "OpenRead" })?;
                    let collection_id = db.collection_id(collection).map_err(ExecError::Engine)?;
                    // `scan` snapshots `live()` HERE, so the row SET is fixed
                    // for this scan — but the ROWS are not read yet. The cursor
                    // is positioned before the first row and fetches one at a
                    // time, which is what makes the loop below lazy.
                    //
                    // A `WHERE` replaces that snapshot with the predicate's
                    // bitmap. Nothing else about the loop changes: a cursor is
                    // an ordinal source, and both are ordinal sources.
                    let cursor = match filter {
                        None => db.scan(collection_id).map_err(ExecError::Engine)?,
                        Some(reg) => {
                            let bitmap = self.bitmap(reg)?;
                            db.scan_over(
                                collection_id,
                                // Owned, not borrowed: the cursor is
                                // `'static`, so it cannot hold a reference to
                                // a register that the program may overwrite.
                                bitmap.iter().collect::<Vec<u32>>().into_iter().map(Ordinal),
                            )
                            .map_err(ExecError::Engine)?
                        }
                    };
                    // A plain scan computes no similarities, so `Op::Score`
                    // against this cursor is an error rather than a zero.
                    self.open(
                        cur,
                        Slot::Read {
                            cursor: Box::new(cursor),
                            scores: None,
                        },
                    )?;
                }
                Op::SeekFirst { cur, end } => {
                    let (cur, end) = (*cur, *end);
                    // `?` the real error, branch on the bool: a failed READ is
                    // not "no rows", and collapsing the two would turn an I/O
                    // error into an empty result set.
                    if !self.read_cursor_mut(cur)?.seek_first()? {
                        // Empty scan: jump past the body rather than run it once
                        // against a cursor parked on nothing.
                        self.pc = end.0 as usize;
                        return Ok(Flow::Next);
                    }
                }
                Op::Next { cur, body } => {
                    let (cur, body) = (*cur, *body);
                    if self.read_cursor_mut(cur)?.next()? {
                        // Another row: back to the top of the body. THE back-edge
                        // — and, because the yield above suspended here, the one
                        // place a resumed stream pulls the next row.
                        self.pc = body.0 as usize;
                        return Ok(Flow::Next);
                    }
                }
                Op::KnnScan {
                    cur,
                    collection,
                    query,
                    k,
                    filter,
                } => {
                    let (cur, query, k, filter) = (*cur, *query, *k, *filter);
                    let db = db.ok_or(ExecError::Detached { op: "KnnScan" })?;
                    let vector = match self.reg(query) {
                        Some(RegValue::Vector(v)) => v.clone(),
                        _ => return Err(ExecError::NotAVector { reg: query.0 }),
                    };
                    let collection_id = db.collection_id(collection).map_err(ExecError::Engine)?;
                    let k = usize::try_from(k).map_err(|_| ExecError::TopKOverflow { k })?;
                    // The one coarse opcode: a whole SIMD top-k pass, not a
                    // per-element interpreted loop. `search` already returns
                    // most-similar first and excludes tombstones.
                    //
                    // With a `WHERE`, the bitmap goes IN to the ranking rather
                    // than being applied to its output: `TOP k` must count rows
                    // that satisfy the predicate, so filtering afterwards would
                    // return fewer than `k` whenever the nearest neighbours
                    // happen not to match — and look entirely plausible.
                    let hits = match filter {
                        None => db.search(collection_id, &vector, k),
                        Some(reg) => {
                            let bitmap = self.bitmap(reg)?;
                            db.search_where(collection_id, &vector, k, &bitmap)
                        }
                    }
                    .map_err(ExecError::Engine)?;
                    // SCORE ORDER IS THE PAYLOAD. Collect in the order `search`
                    // ranked them and hand that straight to the cursor, which
                    // imposes no order of its own — sorting here, or routing the
                    // ordinals through a bitmap, would silently make the result
                    // ascending and still look entirely plausible.
                    let ranked: Vec<Ordinal> = hits.iter().map(|hit| hit.id).collect();
                    // The scores travel with the cursor, keyed by ordinal, so
                    // `Op::Score` can pair each row with its OWN similarity —
                    // see `Slot::Read` for why a positional index would not do.
                    let scores: HashMap<Ordinal, f32> =
                        hits.into_iter().map(|hit| (hit.id, hit.score)).collect();
                    let cursor = db
                        .scan_over(collection_id, ranked.into_iter())
                        .map_err(ExecError::Engine)?;
                    self.open(
                        cur,
                        Slot::Read {
                            cursor: Box::new(cursor),
                            scores: Some(scores),
                        },
                    )?;
                }
                Op::RowId { cur, dst } => {
                    let (cur, dst) = (*cur, *dst);
                    let ordinal = self
                        .read_cursor(cur)
                        .ok_or(ExecError::NotAReadCursor { cur: cur.0 })?
                        .ordinal()
                        .ok_or(ExecError::CursorNotOnARow { cur: cur.0 })?;
                    self.store(dst, RegValue::Int(ordinal.0 as i64))?;
                }
                Op::Score { cur, dst } => {
                    let (cur, dst) = (*cur, *dst);
                    let (cursor, scores) = match self.cursors.get(cur.0 as usize) {
                        Some(Slot::Read { cursor, scores }) => (cursor, scores),
                        Some(Slot::Closed) | None => {
                            return Err(ExecError::CursorNotOpen { cur: cur.0 });
                        }
                        Some(Slot::Write { .. }) => {
                            return Err(ExecError::NotAReadCursor { cur: cur.0 });
                        }
                    };
                    // A plain scan produces no scores. Reporting that rather
                    // than handing back a zero keeps "no similarity was
                    // computed" from reading as "similarity 0".
                    let scores = scores.as_ref().ok_or(ExecError::NoScores { cur: cur.0 })?;
                    let ordinal = cursor
                        .ordinal()
                        .ok_or(ExecError::CursorNotOnARow { cur: cur.0 })?;
                    // Looked up BY THIS ROW'S ORDINAL — never by how many times
                    // the loop has gone round.
                    let score = scores
                        .get(&ordinal)
                        .ok_or(ExecError::NoScoreForRow { ordinal: ordinal.0 })?;
                    self.store(dst, RegValue::Real(*score as f64))?;
                }
                Op::Column { cur, col, dst } => {
                    let (cur, col, dst) = (*cur, *col, *dst);
                    let cursor = self
                        .read_cursor(cur)
                        .ok_or(ExecError::NotAReadCursor { cur: cur.0 })?;
                    // `col` is a STORAGE ColumnId; `Cursor::column` takes a
                    // POSITION IN THE PROJECTION. They coincide for a `Db::scan`
                    // cursor and will not for a narrower one, so ask the cursor
                    // what it projected instead of assuming.
                    let position = cursor
                        .columns()
                        .iter()
                        .position(|&projected| projected == col)
                        .ok_or(ExecError::ColumnNotProjected { cur: cur.0, col })?;
                    // Infallible in-memory read: the row was materialized whole
                    // by the advance that landed on it.
                    let value = cursor
                        .column(position)
                        .ok_or(ExecError::CursorNotOnARow { cur: cur.0 })?;
                    let value = RegValue::from(value);
                    self.store(dst, value)?;
                }

                // -- not built yet ---------------------------------------------
                // `todo!`, never a no-op. Each names the cycle that fills it.
                Op::VectorFetch { .. } => todo!("5b: embedding fetch"),

                // -- DML: pack a record, then write it through a write cursor --
                Op::OpenWrite { cur, collection } => {
                    let cur = *cur;
                    let db = db.ok_or(ExecError::Detached { op: "OpenWrite" })?;
                    // The name→id bridge (Prompt 4): bytecode carries a name
                    // because a program outlives any particular catalog state,
                    // but every engine method is keyed by id.
                    let collection_id = db.collection_id(collection).map_err(ExecError::Engine)?;
                    let schema = db
                        .collections()
                        .into_iter()
                        .find(|c| c.id == collection_id)
                        .map(|c| c.schema)
                        // `collection_id` just resolved it off the same
                        // snapshot, so this is a catalog race, not user error.
                        .ok_or(ExecError::Engine(Error::UnknownCollection {
                            id: collection_id,
                        }))?;
                    self.open(
                        cur,
                        Slot::Write {
                            collection: collection_id,
                            schema,
                        },
                    )?;
                }
                Op::MakeRecord { start, count, dst } => {
                    let (start, count, dst) = (*start, *count, *dst);
                    let mut values = Vec::with_capacity(count as usize);
                    for offset in 0..count {
                        let index = start.0 + offset;
                        let value = self
                            .regs
                            .get(index as usize)
                            .ok_or(ExecError::RegOutOfRange { reg: index })?;
                        // Registers pack in DECLARATION order — that ordering is
                        // the payload `split_record` walks, so nothing here
                        // sorts, filters, or reorders.
                        values.push(
                            value::to_literal(value).ok_or(ExecError::NotAValue { reg: index })?,
                        );
                    }
                    self.store(dst, RegValue::Record(Record { values }))?;
                }
                Op::Insert { cur, rec } => {
                    let (cur, rec) = (*cur, *rec);
                    let db = db.ok_or(ExecError::Detached { op: "Insert" })?;
                    let record = match self.reg(rec) {
                        Some(RegValue::Record(record)) => record.clone(),
                        _ => return Err(ExecError::NotARecord { reg: rec.0 }),
                    };
                    let (collection, schema) = match self.slot(cur)? {
                        Slot::Write { collection, schema } => (*collection, schema.clone()),
                        _ => return Err(ExecError::NotAWriteCursor { cur: cur.0 }),
                    };
                    // THE split (Prompt 4): one packed record in declaration
                    // order becomes the embedding plus a `ColumnId`-keyed row,
                    // because they go to different stores.
                    let (vector, row) = split_record(&record, &schema).map_err(ExecError::Split)?;
                    // Blocks until durable AND applied, so the row is readable
                    // through a cursor the moment this returns.
                    db.insert(collection, &vector, row)
                        .map_err(ExecError::Engine)?;
                }

                // -- DDL: one fat op, nothing about it varies at runtime -------
                Op::CreateCollection {
                    name,
                    schema,
                    capacity,
                } => {
                    let (schema_id, capacity) = (*schema, *capacity);
                    let db = db.ok_or(ExecError::Detached {
                        op: "CreateCollection",
                    })?;
                    // The schema the COMPILER lowered from this statement's DDL
                    // and interned (Prompt 3). There is never a second build of
                    // it — the pool is the single source.
                    let schema = match self.program.consts.get(schema_id) {
                        Some(Const::Schema(schema)) => schema.clone(),
                        other => return Err(ExecError::bad_const(schema_id.0, "a schema", other)),
                    };
                    let capacity = usize::try_from(capacity)
                        .map_err(|_| ExecError::CapacityOverflow { capacity })?;
                    // Blocks until the record is fsync'd AND applied, so on
                    // return the collection is durable and immediately usable.
                    db.create_collection(name, capacity, schema)
                        .map_err(ExecError::Engine)?;
                }
            }
        }
        // Fall-through ops advance by one. A jump sets `pc` itself and returns
        // `Flow::Next` directly rather than reaching here.
        self.pc += 1;
        Ok(Flow::Next)
    }

    /// The program counter — the index of the next instruction. Exposed so a
    /// test can assert WHERE a suspended machine is parked, which is the only
    /// direct evidence that `step` suspended rather than ran to completion.
    pub fn pc(&self) -> usize {
        self.pc
    }

    /// A register's current contents, or `None` if the index is out of range.
    pub fn reg(&self, reg: Reg) -> Option<&RegValue> {
        self.regs.get(reg.0 as usize)
    }

    /// Install a cursor in slot `cur`, replacing whatever was there (re-opening
    /// a slot drops the previous cursor, releasing its `Arc` on the reader).
    fn open(&mut self, cur: BytecodeCursor, slot: Slot) -> Result<(), ExecError> {
        match self.cursors.get_mut(cur.0 as usize) {
            Some(existing) => {
                *existing = slot;
                Ok(())
            }
            None => Err(ExecError::CursorOutOfRange { cur: cur.0 }),
        }
    }

    /// The slot behind `cur`, erroring if it was never opened — an op that
    /// reaches an unopened cursor is an emission-order bug, not a runtime state.
    fn slot(&self, cur: BytecodeCursor) -> Result<&Slot, ExecError> {
        match self.cursors.get(cur.0 as usize) {
            Some(Slot::Closed) | None => Err(ExecError::CursorNotOpen { cur: cur.0 }),
            Some(slot) => Ok(slot),
        }
    }

    /// The read cursor in slot `cur`, or `None` if the slot is closed or holds a
    /// write cursor.
    ///
    /// Public so a caller can inspect a live stream's progress — in particular
    /// [`Cursor::fetched`], which is how `select_is_lazy_not_materialized`
    /// proves the scan is lazy rather than asserting it.
    pub fn read_cursor(&self, cur: BytecodeCursor) -> Option<&Cursor<'static>> {
        match self.cursors.get(cur.0 as usize) {
            Some(Slot::Read { cursor, .. }) => Some(cursor),
            _ => None,
        }
    }

    /// The read cursor in slot `cur`, mutably — for the ops that ADVANCE it.
    fn read_cursor_mut(&mut self, cur: BytecodeCursor) -> Result<&mut Cursor<'static>, ExecError> {
        match self.cursors.get_mut(cur.0 as usize) {
            Some(Slot::Read { cursor, .. }) => Ok(cursor),
            Some(Slot::Closed) | None => Err(ExecError::CursorNotOpen { cur: cur.0 }),
            Some(Slot::Write { .. }) => Err(ExecError::NotAReadCursor { cur: cur.0 }),
        }
    }

    /// Write a register. `validate` proved the index is in range; the check
    /// stays so a bug surfaces as an error rather than a panic.
    fn store(&mut self, reg: Reg, value: RegValue) -> Result<(), ExecError> {
        match self.regs.get_mut(reg.0 as usize) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(ExecError::RegOutOfRange { reg: reg.0 }),
        }
    }

    /// The metadata-index reader for `collection`, by name.
    ///
    /// The bitmap ops carry a collection NAME rather than a cursor, because a
    /// predicate is evaluated before any cursor is open — its result is what
    /// the cursor will be opened over.
    fn metadata(&self, db: &Db, collection: &str) -> Result<meta::Reader, ExecError> {
        let id = db.collection_id(collection).map_err(ExecError::Engine)?;
        db.metadata_reader(id).ok_or(ExecError::Engine(
            crate::Error::UnknownCollectionName {
                name: collection.to_string(),
            },
        ))
    }

    /// Read a register as a metadata [`Value`] — a predicate's comparand.
    ///
    /// The binder type-checked the literal against its column, so the variant
    /// here always matches what the index expects; the error arms exist
    /// because a register is only proved in-range by `validate`, never proved
    /// to hold a particular type.
    fn value(&self, reg: Reg) -> Result<Value, ExecError> {
        match self.reg(reg) {
            Some(RegValue::Int(n)) => Ok(Value::Int(*n)),
            Some(RegValue::Real(f)) => Ok(Value::Float(*f)),
            Some(RegValue::Str(s)) => Ok(Value::Text(s.clone())),
            Some(RegValue::Unset) | None => Err(ExecError::UnsetRegister { reg: reg.0 }),
            // A vector, record, or bitmap as a comparand is an emitter bug: the
            // binder rejects predicates on the embedding, and nothing else can
            // produce one here.
            Some(_) => Err(ExecError::NotAComparand { reg: reg.0 }),
        }
    }

    /// Read a register as a bitmap.
    ///
    /// Clones the `Arc`, not the bitmap: the combining ops need their operands
    /// alive while they write a third register, and the borrow checker cannot
    /// see that `dst` differs from `a` and `b`.
    fn bitmap(&self, reg: Reg) -> Result<Arc<RoaringBitmap>, ExecError> {
        match self.reg(reg) {
            Some(RegValue::Bitmap(b)) => Ok(b.clone()),
            Some(RegValue::Unset) | None => Err(ExecError::UnsetRegister { reg: reg.0 }),
            Some(_) => Err(ExecError::NotABitmap { reg: reg.0 }),
        }
    }

    /// Read the register run `start .. start + count` into a row.
    ///
    /// An [`RegValue::Unset`] register is an ERROR, not an empty value:
    /// `validate` proves a register operand is in range but never that anything
    /// wrote it first, so this is the only place an emitter bug that yields an
    /// unloaded register can be caught.
    fn read_run(&self, start: Reg, count: u32) -> Result<OutputRow, ExecError> {
        let mut values = Vec::with_capacity(count as usize);
        for offset in 0..count {
            let index = start.0 + offset;
            match self.regs.get(index as usize) {
                Some(RegValue::Unset) => return Err(ExecError::UnsetRegister { reg: index }),
                Some(value) => values.push(value.clone()),
                None => return Err(ExecError::RegOutOfRange { reg: index }),
            }
        }
        Ok(OutputRow(values))
    }
}

/// Why execution could not proceed.
///
/// A stage-local error type (CLAUDE.md §5). Every variant here is a
/// *machine* fault — a program that does not mean what the VM can execute —
/// which is why none of them is a [`crate::Error`]: nothing here is an engine,
/// I/O, or durability failure. 5b adds the variants that wrap those, at the
/// boundary where the VM actually calls the engine.
#[derive(Debug)]
pub enum ExecError {
    /// The program failed [`Program::validate`] and was never loaded.
    Invalid(ValidateError),
    /// The program counter left the instruction stream.
    PcOutOfRange {
        /// Where it landed.
        pc: usize,
    },
    /// A register operand is wider than the register file.
    RegOutOfRange {
        /// The out-of-range register index.
        reg: u32,
    },
    /// A register was read before anything wrote it.
    UnsetRegister {
        /// The unwritten register index.
        reg: u32,
    },
    /// An op that needs a [`Db`] ran on a DETACHED machine (via
    /// [`Vm::resume`]). A read stream detaches after its cursors are open, so
    /// reaching this means a program touched storage later than
    /// [`Vm::open_cursors`] proved it would.
    Detached {
        /// The op's mnemonic.
        op: &'static str,
    },
    /// A program cannot be detached: an op after the cursor-opening prologue
    /// still needs a [`Db`]. Raised by [`Vm::open_cursors`] BEFORE any row is
    /// handed out, so the caller can report it as a compile-time failure rather
    /// than have a stream die halfway.
    CannotDetach {
        /// The offending op's mnemonic.
        op: &'static str,
        /// Where it sits in the instruction stream.
        at: usize,
    },
    /// `KnnScan`'s query register does not hold a vector.
    /// A register a bitmap op read does not hold a bitmap.
    NotABitmap {
        /// The offending register.
        reg: u32,
    },
    /// A register used as a predicate's comparand holds something that cannot
    /// be compared against a stored column (a vector, record, or bitmap).
    NotAComparand {
        /// The offending register.
        reg: u32,
    },
    NotAVector {
        /// The register.
        reg: u32,
    },
    /// `Score` was asked of a cursor that has no similarities — a plain scan
    /// rather than a ranked read.
    NoScores {
        /// The slot.
        cur: u8,
    },
    /// `Score` found no similarity for the row the cursor is parked on. The
    /// cursor and the score map disagree about which ordinals the query
    /// produced.
    NoScoreForRow {
        /// The offending ordinal.
        ordinal: u32,
    },
    /// A `TOP k` operand does not fit in a `usize` on this target.
    TopKOverflow {
        /// The operand.
        k: u64,
    },
    /// A capacity operand does not fit in a `usize` on this target.
    CapacityOverflow {
        /// The operand.
        capacity: u64,
    },
    /// The engine refused the operation. The one variant that is NOT a machine
    /// fault: the program was fine and storage said no (a duplicate collection
    /// name, a row that fails schema validation, a full collection, an I/O
    /// error). Kept as its own variant so the boundary stays visible — a
    /// `crate::Error` reaching here means the VM asked storage for something,
    /// not that the VM is confused.
    Engine(crate::Error),
    /// A cursor operand is wider than the cursor file.
    CursorOutOfRange {
        /// The out-of-range cursor slot.
        cur: u8,
    },
    /// An op reached a cursor slot that was never opened.
    CursorNotOpen {
        /// The slot.
        cur: u8,
    },
    /// `Insert` was pointed at something other than a write cursor.
    NotAWriteCursor {
        /// The slot.
        cur: u8,
    },
    /// A read op was pointed at something other than a read cursor.
    NotAReadCursor {
        /// The slot.
        cur: u8,
    },
    /// `Column` asked for a `ColumnId` the cursor's projection does not
    /// materialize — the compiler emitted a read the cursor was not opened for.
    ColumnNotProjected {
        /// The slot.
        cur: u8,
        /// The storage column id asked for.
        col: crate::metadata::common::ColumnId,
    },
    /// `Column` ran while the cursor was not parked on a row — a loop whose
    /// body was entered without a successful `SeekFirst`/`Next`.
    CursorNotOnARow {
        /// The slot.
        cur: u8,
    },
    /// `Insert`'s record operand does not hold a record.
    NotARecord {
        /// The register.
        reg: u32,
    },
    /// `MakeRecord` was handed a register with no value form — unwritten, or
    /// itself a record.
    NotAValue {
        /// The register.
        reg: u32,
    },
    /// A packed record could not be split into `(vector, row)`.
    Split(SplitError),
    /// A constant handle resolved to the wrong kind of payload (or to nothing).
    BadConst {
        /// The handle.
        id: u32,
        /// What the instruction needed.
        expected: &'static str,
        /// What the pool actually holds.
        found: &'static str,
    },
}

/// Any engine failure reaching the VM is an [`ExecError::Engine`]. Having the
/// conversion lets the cursor ops `?` a read error directly, which is what keeps
/// "a failed read" and "no more rows" from collapsing into one another —
/// `seek_first`/`next` return `Result<bool>`, so the error propagates and only
/// the *bool* decides the jump.
impl From<Error> for ExecError {
    fn from(e: Error) -> ExecError {
        ExecError::Engine(e)
    }
}

impl ExecError {
    /// Build a [`BadConst`](ExecError::BadConst), naming what the pool holds
    /// instead. Takes the looked-up `Option` directly so the two call sites do
    /// not each re-describe the payload kinds.
    fn bad_const(id: u32, expected: &'static str, found: Option<&Const>) -> ExecError {
        ExecError::BadConst {
            id,
            expected,
            found: match found {
                Some(Const::Vector(_)) => "a vector",
                Some(Const::Str(_)) => "a string",
                Some(Const::Schema(_)) => "a schema",
                None => "nothing",
            },
        }
    }
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::Invalid(e) => write!(f, "malformed program: {e}"),
            ExecError::PcOutOfRange { pc } => write!(f, "program counter {pc} is past the end"),
            ExecError::RegOutOfRange { reg } => write!(f, "no register r{reg}"),
            ExecError::UnsetRegister { reg } => {
                write!(f, "register r{reg} was read before it was written")
            }
            ExecError::CapacityOverflow { capacity } => {
                write!(f, "capacity {capacity} does not fit in a usize")
            }
            ExecError::Engine(e) => write!(f, "{e}"),
            ExecError::NotAVector { reg } => {
                write!(f, "register r{reg} does not hold a query vector")
            }
            ExecError::NotABitmap { reg } => {
                write!(f, "register r{reg} does not hold a bitmap")
            }
            ExecError::NotAComparand { reg } => {
                write!(f, "register r{reg} does not hold a comparable value")
            }
            ExecError::TopKOverflow { k } => write!(f, "TOP {k} does not fit in a usize"),
            ExecError::NoScores { cur } => {
                write!(f, "cursor {cur} is not a ranked read, so it has no scores")
            }
            ExecError::NoScoreForRow { ordinal } => {
                write!(f, "no similarity was computed for ordinal {ordinal}")
            }
            ExecError::Detached { op } => {
                write!(f, "{op} needs a database, but the program is detached")
            }
            ExecError::CannotDetach { op, at } => {
                write!(f, "{op} at op {at} needs a database after the cursors open")
            }
            ExecError::CursorOutOfRange { cur } => write!(f, "no cursor slot {cur}"),
            ExecError::CursorNotOpen { cur } => write!(f, "cursor {cur} was never opened"),
            ExecError::NotAWriteCursor { cur } => write!(f, "cursor {cur} is not a write cursor"),
            ExecError::NotAReadCursor { cur } => write!(f, "cursor {cur} is not a read cursor"),
            ExecError::ColumnNotProjected { cur, col } => {
                write!(f, "cursor {cur} does not project column {col}")
            }
            ExecError::CursorNotOnARow { cur } => write!(f, "cursor {cur} is not on a row"),
            ExecError::NotARecord { reg } => write!(f, "register r{reg} does not hold a record"),
            ExecError::NotAValue { reg } => {
                write!(f, "register r{reg} holds nothing that can go in a record")
            }
            ExecError::Split(e) => write!(f, "cannot split the record: {e}"),
            ExecError::BadConst {
                id,
                expected,
                found,
            } => write!(f, "constant #{id} should be {expected}, found {found}"),
        }
    }
}

impl std::error::Error for ExecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExecError::Invalid(e) => Some(e),
            ExecError::Engine(e) => Some(e),
            ExecError::Split(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecError, OutputRow, RegValue, Vm};
    use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
    use crate::compiler::constants::{Const, ConstPool};
    use crate::engine::{CollectionConfig, Db, DbOptions};
    use crate::error::Error;
    use crate::metadata::common::{ColumnSpec, ColumnType, Row, Schema, Value};
    use crate::vm::record::SplitError;
    use std::num::NonZeroUsize;

    /// A metadata row for `docs_schema` — keyed by storage `ColumnId`, which is
    /// NOT the declaration order: author #0, title #1, published_at #2 (the
    /// vector has no ColumnId).
    fn row(author: &str, title: &str, published_at: i64) -> Row {
        vec![
            (0, Value::Text(author.into())),
            (1, Value::Text(title.into())),
            (2, Value::Int(published_at)),
        ]
    }

    /// A program with an empty constant pool and no cursors — enough for every
    /// test that does not touch storage, which in 5a is all of them.
    fn program(ops: Vec<Op>, n_regs: u32) -> Program {
        Program {
            ops,
            consts: ConstPool::new(),
            n_regs,
            n_cursors: 0,
        }
    }

    fn vm(ops: Vec<Op>, n_regs: u32) -> Vm {
        Vm::new(program(ops, n_regs)).expect("hand-built program is well-formed")
    }

    /// A scratch database. The 5a programs never reach an arm that touches it,
    /// but `step` takes one because the storage arms do — so they open an empty
    /// one and ignore it. The `TempDir` is returned because dropping it would
    /// delete the directory out from under the open `Db`.
    fn scratch() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Db::open(dir.path(), &[], DbOptions::default()).expect("open");
        (dir, db)
    }

    // -- 1. registers hold what was stored --------------------------------

    #[test]
    fn register_load_read() {
        let (_dir, db) = scratch();
        // The ISA has no `LoadConst`: small scalars load INLINE (`Integer` /
        // `Real`), only large payloads come from the pool. `Integer` is the
        // simplest of the four load ops.
        let mut vm = vm(
            vec![
                Op::Integer {
                    value: 7,
                    dst: Reg(0),
                },
                Op::Halt,
            ],
            1,
        );

        // No `ResultRow`, so the loop runs straight to `Halt` and yields nothing.
        assert_eq!(vm.step(&db).expect("runs"), None);
        assert_eq!(vm.reg(Reg(0)), Some(&RegValue::Int(7)));
    }

    #[test]
    fn every_load_op_reaches_its_register() {
        let (_dir, db) = scratch();
        // All four loads in one program, so no load arm can be wired to the
        // wrong register or the wrong payload without a test noticing.
        let mut consts = ConstPool::new();
        let text = consts.add(Const::Str("alice".into()));
        let embedding = consts.add(Const::Vector(vec![0.5, -0.25]));

        let mut vm = Vm::new(Program {
            ops: vec![
                Op::Integer {
                    value: -1_700_000_000,
                    dst: Reg(0),
                },
                Op::Real {
                    value: 1.5,
                    dst: Reg(1),
                },
                Op::String {
                    id: text,
                    dst: Reg(2),
                },
                Op::VectorConst {
                    id: embedding,
                    dst: Reg(3),
                },
                Op::Halt,
            ],
            consts,
            n_regs: 4,
            n_cursors: 0,
        })
        .expect("well-formed");

        assert_eq!(vm.step(&db).expect("runs"), None);
        assert_eq!(vm.reg(Reg(0)), Some(&RegValue::Int(-1_700_000_000)));
        assert_eq!(vm.reg(Reg(1)), Some(&RegValue::Real(1.5)));
        assert_eq!(vm.reg(Reg(2)), Some(&RegValue::Str("alice".into())));
        // Seam (b): a register CAN hold a vector. Nothing in 5a produces one
        // from storage, but the pool already can, so the variant is exercised
        // rather than merely declared.
        assert_eq!(
            vm.reg(Reg(3)),
            Some(&RegValue::Vector(vec![0.5, -0.25].into()))
        );
    }

    // -- 2. the loop advances and stops -----------------------------------

    #[test]
    fn dispatch_terminates() {
        let (_dir, db) = scratch();
        let mut vm = vm(
            vec![
                Op::Integer {
                    value: 1,
                    dst: Reg(0),
                },
                Op::Real {
                    value: 2.5,
                    dst: Reg(1),
                },
                Op::Halt,
            ],
            2,
        );
        assert_eq!(vm.pc(), 0, "a fresh Vm starts at the first instruction");

        // One step runs every non-yielding op and stops at Halt — it does not
        // spin, and it does not stop early.
        assert_eq!(vm.step(&db).expect("runs"), None);
        assert_eq!(
            vm.pc(),
            2,
            "the pc parks ON the Halt, having advanced across both loads"
        );
        assert_eq!(vm.reg(Reg(0)), Some(&RegValue::Int(1)));
        assert_eq!(vm.reg(Reg(1)), Some(&RegValue::Real(2.5)));
    }

    #[test]
    fn halt_is_idempotent() {
        let (_dir, db) = scratch();
        // A finished stream stays finished. A caller draining with
        // `while let Some(row) = vm.step(&db)?` must not restart the program or
        // start erroring once it runs out of rows.
        let mut vm = vm(vec![Op::Halt], 0);
        for _ in 0..3 {
            assert_eq!(vm.step(&db).expect("runs"), None);
            assert_eq!(vm.pc(), 0, "a halted Vm does not walk past its Halt");
        }
    }

    // -- 3. THE streaming primitive ---------------------------------------

    #[test]
    fn step_yields_and_resumes() {
        let (_dir, db) = scratch();
        // Two ResultRows with a state change BETWEEN them. If `step` ran to
        // completion instead of suspending, the register would already hold the
        // second value by the time the first row came back — so this program
        // cannot pass by accident.
        let mut vm = vm(
            vec![
                Op::Integer {
                    value: 1,
                    dst: Reg(0),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Integer {
                    value: 2,
                    dst: Reg(0),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Halt,
            ],
            1,
        );

        // First row, and then it STOPS.
        assert_eq!(
            vm.step(&db).expect("runs"),
            Some(OutputRow(vec![RegValue::Int(1)]))
        );
        assert_eq!(
            vm.pc(),
            2,
            "suspended just AFTER the first ResultRow, so resuming re-enters the \
             stream where it left off rather than replaying the yield"
        );
        assert_eq!(
            vm.reg(Reg(0)),
            Some(&RegValue::Int(1)),
            "the instruction after the yield has NOT run — the Vm is suspended, \
             not merely reporting rows one at a time from a completed run"
        );

        // Resume: state carried across the call boundary in `self`.
        assert_eq!(
            vm.step(&db).expect("runs"),
            Some(OutputRow(vec![RegValue::Int(2)]))
        );
        assert_eq!(vm.pc(), 4);

        // Drained.
        assert_eq!(vm.step(&db).expect("runs"), None);
    }

    #[test]
    fn a_result_row_spans_a_register_run() {
        let (_dir, db) = scratch();
        // `ResultRow start, count` emits a RUN of registers — seam (c). The row
        // is built from registers, not read off a cursor, which is what later
        // lets `RETURNING id, score` mix a stored column with a computed one.
        let mut vm = vm(
            vec![
                Op::Integer {
                    value: 10,
                    dst: Reg(0),
                },
                Op::Integer {
                    value: 20,
                    dst: Reg(1),
                },
                Op::Integer {
                    value: 30,
                    dst: Reg(2),
                },
                // Deliberately NOT the whole register file: r0 is excluded.
                Op::ResultRow {
                    start: Reg(1),
                    count: 2,
                },
                Op::Halt,
            ],
            3,
        );
        assert_eq!(
            vm.step(&db).expect("runs"),
            Some(OutputRow(vec![RegValue::Int(20), RegValue::Int(30)]))
        );
    }

    // -- failure modes -----------------------------------------------------

    #[test]
    fn reading_an_unwritten_register_is_an_error() {
        let (_dir, db) = scratch();
        // `validate` proves every register operand is IN RANGE, never that it
        // was written first. An emitter bug that yields a register it never
        // loaded must surface, not read as a zero.
        let mut vm = vm(
            vec![
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Halt,
            ],
            1,
        );
        assert!(matches!(
            vm.step(&db),
            Err(ExecError::UnsetRegister { reg: 0 })
        ));
    }

    #[test]
    fn a_malformed_program_is_refused_before_it_runs() {
        // The Vm will not execute a program that fails `Program::validate` —
        // the structural check happens once, at construction, so the dispatch
        // loop never has to re-prove what validation already covers.
        let missing_halt = program(
            vec![Op::Integer {
                value: 1,
                dst: Reg(0),
            }],
            1,
        );
        assert!(matches!(Vm::new(missing_halt), Err(ExecError::Invalid(_))));
    }

    // ======================================================================
    // 5b sub-cycle 1 — CREATE. No reads, no cursor: the simplest arm that
    // touches storage at all.
    // ======================================================================

    /// The vector sits in the MIDDLE, where declaration ordinals and storage
    /// `ColumnId`s diverge — so a schema that survived interning intact is
    /// evidence about both numberings, not just about column names.
    fn docs_schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Scalar {
                name: "author".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(4).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "title".into(),
                ty: ColumnType::Text,
            },
            ColumnSpec::Scalar {
                name: "published_at".into(),
                ty: ColumnType::Int,
            },
        ])
        .unwrap()
    }

    /// `CREATE COLLECTION docs (...) WITH (capacity = 1000)` — the two-op
    /// program the compiler emits for DDL.
    fn create_program(name: &str, schema: Schema, capacity: u64) -> Program {
        let mut consts = ConstPool::new();
        let schema = consts.add(Const::Schema(schema));
        Program {
            ops: vec![
                Op::CreateCollection {
                    name: name.into(),
                    schema,
                    capacity,
                },
                Op::Halt,
            ],
            consts,
            n_regs: 0,
            n_cursors: 0,
        }
    }

    #[test]
    fn create_collection_program() {
        let (_dir, db) = scratch();
        let schema = docs_schema();
        let mut vm = Vm::new(create_program("docs", schema.clone(), 1000)).expect("well-formed");

        // DDL yields no rows — it runs to Halt in one step.
        assert_eq!(vm.step(&db).expect("creates"), None);

        // The catalog gained it, with the schema that was INTERNED — not one
        // rebuilt from the plan. This is the far end of Prompt 3's lowering.
        let collections = db.collections();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name, "docs");
        assert_eq!(collections[0].capacity, 1000);
        assert_eq!(
            collections[0].schema, schema,
            "the persisted schema is the one the program carried"
        );

        // ...and it is immediately usable through the normal engine API, which
        // is what makes a CREATE-then-INSERT session work.
        let id = db.collection_id("docs").expect("resolves by name");
        assert_eq!(id, collections[0].id);
        db.insert(id, &[1.0, 0.0, 0.0, 0.0], row("alice", "My doc", 1))
            .expect("the created collection accepts a row");

        db.close().unwrap();
    }

    #[test]
    fn a_refused_create_surfaces_the_engine_error() {
        let (_dir, db) = scratch();

        let mut first = Vm::new(create_program("docs", docs_schema(), 10)).expect("well-formed");
        assert_eq!(first.step(&db).expect("creates"), None);

        // The same program again. The engine refuses it, and that refusal must
        // arrive as an error rather than a panic or a silent second collection.
        let mut again = Vm::new(create_program("docs", docs_schema(), 10)).expect("well-formed");
        match again.step(&db) {
            Err(ExecError::Engine(Error::CollectionExists { name })) => assert_eq!(name, "docs"),
            other => panic!("expected CollectionExists, got {other:?}"),
        }
        assert_eq!(db.collections().len(), 1, "no half-created collection");

        db.close().unwrap();
    }

    // ======================================================================
    // 5b sub-cycle 2 — INSERT. Now a cursor is open, a record gets packed, and
    // the row is read back THROUGH THE PROMPT-1 CURSOR: the cursor is the
    // oracle, so green means the write path and the read path agree.
    // ======================================================================

    /// A database with `docs` already created — the state every INSERT/SELECT
    /// test starts from.
    fn docs_db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Db::open(
            dir.path(),
            &[CollectionConfig {
                id: 0,
                name: "docs".into(),
                capacity: 1024,
                schema: docs_schema(),
            }],
            DbOptions::default(),
        )
        .expect("open");
        (dir, db)
    }

    /// Every live row, counted by draining a fresh cursor. The control for the
    /// laziness tests: it proves a low `fetched()` means the VM stopped early,
    /// not that the collection was empty.
    fn live_rows(db: &Db) -> u64 {
        let mut cursor = db.scan(0).expect("collection 0 exists");
        let mut n = 0;
        let mut has_row = cursor.seek_first().expect("reads");
        while has_row {
            n += 1;
            has_row = cursor.next().expect("reads");
        }
        n
    }

    fn embedding(n: i64) -> Vec<f32> {
        vec![n as f32, 0.5, -0.25, 0.125]
    }

    /// The program the compiler emits for
    /// `INSERT INTO docs (author, vector, title, published_at) VALUES (...)`.
    /// Values load in DECLARATION order — author, vector, title, published_at —
    /// which is what makes `split_record`'s positional walk meaningful.
    fn insert_program(author: &str, n: i64) -> Program {
        let mut consts = ConstPool::new();
        let author = consts.add(Const::Str(author.into()));
        let vector = consts.add(Const::Vector(embedding(n)));
        let title = consts.add(Const::Str(format!("doc {n}")));
        Program {
            ops: vec![
                Op::OpenWrite {
                    cur: Cursor(0),
                    collection: "docs".into(),
                },
                Op::String {
                    id: author,
                    dst: Reg(0),
                },
                Op::VectorConst {
                    id: vector,
                    dst: Reg(1),
                },
                Op::String {
                    id: title,
                    dst: Reg(2),
                },
                Op::Integer {
                    value: n,
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
            n_regs: 5,
            n_cursors: 1,
        }
    }

    #[test]
    fn insert_then_scan() {
        let (_dir, db) = docs_db();

        let mut vm = Vm::new(insert_program("alice", 7)).expect("well-formed");
        assert_eq!(
            vm.step(&db).expect("inserts"),
            None,
            "INSERT yields no rows"
        );

        // Read it back through the cursor. Two independent paths agreeing is
        // the whole point: the record was packed in DECLARATION order and split
        // by `ColumnId`, and `scan` projects by `ColumnId` — so a split that
        // used positions instead would surface here as scrambled values.
        let mut cursor = db.scan(0).expect("collection 0 exists");
        assert!(cursor.seek_first().expect("reads"), "the row is live");
        assert_eq!(cursor.ordinal().map(|o| o.0), Some(0));
        assert_eq!(
            cursor.row().expect("parked on a row"),
            &[
                Value::Text("alice".into()),
                Value::Text("doc 7".into()),
                Value::Int(7),
            ]
        );
        assert!(!cursor.next().expect("reads"), "exactly one row");

        // The embedding went to the OTHER store, keyed by the same ordinal.
        let reader = db.reader(0).expect("collection 0 exists");
        assert_eq!(
            reader.vector_at(crate::metadata::common::Ordinal(0)),
            Some(&embedding(7)[..])
        );

        db.close().unwrap();
    }

    #[test]
    fn a_rejected_row_surfaces_the_engine_error() {
        let (_dir, db) = docs_db();

        // A 2-element embedding for a VECTOR(4) column. `split_record` catches
        // it first (Prompt 4 checks the dimension it gets from `locate`), so it
        // never reaches the WAL — which is the point: an unappliable record must
        // not become durable.
        let mut consts = ConstPool::new();
        let author = consts.add(Const::Str("alice".into()));
        let vector = consts.add(Const::Vector(vec![1.0, 2.0]));
        let title = consts.add(Const::Str("doc".into()));
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenWrite {
                    cur: Cursor(0),
                    collection: "docs".into(),
                },
                Op::String {
                    id: author,
                    dst: Reg(0),
                },
                Op::VectorConst {
                    id: vector,
                    dst: Reg(1),
                },
                Op::String {
                    id: title,
                    dst: Reg(2),
                },
                Op::Integer {
                    value: 1,
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
            n_regs: 5,
            n_cursors: 1,
        })
        .expect("well-formed");

        assert!(matches!(
            vm.step(&db),
            Err(ExecError::Split(SplitError::DimensionMismatch { .. }))
        ));
        // Nothing was written.
        let mut cursor = db.scan(0).expect("collection 0 exists");
        assert!(!cursor.seek_first().expect("reads"), "no row was stored");

        db.close().unwrap();
    }

    #[test]
    fn opening_an_unknown_collection_is_an_error() {
        let (_dir, db) = docs_db();
        // The name→id bridge is the first thing a cursor op does, so an unknown
        // name must surface here rather than as a panic deeper in.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenWrite {
                    cur: Cursor(0),
                    collection: "ghosts".into(),
                },
                Op::Halt,
            ],
            consts: ConstPool::new(),
            n_regs: 0,
            n_cursors: 1,
        })
        .expect("well-formed");
        assert!(matches!(
            vm.step(&db),
            Err(ExecError::Engine(Error::UnknownCollectionName { .. }))
        ));
        db.close().unwrap();
    }

    // ======================================================================
    // 5b sub-cycle 3 — SELECT. Where streaming meets storage: the VM stepper
    // and the Prompt-1 cursor are BOTH pull-based, so an N-row SELECT is N lazy
    // steps and never a buffer. The fetch counter is what proves that rather
    // than asserting it.
    // ======================================================================

    /// `SELECT author, published_at FROM docs` — the loop the compiler emits.
    /// `SeekFirst` is the entry guard (it jumps past the body when the scan is
    /// empty) and `Next` is the back-edge.
    ///
    /// The `Column` operands are STORAGE `ColumnId`s: author is #0 and
    /// published_at is #2 — declaration ordinals 0 and 3. Picking those two
    /// means a `Column` arm that used the operand as a projection position
    /// would read `title` instead of `published_at`.
    fn select_program() -> Program {
        Program {
            ops: vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".into(),
                    filter: None,
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
                Op::Column {
                    cur: Cursor(0),
                    col: 2,
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
            consts: ConstPool::new(),
            n_regs: 2,
            n_cursors: 1,
        }
    }

    /// Seed `n` rows whose contents are a pure function of their index.
    fn seed(db: &Db, n: i64) {
        for i in 0..n {
            db.insert(0, &embedding(i), row("alice", &format!("doc {i}"), i))
                .expect("seeded row inserts");
        }
    }

    #[test]
    fn select_streams_rows() {
        let (_dir, db) = docs_db();
        seed(&db, 5);

        let mut vm = Vm::new(select_program()).expect("well-formed");

        // One step, one row — in ordinal order, with the right values.
        for i in 0..5 {
            assert_eq!(
                vm.step(&db).expect("reads"),
                Some(OutputRow(vec![
                    RegValue::Str("alice".into()),
                    RegValue::Int(i),
                ])),
                "row {i}"
            );
        }
        // Then the loop falls out of `Next` and halts.
        assert_eq!(vm.step(&db).expect("reads"), None);
        assert_eq!(vm.step(&db).expect("reads"), None, "and stays drained");

        db.close().unwrap();
    }

    #[test]
    fn select_over_an_empty_collection_yields_nothing() {
        let (_dir, db) = docs_db();
        // `SeekFirst`'s whole job: jump PAST the body rather than run it once
        // against a cursor parked on nothing.
        let mut vm = Vm::new(select_program()).expect("well-formed");
        assert_eq!(vm.step(&db).expect("reads"), None);
        assert_eq!(vm.pc(), 6, "jumped straight to the Halt");
        db.close().unwrap();
    }

    #[test]
    fn select_is_lazy_not_materialized() {
        const SEEDED: i64 = 100;
        const CONSUMED: u64 = 3;

        let (_dir, db) = docs_db();
        seed(&db, SEEDED);

        let mut vm = Vm::new(select_program()).expect("well-formed");
        for _ in 0..CONSUMED {
            assert!(vm.step(&db).expect("reads").is_some());
        }

        // THE PROOF. `fetched` is incremented at the one place the tuple store
        // returns a live row, so it IS the number of storage reads issued. Stop
        // pulling after 3 of 100 and storage has been asked for 3.
        let cursor = vm.read_cursor(Cursor(0)).expect("the scan cursor is open");
        assert_eq!(
            cursor.fetched(),
            CONSUMED,
            "consuming {CONSUMED} rows must read {CONSUMED} rows out of the tuple \
             store, not all {SEEDED} — an eager loop shows {SEEDED} here"
        );
        assert_eq!(cursor.skipped(), 0, "nothing was mid-apply or deleted");

        // Sanity: the rows really are all there, so the low count is laziness
        // and not an empty collection.
        assert_eq!(live_rows(&db), SEEDED as u64);

        db.close().unwrap();
    }

    #[test]
    fn select_stream_drops_cleanly() {
        const SEEDED: i64 = 100;

        let (_dir, db) = docs_db();
        seed(&db, SEEDED);

        let fetched = {
            let mut vm = Vm::new(select_program()).expect("well-formed");
            assert!(vm.step(&db).expect("reads").is_some());
            assert!(vm.step(&db).expect("reads").is_some());
            // Read the count out before the block ends — `vm` drops with it.
            vm.read_cursor(Cursor(0))
                .expect("the scan cursor is open")
                .fetched()
            // `vm` drops here, mid-scan, with 98 rows unread. Nothing to unwind:
            // the cursor's tuple handle is an `Arc` clone, so dropping it is a
            // refcount decrement, and the ordinal iterator is a plain `Vec`
            // walk. No panic, no leak, no half-finished read to abandon.
        };
        assert_eq!(fetched, 2, "only the consumed rows were ever fetched");

        // The database is unaffected by the abandoned stream: a fresh scan still
        // sees everything, so nothing was consumed, locked, or poisoned.
        assert_eq!(live_rows(&db), SEEDED as u64);

        db.close().unwrap();
    }

    #[test]
    fn column_selects_the_named_column() {
        let (_dir, db) = docs_db();
        db.insert(0, &embedding(1), row("alice", "the title", 42))
            .expect("inserts");

        // Read ONLY published_at, ColumnId #2 — the last scalar, so a `Column`
        // arm that ignored its operand and took the first value would hand back
        // "alice".
        //
        // HONEST LIMIT: this does NOT prove `Column` resolves the ColumnId
        // through the cursor's projection rather than indexing by it directly.
        // It cannot: `Db::scan` projects every scalar in ColumnId order, so
        // position and ColumnId coincide for every cursor the VM can currently
        // build, and the two implementations are observationally identical. The
        // lookup is defensive against the first narrower projection (a `WHERE`
        // or KNN cursor); `column_outside_the_projection_is_an_error` below is
        // the one behaviour that does distinguish them today.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".into(),
                    filter: None,
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 2,
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
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");

        assert_eq!(
            vm.step(&db).expect("reads"),
            Some(OutputRow(vec![RegValue::Int(42)]))
        );
        db.close().unwrap();
    }

    #[test]
    fn column_outside_the_projection_is_an_error() {
        let (_dir, db) = docs_db();
        db.insert(0, &embedding(1), row("alice", "the title", 42))
            .expect("inserts");

        // `Program::validate` checks register and cursor operands but NOT
        // ColumnIds, so a bad one reaches the arm. It must name the real problem
        // — the cursor does not project that column — rather than surface as
        // "not on a row", which would send a reader hunting the loop structure.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".into(),
                    filter: None,
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::Column {
                    cur: Cursor(0),
                    col: 99,
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
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");

        assert!(matches!(
            vm.step(&db),
            Err(ExecError::ColumnNotProjected { cur: 0, col: 99 })
        ));
        db.close().unwrap();
    }

    #[test]
    fn score_on_a_plain_scan_is_an_error() {
        let (_dir, db) = docs_db();
        db.insert(0, &embedding(1), row("alice", "doc", 1))
            .expect("inserts");

        // `Op::Score` only means something on a cursor `KnnScan` opened. A plain
        // scan computed no similarities, and reporting that beats handing back a
        // `0.0` that reads as "perfectly dissimilar" — a caller cannot tell the
        // two apart, and every row would silently score the same.
        //
        // Unreachable through SQL (the binder only accepts `score` inside a
        // SEARCH `RETURNING`), so hand-built bytecode is the only way to reach
        // the guard at all.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".into(),
                    filter: None,
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::Score {
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
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");

        assert!(matches!(vm.step(&db), Err(ExecError::NoScores { cur: 0 })));
        db.close().unwrap();
    }

    #[test]
    fn row_id_works_on_a_plain_scan() {
        let (_dir, db) = docs_db();
        for i in 0..3 {
            db.insert(0, &embedding(i), row("alice", "doc", i))
                .expect("inserts");
        }

        // `RowId`, unlike `Score`, is meaningful on ANY read cursor — the
        // ordinal is the cursor's own position, not something a ranked query
        // computed. Nothing in V-SQL emits it outside a SEARCH yet, but the op
        // is not KNN-specific and should not pretend to be.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::OpenRead {
                    cur: Cursor(0),
                    collection: "docs".into(),
                    filter: None,
                },
                Op::SeekFirst {
                    cur: Cursor(0),
                    end: Addr(5),
                },
                Op::RowId {
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
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");

        for expected in 0..3 {
            assert_eq!(
                vm.step(&db).expect("reads"),
                Some(OutputRow(vec![RegValue::Int(expected)]))
            );
        }
        assert_eq!(vm.step(&db).expect("reads"), None);
        db.close().unwrap();
    }

    #[test]
    #[should_panic(expected = "5b")]
    fn an_unbuilt_opcode_panics_loudly() {
        let (_dir, db) = scratch();
        // The out-of-scope arms are `todo!`, NOT silent no-ops. A cursor op that
        // quietly did nothing would make `SELECT` return zero rows and look like
        // an empty collection — the single most expensive way to be wrong here.
        let mut vm = Vm::new(Program {
            ops: vec![
                Op::VectorFetch {
                    cur: Cursor(0),
                    dst: Reg(0),
                },
                Op::Halt,
            ],
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");
        let _ = vm.step(&db);
    }
}
