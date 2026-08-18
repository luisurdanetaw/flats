// ============================================================================
// V-SQL bytecode instruction set
// ============================================================================
//
// Register-based VM. A program is a flat Vec<Op> executed
// by a single dispatch loop: `while pc < len { match program[pc] { .. } }`.
//
// CONVENTIONS
//   cur      cursor slot (u8). Usually exactly one — V-SQL has no joins.
//   r, dst   register index. Registers hold SMALL tagged values only.
//   #N       handle into the program's CONSTANT POOL. Used for anything large
//            (vector literals, schemas, long strings). A 768-float embedding
//            NEVER lives in a register — the register holds the handle.
//   →addr    absolute instruction index. Emitted as a placeholder and
//            BACKPATCHED once the target address is known (forward jumps can't
//            be resolved until the body they skip has been generated).
//
// Jump-taking instructions MAY FALL THROUGH. That is what lets a loop work
// without a separate condition check: SeekFirst is the loop's entry guard
// (empty collection -> skip the body entirely) and Next is its back-edge
// (another row -> repeat). Both decide whether to jump or fall through.
//
// COLUMN NUMBERING: the `colId` operand of Column is a STORAGE ColumnId
// (scalar-only space, vector excluded). NEVER a plan/declaration ordinal.
// The compiler translates via the schema lookup before emitting. The bytecode
// speaks storage's numbering, never the planner's.
//
// ---------------------------------------------------------------------------
// CORE SET — everything needed for CREATE COLLECTION / SELECT / INSERT
// ---------------------------------------------------------------------------
//
// Cursors & scanning
//   OpenRead     cur, collection      open a READ cursor, positioned before the
//                                     first row
//   OpenWrite    cur, collection      same, with write access (DML)
//   SeekFirst    cur, →end            move to first row; if the collection is
//                                     EMPTY jump to `end`, else fall through
//   Next         cur, →body           advance the cursor; if a row EXISTS jump
//                                     back to `body`, else fall through
//   Column       cur, colId, dst      read column `colId` of the current row
//                                     into register `dst` (storage ColumnId)
//   VectorFetch  cur, dst             fetch the current row's embedding from the
//                                     FLAT INDEX into `dst` (as a handle).
//                                     Emitted ONLY when include_vector is true —
//                                     the vector is never read via Column.
//
//   RowId        cur, dst             read the current row's ORDINAL (its id)
//                                     into `dst`
//   Score        cur, dst             read the current row's similarity score
//                                     into `dst`. KnnScan cursors only.
//
// Loading values
//   Integer      value, dst           load an i64 literal
//   Real         value, dst           load an f64 literal
//   String       #N, dst              load a string from the constant pool
//   VectorConst  #N, dst              load a vector literal handle from the pool
//
// Output & mutation
//   ResultRow    start, count         emit registers start..start+count as one
//                                     output row
//   MakeRecord   start, count, dst    pack registers start..start+count into a
//                                     record in `dst`
//   Insert       cur, rec             write the record in `rec` through the
//                                     cursor: WAL + tuple store + bitmap index
//                                     + flat index
//   KnnScan      cur, collection, query, k
//                                     SIMD top-k over the flat index. Opens
//                                     `cur` over the winning ordinals IN SCORE
//                                     ORDER (nearest first). Occupies the same
//                                     slot OpenRead does for a SELECT.
//   CreateCollection name, #schema, capacity
//                                     provision a collection: WAL append,
//                                     catalog write, and allocation of all three
//                                     stores. One fat op — nothing about DDL
//                                     varies at runtime, so there is nothing to
//                                     interpret.
//
// WHERE — predicates are SET ALGEBRA, not per-row comparisons
//   BitmapEq     collection, colId, valueReg, dst
//                                     ordinals where col == value
//   BitmapRange  collection, colId, rangeOp, valueReg, dst
//                                     ordinals where col <op> value
//   BitmapAnd    rA, rB, dst          intersection
//   BitmapOr     rA, rB, dst          union
//   BitmapNot    collection, r, dst   complement WITHIN the live set
//
//   There are deliberately NO `Lt`/`Ge`/`JumpIfFalse` opcodes and no per-row
//   predicate evaluation. Every V-SQL predicate is `col <op> literal` joined by
//   AND/OR, and the metadata index answers exactly that shape as a roaring
//   bitmap — so a WHERE clause is a handful of set operations producing an
//   ORDINAL SOURCE, which is precisely what a cursor already consumes.
//   `OpenRead`/`KnnScan` take that bitmap as an optional `filter` operand.
//
//   Same granularity argument as KnnScan below: interpreting a comparison per
//   row per predicate would be thousands of dispatches where the index does one
//   sorted-range walk. The day a predicate arrives that the index CANNOT
//   answer (arithmetic, column-to-column) is the day per-row comparison
//   opcodes earn their place — and the DESIGN NOTE that used to sit here,
//   agonizing over comparison-with-branch vs. boolean-producing comparisons,
//   is the decision to make then.
//
// Control
//   Halt                              stop execution
//
// 24 opcodes = a working engine, vector search, RETURNING and WHERE included.
//
// ---------------------------------------------------------------------------
// EXTEND: — not built yet. Added one statement at a time, as emission demands.
// ---------------------------------------------------------------------------
//
// DML
//   Delete       cur                  delete the row under the cursor
//   Update       cur, rec             replace the row under the cursor
//
// SEARCH — the rest
//   GRANULARITY NOTE: KnnScan is deliberately COARSE. Interpreting the distance
//   loop per-element would be catastrophic — 100k distances must be one tight
//   SIMD loop inside one opcode, not 100k dispatches. This is the one place the
//   "one opcode = one operation on one row" rule is intentionally broken.
//
//   SCORES reach the caller through `Score`, which parks one in a register that
//   `ResultRow` already knows how to emit — no new output machinery, because
//   `ResultRow` reads REGISTERS rather than columns (seam (c)). That is also why
//   `RETURNING title, score` costs nothing extra: by the time the row is emitted
//   a stored column and a computed one are indistinguishable.
//
// LIMIT
//   SetCounter      k, r              initialize a counter register
//   DecrJumpIfZero  r, →end           decrement; jump out when exhausted
//
// ---------------------------------------------------------------------------
// WORKED EXAMPLES — the target programs these opcodes were derived from
// ---------------------------------------------------------------------------
//
//   CREATE COLLECTION docs (...) WITH (capacity = 1000000);
//
//     0: CreateCollection "docs", #0, 1000000
//     1: Halt
//
//   Nothing varies at runtime, so DDL is one op. No cursor, no loop, no body.
//
//
//   SELECT x, y FROM docs;
//
//     0: OpenRead  cur0, docs
//     1: SeekFirst cur0, →6          // empty collection -> skip to Halt
//     2:   Column  cur0, 0, r1       // ┐
//     3:   Column  cur0, 1, r2       // │ loop BODY — runs once per row
//     4:   ResultRow r1, 2           // ┘
//     5: Next      cur0, →2          // advance; another row -> back to body
//     6: Halt
//
//   Note the shape: SeekFirst/Next are the loop's TOP and BOTTOM, and the
//   projection's code sits BETWEEN them because reading columns happens once
//   per row. This is why the compiler recurses — the loop's closing op cannot
//   be emitted until the body it wraps has been generated.
//
//
//   INSERT INTO docs (vector, author, title, published_at)
//   VALUES ([0.1, 0.2, 0.3], 'alice', 'My doc', 1700000000);
//
//     0: OpenWrite   cur0, docs
//     1: VectorConst #1, r1          // handle — not 768 floats in a register
//     2: String      #2, r2          // 'alice'
//     3: String      #3, r3          // 'My doc'
//     4: Integer     1700000000, r4
//     5: MakeRecord  r1, 4, r5       // pack r1..r4 into a record
//     6: Insert      cur0, r5
//     7: Halt
//
//   NO SeekFirst, NO Next — INSERT does not walk existing rows, so it has no
//   loop. That structural difference falls straight out of the fact that the
//   imperative version never says "for each row".
//
// ---------------------------------------------------------------------------
// HOW NEW OPCODES GET DISCOVERED (do not design the set upfront)
// ---------------------------------------------------------------------------
//
//   1. Write the statement's execution as plain imperative pseudocode.
//   2. Each line becomes one opcode. A line doing three things splits into three.
//   3. "hold this value" -> a register operand
//      (large value -> constant-pool handle instead)
//   4. "if / repeat"     -> a jump operand (backpatched if forward)
//   5. nothing varies at runtime -> one fat op
//
//   Smells: always emitting X and Y adjacently -> fuse them. An op with a flag
//   that switches its behavior, or that can't be reused by a second statement
//   -> it's really several ops.
// ============================================================================

use std::fmt;

use crate::compiler::constants::{ConstId, ConstPool};
use crate::metadata::common::{ColumnId, RangeOp};

// ---------------------------------------------------------------------------
// Operand newtypes
//
// Each operand space is a distinct newtype so the type system keeps them apart:
// a register index can never be passed where a cursor slot, jump target, or
// constant handle is expected. ([`ConstId`] lives in the constants module
// alongside the pool it indexes.)
// ---------------------------------------------------------------------------

/// A register index. Registers hold small tagged values or constant-pool
/// handles — never large payloads. Allocated by a monotonic counter in the
/// compiler; [`Program::n_regs`] is the final count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

/// A cursor slot. V-SQL has no joins, so in practice a program uses exactly one;
/// [`Program::n_cursors`] is the final count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cursor(pub u8);

/// An absolute instruction index — a jump target. A forward jump is emitted as
/// [`Addr::PLACEHOLDER`] and BACKPATCHED once its target address is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Addr(pub u32);

impl Addr {
    /// The sentinel a forward jump carries until it is backpatched. A finished
    /// program that still contains one is malformed — [`Program::validate`]
    /// rejects it as an unpatched jump.
    pub const PLACEHOLDER: Addr = Addr(u32::MAX);
}

// ---------------------------------------------------------------------------
// The instruction set
// ---------------------------------------------------------------------------

/// One bytecode instruction. See the module header for the full opcode
/// reference and the worked target programs.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Open a READ cursor on `collection`, positioned before the first row.
    OpenRead {
        /// Cursor slot to open.
        cur: Cursor,
        /// Collection name.
        collection: String,
        /// A register holding the `WHERE`-derived bitmap, or `None` to walk
        /// every live row.
        ///
        /// An `Option` operand rather than a second opcode: this selects the
        /// cursor's ORDINAL SOURCE, it does not switch what the instruction
        /// does. (Contrast the module header's "a flag that switches its
        /// behavior means it's really several ops" — that is about behavior,
        /// not data.)
        filter: Option<Reg>,
    },
    /// Open a WRITE cursor on `collection` (DML).
    OpenWrite {
        /// Cursor slot to open.
        cur: Cursor,
        /// Collection name.
        collection: String,
    },
    /// Move to the first row; if the collection is EMPTY jump to `end`, else
    /// fall through into the loop body. The loop's entry guard.
    SeekFirst {
        /// Cursor to seek.
        cur: Cursor,
        /// Where to jump when there is no first row (loop exit).
        end: Addr,
    },
    /// Advance the cursor; if another row EXISTS jump back to `body`, else fall
    /// through past the loop. The loop's back-edge.
    Next {
        /// Cursor to advance.
        cur: Cursor,
        /// First instruction of the loop body (jump target on "row exists").
        body: Addr,
    },
    /// Read scalar column `col` of the current row into `dst`. `col` is a
    /// STORAGE [`ColumnId`] (scalar-only space) — never a declaration ordinal.
    Column {
        /// Cursor whose current row is read.
        cur: Cursor,
        /// Storage column id (scalar-only numbering).
        col: ColumnId,
        /// Destination register.
        dst: Reg,
    },
    /// Fetch the current row's embedding from the flat index into `dst` (as a
    /// handle). Emitted ONLY when `include_vector` is set.
    VectorFetch {
        /// Cursor whose current row's embedding is fetched.
        cur: Cursor,
        /// Destination register (receives a handle).
        dst: Reg,
    },
    /// Read the current row's ORDINAL — its stable id within the collection —
    /// into `dst`.
    ///
    /// A per-row read off a cursor, like [`Op::Column`], but the value is the
    /// row's identity rather than one of its stored columns. No existing
    /// register-load covers it: `Integer` is a compile-time literal and `Column`
    /// reads the tuple store, while the ordinal is the cursor's own position.
    RowId {
        /// Cursor whose current row is read.
        cur: Cursor,
        /// Destination register.
        dst: Reg,
    },
    /// Read the current row's SIMILARITY SCORE into `dst`.
    ///
    /// Valid only on a cursor opened by [`Op::KnnScan`] — a plain scan has no
    /// scores, and asking for one is an error rather than a zero. Separate from
    /// [`Op::RowId`] rather than one op with a selector flag: an op whose
    /// behavior switches on a flag is really several ops (see the module
    /// header's smells).
    Score {
        /// Cursor whose current row is read.
        cur: Cursor,
        /// Destination register.
        dst: Reg,
    },
    /// Load an `i64` literal into `dst`.
    Integer {
        /// The literal value.
        value: i64,
        /// Destination register.
        dst: Reg,
    },
    /// Load an `f64` literal into `dst`.
    Real {
        /// The literal value.
        value: f64,
        /// Destination register.
        dst: Reg,
    },
    /// Load a pooled string into `dst`.
    String {
        /// Constant-pool handle of the string.
        id: ConstId,
        /// Destination register.
        dst: Reg,
    },
    /// Load a pooled vector-literal handle into `dst`.
    VectorConst {
        /// Constant-pool handle of the vector literal.
        id: ConstId,
        /// Destination register.
        dst: Reg,
    },
    /// Emit registers `start .. start + count` as one output row.
    ResultRow {
        /// First register of the run.
        start: Reg,
        /// How many registers the row spans.
        count: u32,
    },
    /// Pack registers `start .. start + count` into a record in `dst`.
    MakeRecord {
        /// First register of the run.
        start: Reg,
        /// How many registers the record spans.
        count: u32,
        /// Destination register (receives the record).
        dst: Reg,
    },
    /// Write the record in `rec` through cursor `cur` (WAL + tuple store +
    /// bitmap index + flat index).
    Insert {
        /// Write cursor.
        cur: Cursor,
        /// Register holding the record to write.
        rec: Reg,
    },
    /// SIMD top-`k` over the flat index: rank `collection`'s rows by similarity
    /// to the query vector in `query`, and open `cur` over the winning ordinals
    /// **in score order, nearest first**.
    ///
    /// Deliberately COARSE (see the module header's granularity note): the
    /// distance loop is one tight SIMD pass inside one opcode, never 100k
    /// dispatches. It occupies exactly the slot `OpenRead` does for a `SELECT` —
    /// it opens the read cursor — so everything after it is the same loop.
    KnnScan {
        /// Cursor slot to open over the ranked ordinals.
        cur: Cursor,
        /// Collection to search.
        collection: String,
        /// Register holding the query vector.
        query: Reg,
        /// How many nearest rows to take (`>= 1`, enforced by the binder).
        k: u64,
        /// A register holding the `WHERE`-derived bitmap, or `None` to rank
        /// every live row.
        ///
        /// A PREFILTER: rows outside it never enter the top-`k` heap, so `k`
        /// counts rows that satisfy the predicate.
        filter: Option<Reg>,
    },
    /// Ordinals where `col == value`, as a bitmap in `dst`.
    ///
    /// The comparand comes from a REGISTER, loaded by the same `Integer` /
    /// `Real` / `String` ops an `INSERT` uses — so a predicate literal and an
    /// inserted literal travel by exactly one mechanism.
    BitmapEq {
        /// Collection whose metadata index is consulted. Carried on the op
        /// because a bitmap is built BEFORE any cursor is open — there is no
        /// cursor operand to take the collection from.
        collection: String,
        /// Storage column id (scalar-only numbering).
        col: ColumnId,
        /// Register holding the comparand.
        value: Reg,
        /// Destination register (receives the bitmap).
        dst: Reg,
    },
    /// Ordinals where `col <op> value`, as a bitmap in `dst`.
    ///
    /// Ordered comparison only — `=` is [`Op::BitmapEq`], because the metadata
    /// index answers the two by different structures (a dictionary vs. a
    /// sorted key range).
    BitmapRange {
        /// Collection whose metadata index is consulted.
        collection: String,
        /// Storage column id (scalar-only numbering).
        col: ColumnId,
        /// Which ordered comparison.
        op: RangeOp,
        /// Register holding the comparand.
        value: Reg,
        /// Destination register (receives the bitmap).
        dst: Reg,
    },
    /// Intersection of two bitmaps — `AND`.
    BitmapAnd {
        /// Left operand register.
        a: Reg,
        /// Right operand register.
        b: Reg,
        /// Destination register.
        dst: Reg,
    },
    /// Union of two bitmaps — `OR`.
    BitmapOr {
        /// Left operand register.
        a: Reg,
        /// Right operand register.
        b: Reg,
        /// Destination register.
        dst: Reg,
    },
    /// Complement of a bitmap WITHIN THE LIVE SET — how `!=` is built.
    ///
    /// Complementing against the live rows rather than the whole ordinal space
    /// is what keeps tombstones out: a deleted row satisfies no predicate, so
    /// negating `a = 1` must not resurrect it.
    BitmapNot {
        /// Collection whose live set bounds the complement.
        collection: String,
        /// Register holding the bitmap to complement.
        src: Reg,
        /// Destination register.
        dst: Reg,
    },
    /// Provision a collection. One fat op — nothing about DDL varies at runtime.
    CreateCollection {
        /// New collection name.
        name: String,
        /// Constant-pool handle of the resolved schema.
        schema: ConstId,
        /// Requested capacity.
        capacity: u64,
    },
    /// Stop execution.
    Halt,
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// A compiled program: the instruction stream plus its constant pool and the
/// register/cursor counts the VM allocates before executing.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// The instruction stream, executed by a single dispatch loop.
    pub ops: Vec<Op>,
    /// Side table of values too large for a register (see [`ConstPool`]).
    pub consts: ConstPool,
    /// Number of registers the program uses (the compiler's final counter).
    pub n_regs: u32,
    /// Number of cursor slots the program uses (the compiler's final counter).
    pub n_cursors: u8,
}

impl Program {
    /// Structural well-formedness check, independent of what any instruction
    /// *does*. A program the compiler emits must pass this before the VM runs
    /// it. Checks, in order:
    ///
    ///  * the stream is non-empty and ends in [`Op::Halt`];
    ///  * no jump still carries [`Addr::PLACEHOLDER`] (every forward jump was
    ///    backpatched) and every jump target is in range;
    ///  * every register operand is `< n_regs` and every cursor `< n_cursors`;
    ///  * every constant handle resolves inside the pool.
    pub fn validate(&self) -> Result<(), ValidateError> {
        match self.ops.last() {
            None => return Err(ValidateError::Empty),
            Some(Op::Halt) => {}
            Some(_) => return Err(ValidateError::NotHalted),
        }

        for (at, op) in self.ops.iter().enumerate() {
            match op {
                Op::OpenRead { cur, filter, .. } => {
                    self.check_cursor(at, *cur)?;
                    if let Some(r) = filter {
                        self.check_reg(at, *r)?;
                    }
                }
                Op::OpenWrite { cur, .. } => {
                    self.check_cursor(at, *cur)?;
                }
                Op::SeekFirst { cur, end } => {
                    self.check_cursor(at, *cur)?;
                    self.check_addr(at, *end)?;
                }
                Op::Next { cur, body } => {
                    self.check_cursor(at, *cur)?;
                    self.check_addr(at, *body)?;
                }
                Op::Column { cur, dst, .. } => {
                    self.check_cursor(at, *cur)?;
                    self.check_reg(at, *dst)?;
                }
                Op::VectorFetch { cur, dst } | Op::RowId { cur, dst } | Op::Score { cur, dst } => {
                    self.check_cursor(at, *cur)?;
                    self.check_reg(at, *dst)?;
                }
                Op::Integer { dst, .. } | Op::Real { dst, .. } => {
                    self.check_reg(at, *dst)?;
                }
                Op::String { id, dst } | Op::VectorConst { id, dst } => {
                    self.check_const(at, *id)?;
                    self.check_reg(at, *dst)?;
                }
                Op::ResultRow { start, count } => {
                    self.check_reg_run(at, *start, *count)?;
                }
                Op::MakeRecord { start, count, dst } => {
                    self.check_reg_run(at, *start, *count)?;
                    self.check_reg(at, *dst)?;
                }
                Op::Insert { cur, rec } => {
                    self.check_cursor(at, *cur)?;
                    self.check_reg(at, *rec)?;
                }
                Op::KnnScan {
                    cur, query, filter, ..
                } => {
                    self.check_cursor(at, *cur)?;
                    self.check_reg(at, *query)?;
                    if let Some(r) = filter {
                        self.check_reg(at, *r)?;
                    }
                }
                Op::BitmapEq { value, dst, .. } | Op::BitmapRange { value, dst, .. } => {
                    self.check_reg(at, *value)?;
                    self.check_reg(at, *dst)?;
                }
                Op::BitmapAnd { a, b, dst } | Op::BitmapOr { a, b, dst } => {
                    self.check_reg(at, *a)?;
                    self.check_reg(at, *b)?;
                    self.check_reg(at, *dst)?;
                }
                Op::BitmapNot { src, dst, .. } => {
                    self.check_reg(at, *src)?;
                    self.check_reg(at, *dst)?;
                }
                Op::CreateCollection { schema, .. } => {
                    self.check_const(at, *schema)?;
                }
                Op::Halt => {}
            }
        }
        Ok(())
    }

    fn check_reg(&self, at: usize, r: Reg) -> Result<(), ValidateError> {
        if r.0 < self.n_regs {
            Ok(())
        } else {
            Err(ValidateError::RegOutOfRange { at, reg: r.0 })
        }
    }

    /// Check the register run `start .. start + count` fits in `n_regs`.
    fn check_reg_run(&self, at: usize, start: Reg, count: u32) -> Result<(), ValidateError> {
        // `end` uses u64 so a large `start + count` cannot wrap.
        let end = start.0 as u64 + count as u64;
        if end <= self.n_regs as u64 {
            Ok(())
        } else {
            // Report the first register past the valid range.
            Err(ValidateError::RegOutOfRange {
                at,
                reg: (end - 1) as u32,
            })
        }
    }

    fn check_cursor(&self, at: usize, c: Cursor) -> Result<(), ValidateError> {
        if c.0 < self.n_cursors {
            Ok(())
        } else {
            Err(ValidateError::CursorOutOfRange { at, cursor: c.0 })
        }
    }

    fn check_addr(&self, at: usize, a: Addr) -> Result<(), ValidateError> {
        if a == Addr::PLACEHOLDER {
            return Err(ValidateError::UnpatchedJump { at });
        }
        if (a.0 as usize) < self.ops.len() {
            Ok(())
        } else {
            Err(ValidateError::JumpOutOfRange {
                at,
                target: a.0 as usize,
            })
        }
    }

    fn check_const(&self, at: usize, id: ConstId) -> Result<(), ValidateError> {
        if (id.0 as usize) < self.consts.len() {
            Ok(())
        } else {
            Err(ValidateError::BadConstRef {
                at,
                id: id.0 as usize,
            })
        }
    }
}

/// Why a [`Program`] failed [`Program::validate`]. Each variant names the
/// offending instruction index (`at`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidateError {
    /// The instruction stream is empty.
    Empty,
    /// The last instruction is not [`Op::Halt`].
    NotHalted,
    /// A jump at `at` still carries [`Addr::PLACEHOLDER`].
    UnpatchedJump {
        /// Index of the offending jump.
        at: usize,
    },
    /// A jump at `at` targets `target`, past the end of the stream.
    JumpOutOfRange {
        /// Index of the offending jump.
        at: usize,
        /// The out-of-range target.
        target: usize,
    },
    /// A register operand at `at` is `>= n_regs`.
    RegOutOfRange {
        /// Index of the offending instruction.
        at: usize,
        /// The out-of-range register.
        reg: u32,
    },
    /// A cursor operand at `at` is `>= n_cursors`.
    CursorOutOfRange {
        /// Index of the offending instruction.
        at: usize,
        /// The out-of-range cursor.
        cursor: u8,
    },
    /// A constant handle at `at` does not resolve inside the pool.
    BadConstRef {
        /// Index of the offending instruction.
        at: usize,
        /// The out-of-range handle.
        id: usize,
    },
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidateError::Empty => write!(f, "empty program"),
            ValidateError::NotHalted => write!(f, "program does not end in Halt"),
            ValidateError::UnpatchedJump { at } => write!(f, "unpatched jump at op {at}"),
            ValidateError::JumpOutOfRange { at, target } => {
                write!(f, "jump at op {at} targets out-of-range address {target}")
            }
            ValidateError::RegOutOfRange { at, reg } => {
                write!(f, "op {at} references out-of-range register r{reg}")
            }
            ValidateError::CursorOutOfRange { at, cursor } => {
                write!(f, "op {at} references out-of-range cursor {cursor}")
            }
            ValidateError::BadConstRef { at, id } => {
                write!(f, "op {at} references out-of-range constant #{id}")
            }
        }
    }
}

impl std::error::Error for ValidateError {}
