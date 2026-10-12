//! The front door: [`Db::execute`] and the [`RowStream`] it hands back.
//!
//! Everything below this point has been built and tested in isolation. This
//! module is the WIRING — it runs a real SQL string through **lexer → parser →
//! binder → planner → compiler → VM** and turns the result into a lazy stream.
//!
//! # A query is a stream, not a `Vec`
//!
//! [`Db::execute`] returns a [`RowStream`]: an `Iterator` that pulls ONE row per
//! `next()`, all the way down to one tuple-store read. `SELECT * FROM docs` over
//! a million rows costs one row of memory unless the caller says otherwise —
//! and `.collect()` is exactly how they say otherwise. Materializing is a
//! decision the caller makes, not a cost the API imposes.
//!
//! # Two levels of error, and why they are different
//!
//! ```text
//!   execute(sql) -> Result<RowStream, Error>     <- COMPILE-TIME failures
//!                          |
//!                          +-> Iterator::Item = Result<OutputRow, Error>
//!                                               <- RUNTIME failures, mid-stream
//! ```
//!
//! The outer `Result` covers everything that happens BEFORE a row can exist:
//! bad syntax, an unknown column, a schema that will not lower, a malformed
//! program. Those are properties of the statement, and they are known before
//! anything runs.
//!
//! The inner `Result` exists because a statement that compiled perfectly can
//! still fail on row 4 of 10 — a read error, a machine fault. By then `execute`
//! has long since returned `Ok`, so there is nowhere else for the failure to
//! go. Collapsing the two levels would mean either buffering every row before
//! returning (killing the stream) or swallowing mid-stream faults as an early
//! end-of-iteration (indistinguishable from success). Both are worse.
//!
//! # Eager mutation, lazy read
//!
//! A **mutation** (`CREATE COLLECTION`, `INSERT`) is drained to completion
//! INSIDE `execute`. It produces no rows, and its side effect must land before
//! the call returns — a lazy mutation would be silently discarded by a caller
//! who dropped the stream, which is not a hazard worth having.
//!
//! A **read** (`SELECT`) opens its cursors inside `execute` and then DETACHES:
//! the returned stream is `'static` and holds no borrow on the `Db`, so it can
//! be stored, returned from a function, or iterated while other statements run.
//! That falls straight out of the `Cursor<'static>` decision — the cursor owns
//! an `Arc` clone of the tuple reader rather than borrowing it.
//!
//! The consequence to know: a read stream sees its cursor's **open-time
//! snapshot**. Rows inserted after `execute` returned are not in it. That is the
//! semantics [`Db::scan`](crate::Db::scan) already guarantees, now visible at
//! the API — and it is what makes slow iteration under concurrent writes safe
//! rather than merely lucky.

use std::fmt;

use crate::compiler::bytecode::{Op, Program, ValidateError};
use crate::compiler::compile;
use crate::engine::Db;
use crate::sql::bind::{BoundStatement, Projected};
use crate::sql::{BindError, ParseError, Statement, analyze, parse, plan};
use crate::vm::exec::{ExecError, OutputRow, Vm};

/// How many times a conflicted write is run again before giving up.
///
/// A copy-on-write UPDATE that loses a race did nothing and is simply re-run
/// against a fresh snapshot, so retrying is cheap and almost always settles on
/// the first one. The cap exists so pathological contention surfaces as
/// `Error::WriteConflict` instead of spinning forever.
const MAX_WRITE_ATTEMPTS: usize = 10;

/// A lazy stream of result rows.
///
/// Owns the suspended [`Vm`] that produces them, and borrows nothing — see the
/// module header. Iterating pulls one row at a time; the stream is FUSED, so
/// once it ends (drained or errored) it keeps returning `None`.
pub struct RowStream {
    /// Column labels, known from the bound projection before any row is read.
    labels: Vec<String>,
    /// The suspended machine. `'static`: no `Db` borrow lives here.
    vm: Vm,
    /// Set on `Halt` or on an error, so iteration cannot restart or re-error.
    done: bool,
}

impl RowStream {
    /// Wrap a primed machine and its labels.
    fn over(labels: Vec<String>, vm: Vm) -> RowStream {
        RowStream {
            labels,
            vm,
            done: false,
        }
    }

    /// A finished stream with no rows — what a mutation returns, having already
    /// run.
    fn empty() -> RowStream {
        RowStream {
            labels: Vec::new(),
            vm: Vm::halted(),
            done: true,
        }
    }

    /// The result columns' labels, in output order.
    ///
    /// Available BEFORE the first row and unchanged by iteration: they come from
    /// the bound projection, which exists as soon as the statement is analyzed.
    /// A caller can print a header, size a table, or decide not to iterate at
    /// all. Empty for a mutation, which produces no columns.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

impl Iterator for RowStream {
    type Item = Result<OutputRow, Error>;

    /// Pull one row. `None` ends the stream; `Some(Err(_))` is a RUNTIME
    /// failure that happened after `execute` already succeeded.
    ///
    /// [`Vm::resume`] rather than `step`, because a read stream is detached: its
    /// cursor is already open and the rest of the scan needs no database.
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.vm.resume() {
            Ok(Some(row)) => Some(Ok(row)),
            Ok(None) => {
                self.done = true;
                None
            }
            // Fuse on error too: a `for row in stream` loop that ignores errors
            // must terminate rather than spin on the same fault.
            Err(e) => {
                self.done = true;
                Some(Err(e.into()))
            }
        }
    }
}

/// Why a statement could not be run.
///
/// The TOP-LEVEL boundary type (CLAUDE.md §5): each stage of the query pipeline
/// keeps its own error, carrying its own concerns — a `ParseError` has a source
/// span, a `BindError` has a column name, [`crate::Error`] has durability
/// failures — and this wraps them rather than flattening every stage into one
/// enum. The variant tells you WHICH stage rejected the statement, which is most
/// of the diagnosis.
///
/// There is no `Lex` variant: the parser wraps lexer errors in its own
/// `ParseError`, so one can never surface here on its own.
#[derive(Debug)]
pub enum Error {
    /// A write kept losing to other writers and ran out of attempts.
    ///
    /// A copy-on-write UPDATE that finds its rows already rewritten is retried
    /// from the top against a fresh snapshot. This is what a caller hears when
    /// that keeps happening — sustained contention on the same rows, which is
    /// a real condition rather than something to hide behind an unbounded
    /// loop. Retrying the statement is a reasonable response.
    WriteConflict {
        /// How many times it was run before giving up.
        attempts: usize,
    },
    /// Bad syntax (lexing included).
    Parse(ParseError),
    /// Names, types, or arity: the statement parsed but does not mean anything.
    Bind(BindError),
    /// The plan could not be lowered — today, a `CREATE` whose schema is not
    /// constructible (`VECTOR(0)`, duplicate columns).
    Compile(crate::compiler::CompileError),
    /// The compiler emitted a structurally malformed program. A compiler bug,
    /// not a user error — surfaced rather than trusted.
    Validate(ValidateError),
    /// Storage refused the operation, or failed while performing it.
    Engine(crate::Error),
    /// The machine could not run the program. Machine faults only — the engine
    /// and validation cases are normalized into their own variants above.
    Exec(ExecError),
    /// The statement is valid V-SQL, but this build does not implement it yet.
    /// An ERROR rather than the `todo!` the opcode arm holds: a public API must
    /// not panic on input a user can type.
    Unsupported {
        /// What is missing.
        feature: &'static str,
    },
}

impl From<ExecError> for Error {
    /// Normalize: the two [`ExecError`] cases that are not machine faults get
    /// their own top-level variants, so a caller matching on `Error::Engine`
    /// catches every storage failure regardless of which layer surfaced it.
    fn from(e: ExecError) -> Error {
        match e {
            ExecError::Invalid(v) => Error::Validate(v),
            ExecError::Engine(e) => Error::Engine(e),
            other => Error::Exec(other),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::WriteConflict { attempts } => write!(
                f,
                "write conflict: the statement lost to concurrent writers {attempts} times"
            ),
            Error::Parse(e) => write!(f, "parse error: {e}"),
            Error::Bind(e) => write!(f, "bind error: {e}"),
            Error::Compile(e) => write!(f, "compile error: {e}"),
            Error::Validate(e) => write!(f, "malformed program: {e}"),
            Error::Engine(e) => write!(f, "engine error: {e}"),
            Error::Exec(e) => write!(f, "execution error: {e}"),
            Error::Unsupported { feature } => write!(f, "not supported yet: {feature}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parse(e) => Some(e),
            Error::Bind(e) => Some(e),
            Error::Compile(e) => Some(e),
            Error::Validate(e) => Some(e),
            Error::Engine(e) => Some(e),
            Error::Exec(e) => Some(e),
            Error::Unsupported { .. } | Error::WriteConflict { .. } => None,
        }
    }
}

impl Db {
    /// Run one V-SQL statement.
    ///
    /// The whole pipeline: lex → parse → bind → plan → compile → validate →
    /// execute. Returns a lazy [`RowStream`]; see the module header for the
    /// two-level error structure and the eager-mutation / lazy-read split.
    ///
    /// Takes `&self`, not `&mut self`: every other `Db` method does, because the
    /// engine is single-writer / MULTI-READER by design (`create_collection`
    /// serializes DDL through its own mutex). Requiring `&mut` here would
    /// serialize every statement through one caller and make parallel `SELECT`s
    /// impossible — the opposite of what the read path was built for.
    pub fn execute(&self, sql: &str) -> Result<RowStream, Error> {
        // 1. PARSE (lexing included — the parser wraps lex errors).
        let statement = parse(sql).map_err(Error::Parse)?;
        // The compiler needs the TARGET collection's storage schema, and only
        // the statement says which collection that is — read it before `analyze`
        // consumes the statement.
        let target = Target::of(&statement);

        // 2. BIND — the one stage that touches the catalog. `Db` implements the
        //    binder's `Catalog`, so names resolve against real collections.
        let bound = analyze(statement, self).map_err(Error::Bind)?;
        // Labels come from the BOUND projection, so they exist long before the
        // first row is fetched.
        let labels = labels_of(&bound);
        // A DELETE would also finish inside `open_cursors` — its `Mutate` needs
        // the database, so it runs before the machine detaches — but that is
        // the read path's detach rule doing a write by coincidence. Mutations
        // are drained on the write path, where "durable before `execute`
        // returns" is the stated contract rather than a consequence.
        let mutates = matches!(
            bound,
            BoundStatement::Insert(_)
                | BoundStatement::CreateCollection(_)
                | BoundStatement::Delete(_)
                | BoundStatement::Update(_)
        );

        // 3. PLAN — infallible: binding already validated everything, so this
        //    only changes shape.
        let logical = plan(bound);

        // 4. COMPILE — against THE CATALOG'S schema for the target collection.
        //    The catalog is the single source of truth: the same stored
        //    `metadata::Schema` the binder resolved names against (via
        //    `bind_schema`) is what translates this statement's declaration
        //    ordinals into storage `ColumnId`s. Nothing is reconstructed.
        let schema = match &target {
            Target::Existing(name) => Some(
                self.collections()
                    .into_iter()
                    .find(|c| &c.name == name)
                    .map(|c| c.schema)
                    // Binding just resolved this name off the same catalog, so a
                    // miss is a race, not user error.
                    .ok_or_else(|| {
                        Error::Engine(crate::Error::UnknownCollectionName { name: name.clone() })
                    })?,
            ),
            // A `CREATE` has no catalog entry to read — the collection is what
            // it is about to make — so its storage schema can only come from its
            // own DDL, which `compile` lowers itself. `None` says that in the
            // type rather than making this pass a schema nobody reads.
            Target::New => None,
        };
        let program = compile(logical, schema.as_ref()).map_err(Error::Compile)?;

        // 5. Refuse what this build cannot run, so the unbuilt opcode arms are
        //    unreachable from public input.
        reject_unsupported(&program)?;

        // 6. BUILD — `Vm::new` runs `Program::validate`. The program is kept
        //    so a conflicted write can be run again from the top.
        let mut vm = Vm::new(program.clone())?;

        if mutates {
            // EAGER: drain inside `execute`. A mutation yields no rows, and its
            // side effect must be durable before this returns.
            //
            // A copy-on-write UPDATE can come back conflicted: the rows it read
            // were rewritten or retired by another writer between the read and
            // the commit, so it did nothing. Retrying means running the WHOLE
            // program again — the predicate must be resolved against a fresh
            // snapshot, since which rows match may have changed — so a fresh
            // `Vm` runs the same compiled program. Callers never see the
            // conflict, only the applied result.
            for attempt in 1..=MAX_WRITE_ATTEMPTS {
                match (|| -> Result<(), Error> {
                    while vm.step(self)?.is_some() {}
                    Ok(())
                })() {
                    Ok(()) => return Ok(RowStream::empty()),
                    Err(e) if is_retryable(&e) && attempt < MAX_WRITE_ATTEMPTS => {
                        vm = Vm::new(program.clone())?;
                    }
                    // Out of attempts. Sustained contention on one row is a
                    // real condition a caller has to hear about rather than
                    // have hidden behind an unbounded loop.
                    Err(e) if is_retryable(&e) => {
                        return Err(Error::WriteConflict {
                            attempts: MAX_WRITE_ATTEMPTS,
                        });
                    }
                    // Everything else — including a REFUSED record, which is
                    // final — travels as itself.
                    Err(e) => return Err(e),
                }
            }
            unreachable!("the loop returns on its last attempt")
        } else {
            // LAZY: open the cursors, then detach. Everything after this pulls
            // without a `Db`.
            vm.open_cursors(self)?;
            Ok(RowStream::over(labels, vm))
        }
    }
}

/// Which failures earn another attempt.
///
/// Exactly one does. A CONFLICT means some other caller got to these rows
/// first, so re-resolving the predicate against a fresh snapshot is likely to
/// succeed. Every other failure — a refused record above all — would reach the
/// identical outcome on every attempt, so retrying it only delays the error the
/// caller is owed.
fn is_retryable(e: &Error) -> bool {
    matches!(e, Error::Exec(ExecError::Conflict))
}

/// Which collection a statement compiles against — and therefore where its
/// storage schema comes from.
enum Target {
    /// An existing collection, by name: its schema is a catalog lookup.
    Existing(String),
    /// A `CREATE`, which is about to make its collection. Nothing to look up.
    New,
}

impl Target {
    fn of(statement: &Statement) -> Target {
        match statement {
            Statement::Select(s) => Target::Existing(s.from.clone()),
            Statement::Insert(i) => Target::Existing(i.collection.clone()),
            Statement::CreateCollection(_) => Target::New,
            // A SEARCH reads an existing collection, like a SELECT. (The binder
            // rejects it for now — see `BindError::Unbound` — but this is the
            // right answer, so the arm is real rather than a placeholder.)
            Statement::Search(s) => Target::Existing(s.collection.clone()),
            // A DELETE's predicate is resolved against the collection it names.
            Statement::Delete(d) => Target::Existing(d.from.clone()),
            // An UPDATE reads and rewrites rows of the collection it names.
            Statement::Update(u) => Target::Existing(u.collection.clone()),
        }
    }
}

/// The result columns' labels. Only a `SELECT` has any; a mutation produces no
/// columns.
///
/// Taken from the bound projection, which the binder has already expanded (`*`
/// becomes every scalar, the embedding excluded) — so the labels and the emitted
/// `Column` ops come from the same list and cannot disagree in arity.
fn labels_of(bound: &BoundStatement) -> Vec<String> {
    match bound {
        BoundStatement::Select(select) => {
            select.projection.iter().map(|c| c.name.clone()).collect()
        }
        // A SEARCH's labels are its RETURNING names in order — or, with no
        // clause, the default projection's, like a `SELECT *`.
        BoundStatement::Search(search) => search
            .projection
            .iter()
            .map(|item| match item {
                // The SCHEMA's spelling, not the source's — the same rule a
                // SELECT column follows.
                Projected::Column(col) => col.name.clone(),
                // Canonical lowercase, for the same reason: a label is what the
                // column IS, not how this particular query happened to spell it.
                Projected::Pseudo(pseudo) => pseudo.name().to_string(),
            })
            .collect(),
        BoundStatement::Insert(_)
        | BoundStatement::CreateCollection(_)
        | BoundStatement::Delete(_)
        | BoundStatement::Update(_) => Vec::new(),
    }
}

/// Reject a program containing an opcode this build does not implement.
///
/// The alternative is reaching the arm's `todo!` — a panic in library code, on
/// input a user can type. The check is here rather than in the VM because only
/// the boundary knows to phrase it as a feature, not a machine fault.
fn reject_unsupported(program: &Program) -> Result<(), Error> {
    if program
        .ops
        .iter()
        .any(|op| matches!(op, Op::VectorFetch { .. }))
    {
        return Err(Error::Unsupported {
            feature: "selecting a vector column (the embedding is stored separately)",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Error, RowStream, is_retryable};
    use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
    use crate::compiler::constants::ConstPool;
    use crate::engine::{Db, DbOptions};
    use crate::error::Error as EngineError;
    use crate::sql::BindError;
    use crate::vm::exec::{RegValue, Vm};

    /// Only a CONFLICT earns another attempt. A refusal is final — the record
    /// was durable, the applier looked at it and said never, and re-running the
    /// statement would produce the identical verdict — so it must leave the
    /// retry loop immediately and reach the caller as itself.
    #[test]
    fn only_a_conflict_is_retried() {
        use crate::vm::exec::ExecError;
        use crate::wal::wal::RejectReason;

        assert!(is_retryable(&Error::Exec(ExecError::Conflict)));
        assert!(!is_retryable(&Error::Exec(ExecError::Rejected(
            RejectReason::DuplicateKey
        ))));
        assert!(!is_retryable(&Error::WriteConflict { attempts: 10 }));
    }

    const CREATE: &str = "CREATE COLLECTION docs (
        author TEXT PRIMARY KEY,
        vector VECTOR(4),
        title TEXT,
        published_at INT
    ) WITH (capacity = 1024);";

    /// A database with `docs` created THROUGH `execute` — so even the fixture
    /// exercises the front door.
    fn docs_db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Db::open(dir.path(), &[], DbOptions::default()).expect("open");
        db.execute(CREATE).expect("the collection is created");
        (dir, db)
    }

    fn insert(db: &Db, author: &str, title: &str, n: i64) {
        let sql = format!(
            "INSERT INTO docs (author, vector, title, published_at) \
             VALUES ('{author}', [0.1, 0.2, 0.3, 0.4], '{title}', {n});"
        );
        db.execute(&sql).expect("the row is inserted");
    }

    /// Drain a stream into rows, failing on the first `Err` item.
    fn collect(stream: RowStream) -> Vec<Vec<RegValue>> {
        stream
            .map(|row| row.expect("no runtime error").0)
            .collect::<Vec<_>>()
    }

    // -- 1. seam (c): ResultRow emits REGISTERS, not columns ----------------

    #[test]
    fn result_row_emits_registers() {
        let (_dir, db) = docs_db();
        insert(&db, "alice", "My doc", 7);

        // One STORED column and one COMPUTED value in the same output row. The
        // computed register has no column behind it at all, so a `ResultRow`
        // that read off the cursor instead of the register file could not
        // produce this row. That is seam (c), and it is what makes
        // `RETURNING id, score` free later: `score` will be exactly this — a
        // register a KNN op parked, sitting beside a `Column`.
        let mut vm = Vm::new(Program {
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
                // Computed: no cursor, no column, no storage.
                Op::Real {
                    value: 0.75,
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
        })
        .expect("well-formed");
        vm.open_cursors(&db).expect("the scan cursor opens");

        let stream = RowStream::over(vec!["author".into(), "score".into()], vm);
        assert_eq!(
            collect(stream),
            vec![vec![RegValue::Str("alice".into()), RegValue::Real(0.75)]]
        );

        db.close().unwrap();
    }

    // -- 2. labels land before rows ----------------------------------------

    #[test]
    fn labels_available_before_first_row() {
        let (_dir, db) = docs_db();
        insert(&db, "alice", "My doc", 7);

        let mut stream = db
            .execute("SELECT title, published_at FROM docs;")
            .expect("compiles");

        // BEFORE consuming anything — the labels come from the BOUND
        // projection, which exists long before the first row is fetched. A
        // caller can print a header, size a table, or decide not to iterate at
        // all.
        assert_eq!(stream.labels(), ["title", "published_at"]);

        // ...and they survive iteration.
        let first = stream.next().expect("a row").expect("no runtime error");
        assert_eq!(
            first.0,
            vec![RegValue::Str("My doc".into()), RegValue::Int(7)]
        );
        assert_eq!(stream.labels(), ["title", "published_at"]);

        // `SELECT *` labels every scalar, and NOT the embedding.
        let star = db.execute("SELECT * FROM docs;").expect("compiles");
        assert_eq!(star.labels(), ["author", "title", "published_at"]);

        // A mutation produces no columns, so it has no labels.
        let mutation = db
            .execute(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('bob', [1.0, 0.0, 0.0, 0.0], 'other', 9);",
            )
            .expect("compiles");
        assert!(mutation.labels().is_empty());

        db.close().unwrap();
    }

    // -- 3. OUTER errors: every stage wrapped in its own variant ------------

    #[test]
    fn error_boundary_wraps_each_layer() {
        let (_dir, db) = docs_db();

        // PARSE — the lexer's errors arrive inside `ParseError` (the parser
        // wraps them), so there is no separate `Lex` variant to reach.
        assert!(matches!(
            db.execute("SELECT author docs;"),
            Err(Error::Parse(_))
        ));
        assert!(matches!(db.execute("NONSENSE;"), Err(Error::Parse(_))));

        // BIND — including UNKNOWN COLLECTION, which never reaches the engine:
        // the binder resolves names against the catalog, so it rejects them
        // first. `Error::Engine` is for failures storage raises while running.
        assert!(matches!(
            db.execute("SELECT author FROM ghosts;"),
            Err(Error::Bind(BindError::CollectionNotFound(name))) if name == "ghosts"
        ));
        assert!(matches!(
            db.execute("SELECT publisher FROM docs;"),
            Err(Error::Bind(BindError::ColumnNotFound(name))) if name == "publisher"
        ));

        // COMPILE — binds cleanly, cannot lower. The binder counts vector
        // columns but never checks the dimension.
        assert!(matches!(
            db.execute("CREATE COLLECTION broken (v VECTOR(0), a INT PRIMARY KEY) WITH (capacity = 10);"),
            Err(Error::Compile(_))
        ));

        // BIND also catches "collection already exists" — same reason. Worth
        // pinning: it is easy to assume a duplicate CREATE is the engine's
        // refusal, and it is not.
        assert!(matches!(
            db.execute(CREATE),
            Err(Error::Bind(BindError::CollectionExists(name))) if name == "docs"
        ));

        // ENGINE — reachable only for rules the BINDER does not know, because
        // it pre-empts everything it does know. One exists:
        //
        //   * capacity is a RUNTIME property — how many rows are already there —
        //     so only the engine can know the collection is full. This one also
        //     shows why the eager/lazy split matters: the INSERT ran INSIDE
        //     `execute`, so its failure is an outer error, not a stream item.
        db.execute("CREATE COLLECTION tiny (v VECTOR(2), a INT PRIMARY KEY) WITH (capacity = 1);")
            .expect("creates");
        let fill = "INSERT INTO tiny (v, a) VALUES ([1.0, 0.0], 1);";
        db.execute(fill).expect("the first row fits");
        assert!(matches!(
            db.execute(fill),
            Err(Error::Engine(EngineError::CapacityExceeded { .. }))
        ));

        // UNSUPPORTED — the one built-out gap. It must be an ERROR, not the
        // `todo!` panic the arm holds: a public API may not panic on input a
        // user can type.
        assert!(matches!(
            db.execute("SELECT vector FROM docs;"),
            Err(Error::Unsupported { .. })
        ));

        db.close().unwrap();
    }

    // -- 4. INNER errors: a failure PARTWAY THROUGH a stream ----------------

    #[test]
    fn error_mid_stream_is_an_err_item() {
        let (_dir, db) = docs_db();

        // Two good rows, then an op that fails. `Column` on a cursor slot that
        // was allocated but never opened passes `validate` (which checks the
        // slot is in RANGE, not that it is open) and fails at run time — which
        // is precisely the shape being pinned: the program COMPILED, and the
        // failure happens after rows have already been handed out.
        let vm = Vm::new(Program {
            ops: vec![
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
                Op::Column {
                    cur: Cursor(0),
                    col: 0,
                    dst: Reg(0),
                },
                Op::ResultRow {
                    start: Reg(0),
                    count: 1,
                },
                Op::Halt,
            ],
            consts: ConstPool::new(),
            n_regs: 1,
            n_cursors: 1,
        })
        .expect("well-formed");
        let mut stream = RowStream::over(vec!["n".into()], vm);

        // Rows first...
        assert_eq!(
            stream.next().expect("a row").expect("no error yet").0,
            vec![RegValue::Int(1)]
        );
        assert_eq!(
            stream.next().expect("a row").expect("no error yet").0,
            vec![RegValue::Int(2)]
        );
        // ...then the failure, as an ITEM. This is the whole two-level
        // structure: `execute` already returned `Ok`, so a mid-stream fault has
        // nowhere else to go.
        let item = stream.next().expect("a third item");
        assert!(
            matches!(item, Err(Error::Exec(_))),
            "expected a runtime error item, got {item:?}"
        );

        // And iteration stops cleanly rather than re-erroring or looping: a
        // `for row in stream` that ignores errors must terminate.
        assert!(
            stream.next().is_none(),
            "the stream is fused after an error"
        );
        assert!(stream.next().is_none());

        db.close().unwrap();
    }

    // -- 5. first light ----------------------------------------------------

    #[test]
    fn execute_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();

        // Real SQL strings, all the way down: lexer → parser → binder → planner
        // → compiler → VM → the three stores.
        db.execute(CREATE).expect("CREATE runs");

        for (author, title, n) in [
            ("alice", "first", 1),
            ("bob", "second", 2),
            ("carol", "third", 3),
        ] {
            insert(&db, author, title, n);
        }

        let stream = db
            .execute("SELECT author, published_at FROM docs;")
            .expect("SELECT compiles");
        assert_eq!(stream.labels(), ["author", "published_at"]);

        // `.collect()` materializes because the CALLER asked it to — the stream
        // itself pulled one row at a time to get here.
        assert_eq!(
            collect(stream),
            vec![
                vec![RegValue::Str("alice".into()), RegValue::Int(1)],
                vec![RegValue::Str("bob".into()), RegValue::Int(2)],
                vec![RegValue::Str("carol".into()), RegValue::Int(3)],
            ]
        );

        // The whole projection, in schema order, through `*`.
        let star = db.execute("SELECT * FROM docs;").expect("compiles");
        assert_eq!(star.labels(), ["author", "title", "published_at"]);
        assert_eq!(collect(star).len(), 3);

        db.close().unwrap();
    }

    // -- SEARCH: the first vector query -------------------------------------

    /// Seed rows whose embeddings have a KNOWN dot-product order against the
    /// query `[1, 0, 0, 0]`, and whose INSERT order is deliberately different
    /// from their score order — so a result that came back in ordinal order
    /// cannot be mistaken for a result that came back ranked.
    ///
    ///   title  embedding      dot([1,0,0,0])   ordinal   rank
    ///   far    [1, 0, 0, 0]        1.0            0        3rd
    ///   near   [3, 0, 0, 0]        3.0            1        1st
    ///   mid    [2, 0, 0, 0]        2.0            2        2nd
    fn seed_ranked(db: &Db) {
        for (title, x, n) in [("far", 1.0, 1), ("near", 3.0, 2), ("mid", 2.0, 3)] {
            db.execute(&format!(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('alice', [{x}, 0.0, 0.0, 0.0], '{title}', {n});"
            ))
            .expect("seeded row inserts");
        }
    }

    /// The `title` column of each returned row, in the order the stream gave
    /// them — the whole point of a ranked query.
    fn titles(stream: RowStream) -> Vec<String> {
        collect(stream)
            .into_iter()
            .map(|row| match &row[1] {
                RegValue::Str(s) => s.clone(),
                other => panic!("expected a title string, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn bare_search_returns_nearest_in_score_order() {
        let (_dir, db) = docs_db();
        seed_ranked(&db);

        let stream = db
            .execute("SEARCH TOP 3 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
            .expect("compiles and runs");
        // Bare SEARCH projects every scalar, like `SELECT *`.
        assert_eq!(stream.labels(), ["author", "title", "published_at"]);

        // NEAREST FIRST. Ordinal order would be far, near, mid — so this
        // assertion fails outright if the ranking is dropped anywhere between
        // `search()` and the cursor.
        assert_eq!(titles(stream), ["near", "mid", "far"]);

        // The scalar columns come back whole, not just the ranked ordinals.
        let stream = db
            .execute("SEARCH TOP 1 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
            .expect("compiles and runs");
        assert_eq!(
            collect(stream),
            vec![vec![
                RegValue::Str("alice".into()),
                RegValue::Str("near".into()),
                RegValue::Int(2),
            ]]
        );

        db.close().unwrap();
    }

    #[test]
    fn search_top_k_limits_the_result() {
        let (_dir, db) = docs_db();
        // Five rows, so `k` is neither the row count nor a coincidence.
        for n in 1..=5 {
            db.execute(&format!(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('alice', [{n}.0, 0.0, 0.0, 0.0], 'doc {n}', {n});"
            ))
            .expect("inserts");
        }

        for k in [1, 3, 5] {
            let stream = db
                .execute(&format!(
                    "SEARCH TOP {k} NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;"
                ))
                .expect("compiles and runs");
            assert_eq!(collect(stream).len(), k, "TOP {k}");
        }
        // Asking for more than exist yields what exists, not an error.
        let stream = db
            .execute("SEARCH TOP 50 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
            .expect("compiles and runs");
        assert_eq!(collect(stream).len(), 5);

        db.close().unwrap();
    }

    #[test]
    fn an_enormous_top_k_does_not_abort_the_process() {
        let (_dir, db) = docs_db();
        seed_ranked(&db);

        // REGRESSION: `TOP i64::MAX` reached `BinaryHeap::with_capacity(k)` in
        // the flat index and aborted the process with a capacity overflow. The
        // binder only floors `k` at 1 and `usize::try_from` succeeds on 64-bit,
        // so nothing upstream bounded it — and a panic in library code on input
        // a user can TYPE is exactly what the boundary exists to prevent
        // (CLAUDE.md §5). The heap is now reserved for `min(k, committed)`.
        //
        // The assertion is simply that this RETURNS: reaching the next line at
        // all is the property under test. Every row is returned, because asking
        // for more rows than exist yields what exists.
        let stream = db
            .execute(&format!(
                "SEARCH TOP {} NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;",
                i64::MAX
            ))
            .expect("an absurd k compiles and runs rather than aborting");
        assert_eq!(collect(stream).len(), 3, "every seeded row comes back");

        // The same operand against an EMPTY collection: `committed` is 0 there,
        // so the clamp is doing all the work and a stray `k`-sized reservation
        // would still abort.
        db.execute(
            "CREATE COLLECTION empty_docs (vector VECTOR(4), author TEXT PRIMARY KEY) \
             WITH (capacity = 16);",
        )
        .expect("the collection is created");
        let stream = db
            .execute(&format!(
                "SEARCH TOP {} NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM empty_docs;",
                i64::MAX
            ))
            .expect("an absurd k over an empty collection runs too");
        assert_eq!(collect(stream).len(), 0);

        db.close().unwrap();
    }

    #[test]
    fn search_dimension_mismatch_is_a_bind_error() {
        let (_dir, db) = docs_db();
        seed_ranked(&db);

        // `docs.vector` is VECTOR(4). A wrong-length query is caught by the
        // BINDER, before anything is compiled or run — the same rule and the
        // same error an INSERT's embedding gets.
        assert!(matches!(
            db.execute("SEARCH TOP 3 NEAREST TO [1.0, 0.0] FROM docs;"),
            Err(Error::Bind(BindError::DimensionMismatch {
                expected: 4,
                found: 2,
                ..
            }))
        ));
        assert!(matches!(
            db.execute("SEARCH TOP 3 NEAREST TO [1.0, 0.0, 0.0, 0.0, 0.0] FROM docs;"),
            Err(Error::Bind(BindError::DimensionMismatch {
                expected: 4,
                found: 5,
                ..
            }))
        ));

        db.close().unwrap();
    }

    #[test]
    fn search_unknown_collection_is_a_bind_error() {
        let (_dir, db) = docs_db();
        // Same resolution path, same error a SELECT gets.
        assert!(matches!(
            db.execute("SEARCH TOP 3 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM ghosts;"),
            Err(Error::Bind(BindError::CollectionNotFound(name))) if name == "ghosts"
        ));
        db.close().unwrap();
    }

    #[test]
    fn search_top_zero_is_a_bind_error() {
        let (_dir, db) = docs_db();
        seed_ranked(&db);

        // `TOP 0` PARSES (pinned in the parser tests) — "k >= 1" is semantics,
        // and this is where semantics live. It must not reach `Db::search`,
        // which has its own opinion about k, nor quietly become an empty result.
        assert!(matches!(
            db.execute("SEARCH TOP 0 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;"),
            Err(Error::Bind(BindError::InvalidTopK { k: 0 }))
        ));

        db.close().unwrap();
    }

    #[test]
    fn bare_search_end_to_end() {
        // THE HEADLINE: a vector query, from a SQL string to ranked rows.
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();

        db.execute(CREATE).expect("CREATE runs");
        seed_ranked(&db);

        let stream = db
            .execute("SEARCH TOP 5 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
            .expect("SEARCH compiles");
        assert_eq!(stream.labels(), ["author", "title", "published_at"]);
        assert_eq!(
            collect(stream),
            vec![
                vec![
                    RegValue::Str("alice".into()),
                    RegValue::Str("near".into()),
                    RegValue::Int(2),
                ],
                vec![
                    RegValue::Str("alice".into()),
                    RegValue::Str("mid".into()),
                    RegValue::Int(3),
                ],
                vec![
                    RegValue::Str("alice".into()),
                    RegValue::Str("far".into()),
                    RegValue::Int(1),
                ],
            ],
            "nearest first, with every scalar column"
        );

        db.close().unwrap();
    }

    // -- RETURNING: the register seam, cashed in ----------------------------

    /// Seed `n` rows where row `i` has embedding `[(i+1), 0, 0, 0]` and title
    /// `doc i`. Against the query `[1, 0, 0, 0]` the dot product is exactly
    /// `i + 1`, so:
    ///
    ///   * scores are DISTINCT and strictly ordered — an off-by-one between a
    ///     row's id and its score is observable, not masked by a tie;
    ///   * `score == id + 1` is an invariant every returned row must satisfy,
    ///     which is a per-row check rather than a whole-list comparison;
    ///   * insert order is the REVERSE of score order, so a result that came
    ///     back unranked cannot pass.
    fn seed_scored(db: &Db, n: i64) {
        for i in 0..n {
            db.execute(&format!(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('a{i}', [{}.0, 0.0, 0.0, 0.0], 'doc {i}', {i});",
                i + 1
            ))
            .expect("seeded row inserts");
        }
    }

    const QUERY: &str = "[1.0, 0.0, 0.0, 0.0]";

    #[test]
    fn returning_a_column_and_score() {
        let (_dir, db) = docs_db();
        seed_scored(&db, 4);

        let stream = db
            .execute(&format!(
                "SEARCH TOP 4 NEAREST TO {QUERY} FROM docs RETURNING published_at, score;"
            ))
            .expect("compiles and runs");
        // Labels are the RETURNING names, in order — NOT the collection's
        // columns, which is what a bare SEARCH would have reported.
        assert_eq!(stream.labels(), ["published_at", "score"]);

        let rows = collect(stream);
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows,
            vec![
                vec![RegValue::Int(3), RegValue::Real(4.0)],
                vec![RegValue::Int(2), RegValue::Real(3.0)],
                vec![RegValue::Int(1), RegValue::Real(2.0)],
                vec![RegValue::Int(0), RegValue::Real(1.0)],
            ],
            "nearest first, each row paired with its own score"
        );

        db.close().unwrap();
    }

    #[test]
    fn score_aligns_with_its_row() {
        // THE load-bearing test. `score == published_at + 1` holds for every
        // seeded row by construction, so ANY drift between a row and the score
        // reported beside it breaks the invariant — including an index advanced
        // one step out of step with the cursor, which is invisible to an
        // assertion that only checks "scores descend".
        let (_dir, db) = docs_db();
        seed_scored(&db, 6);

        for k in [1, 3, 6] {
            let stream = db
                .execute(&format!(
                    "SEARCH TOP {k} NEAREST TO {QUERY} FROM docs RETURNING published_at, score;"
                ))
                .expect("compiles and runs");
            let rows = collect(stream);
            assert_eq!(rows.len(), k, "TOP {k}");

            let mut previous: Option<f64> = None;
            for row in rows {
                let (n, score) = match (&row[0], &row[1]) {
                    (RegValue::Int(n), RegValue::Real(score)) => (*n, *score),
                    other => panic!("expected (Int, Real), got {other:?}"),
                };
                assert_eq!(
                    score,
                    (n + 1) as f64,
                    "row {n} was reported with score {score}, but its own \
                     similarity is {}",
                    n + 1
                );
                // ...and the ranking still descends, so the invariant above is
                // not being satisfied by an accidentally-sorted-by-id result.
                if let Some(previous) = previous {
                    assert!(score < previous, "scores must descend");
                }
                previous = Some(score);
            }
        }

        db.close().unwrap();
    }

    #[test]
    fn returning_mixes_stored_and_computed() {
        // SEAM (c), exercised rather than asserted: one value read from storage
        // (`title`, via `Column`) sitting beside one the query computed
        // (`score`). Both are just registers by the time `ResultRow` runs, which
        // is exactly why this costs nothing. A design that special-cased
        // pseudo-columns — emitting them through a separate path instead of into
        // the register file — would fail here even if `RETURNING title, score`
        // worked.
        let (_dir, db) = docs_db();
        seed_scored(&db, 3);

        let stream = db
            .execute(&format!(
                "SEARCH TOP 3 NEAREST TO {QUERY} FROM docs RETURNING title, score;"
            ))
            .expect("compiles and runs");
        assert_eq!(stream.labels(), ["title", "score"]);
        assert_eq!(
            collect(stream),
            vec![
                vec![RegValue::Str("doc 2".into()), RegValue::Real(3.0)],
                vec![RegValue::Str("doc 1".into()), RegValue::Real(2.0)],
                vec![RegValue::Str("doc 0".into()), RegValue::Real(1.0)],
            ]
        );

        // Interleaved the other way, and with all three kinds at once.
        let stream = db
            .execute(&format!(
                "SEARCH TOP 2 NEAREST TO {QUERY} FROM docs RETURNING score, published_at, title, author;"
            ))
            .expect("compiles and runs");
        assert_eq!(
            stream.labels(),
            ["score", "published_at", "title", "author"]
        );
        assert_eq!(
            collect(stream),
            vec![
                vec![
                    RegValue::Real(3.0),
                    RegValue::Int(2),
                    RegValue::Str("doc 2".into()),
                    RegValue::Str("a2".into()),
                ],
                vec![
                    RegValue::Real(2.0),
                    RegValue::Int(1),
                    RegValue::Str("doc 1".into()),
                    RegValue::Str("a1".into()),
                ],
            ]
        );

        db.close().unwrap();
    }

    #[test]
    fn returning_labels_match_the_clause() {
        let (_dir, db) = docs_db();
        seed_scored(&db, 2);

        for (clause, expected) in [
            ("published_at", vec!["published_at"]),
            ("score", vec!["score"]),
            ("published_at, score", vec!["published_at", "score"]),
            ("score, published_at", vec!["score", "published_at"]),
            ("title", vec!["title"]),
            (
                "title, published_at, score, author",
                vec!["title", "published_at", "score", "author"],
            ),
        ] {
            let stream = db
                .execute(&format!(
                    "SEARCH TOP 1 NEAREST TO {QUERY} FROM docs RETURNING {clause};"
                ))
                .expect("compiles and runs");
            assert_eq!(stream.labels(), expected.as_slice(), "RETURNING {clause}");
            // Arity of the rows matches the labels, so a header can never
            // promise more columns than the stream delivers.
            assert_eq!(collect(stream)[0].len(), expected.len());
        }

        // No clause -> the bare-SEARCH default is untouched.
        let stream = db
            .execute(&format!("SEARCH TOP 1 NEAREST TO {QUERY} FROM docs;"))
            .expect("compiles and runs");
        assert_eq!(stream.labels(), ["author", "title", "published_at"]);

        db.close().unwrap();
    }

    #[test]
    fn returning_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();

        db.execute(CREATE).expect("CREATE runs");
        seed_scored(&db, 5);

        let stream = db
            .execute(&format!(
                "SEARCH TOP 5 NEAREST TO {QUERY} FROM docs RETURNING published_at, score;"
            ))
            .expect("SEARCH compiles");
        assert_eq!(stream.labels(), ["published_at", "score"]);
        assert_eq!(
            collect(stream),
            vec![
                vec![RegValue::Int(4), RegValue::Real(5.0)],
                vec![RegValue::Int(3), RegValue::Real(4.0)],
                vec![RegValue::Int(2), RegValue::Real(3.0)],
                vec![RegValue::Int(1), RegValue::Real(2.0)],
                vec![RegValue::Int(0), RegValue::Real(1.0)],
            ]
        );

        db.close().unwrap();
    }

    // -- the catalog is the schema's single source of truth -----------------

    #[test]
    fn select_uses_catalog_schema_not_a_reconstruction() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();

        // Two collections with the SAME column names in DIFFERENT positions, so
        // the same name resolves to a different declaration ordinal AND a
        // different storage ColumnId in each:
        //
        //   alpha: emb@0        title@1 #0  tag@2 #1
        //   beta:  tag@0 #0     emb@1       title@2 #1
        //
        // `title` is ordinal 1 / ColumnId 0 in alpha, ordinal 2 / ColumnId 1 in
        // beta. Compiling either statement against the wrong collection's schema
        // therefore reads the wrong column — or, for alpha's ordinal 1 under
        // beta's schema, resolves to the VECTOR and emits nothing at all.
        db.execute(
            "CREATE COLLECTION alpha (emb VECTOR(2), title TEXT PRIMARY KEY, tag TEXT) WITH (capacity = 8);",
        )
        .expect("alpha is created");
        db.execute(
            "CREATE COLLECTION beta (tag TEXT PRIMARY KEY, emb VECTOR(2), title TEXT) WITH (capacity = 8);",
        )
        .expect("beta is created");

        db.execute("INSERT INTO alpha (emb, title, tag) VALUES ([1.0, 0.0], 'A-title', 'A-tag');")
            .expect("alpha row inserts");
        db.execute("INSERT INTO beta (tag, emb, title) VALUES ('B-tag', [0.0, 1.0], 'B-title');")
            .expect("beta row inserts");

        // Each statement must use ITS OWN collection's catalog schema.
        let alpha = db
            .execute("SELECT title, tag FROM alpha;")
            .expect("compiles");
        assert_eq!(alpha.labels(), ["title", "tag"]);
        assert_eq!(
            collect(alpha),
            vec![vec![
                RegValue::Str("A-title".into()),
                RegValue::Str("A-tag".into())
            ]]
        );

        let beta = db
            .execute("SELECT title, tag FROM beta;")
            .expect("compiles");
        assert_eq!(beta.labels(), ["title", "tag"]);
        assert_eq!(
            collect(beta),
            vec![vec![
                RegValue::Str("B-title".into()),
                RegValue::Str("B-tag".into())
            ]]
        );

        // HONEST LIMIT: this proves the schema used is the TARGET COLLECTION'S,
        // as held by the catalog. It cannot distinguish the catalog's schema
        // from an equal-but-separately-rebuilt one, because no such divergence
        // is constructible today: the bind schema is itself derived from the
        // catalog schema, and that round trip is lossless (pinned by
        // `compiler::schema::roundtrip_bind_meta_bind`). Schemas are also
        // immutable — there is no ALTER — so a reconstruction cannot go stale.
        // The first feature that could break the tie (schema evolution, or a
        // plan node that synthesizes columns) is what would give this teeth on
        // that finer point.

        db.close().unwrap();
    }

    // -- the eager/lazy split, and stream hygiene ---------------------------

    #[test]
    fn mutations_land_before_execute_returns() {
        let (_dir, db) = docs_db();

        // The signed-off split: a mutation is drained INSIDE `execute`. Its
        // effect is visible the moment the call returns, with nothing stepped —
        // if it were lazy, a caller that dropped the stream would silently
        // discard the write.
        let stream = db
            .execute(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('alice', [0.1, 0.2, 0.3, 0.4], 'My doc', 7);",
            )
            .expect("compiles");
        drop(stream);

        let rows = collect(db.execute("SELECT author FROM docs;").expect("compiles"));
        assert_eq!(rows, vec![vec![RegValue::Str("alice".into())]]);

        db.close().unwrap();
    }

    #[test]
    fn a_read_stream_holds_no_borrow_on_the_db() {
        let (_dir, db) = docs_db();
        insert(&db, "alice", "My doc", 7);

        // The stream is `'static`: it can outlive the call that made it, be
        // returned from a function, and be iterated while OTHER statements run
        // against the same `Db`. That is what the `'static` cursor bought.
        fn make(db: &Db) -> RowStream {
            db.execute("SELECT author FROM docs;").expect("compiles")
        }
        let mut stream = make(&db);

        // A concurrent write mid-iteration. The stream reads its cursor's
        // OPEN-TIME snapshot, so this row is not in it — the same semantics
        // `Db::scan` already guarantees, now visible at the API.
        insert(&db, "bob", "later", 9);

        assert_eq!(
            stream.next().expect("a row").expect("no error").0,
            vec![RegValue::Str("alice".into())]
        );
        assert!(
            stream.next().is_none(),
            "a row inserted after the scan opened is not in this stream"
        );
        // ...but it is there for the next statement.
        assert_eq!(
            collect(db.execute("SELECT author FROM docs;").expect("compiles")).len(),
            2
        );

        db.close().unwrap();
    }

    #[test]
    fn a_drained_stream_is_fused() {
        let (_dir, db) = docs_db();
        insert(&db, "alice", "My doc", 7);

        let mut stream = db.execute("SELECT author FROM docs;").expect("compiles");
        assert!(stream.next().is_some());
        for _ in 0..3 {
            assert!(stream.next().is_none(), "a drained stream stays drained");
        }

        db.close().unwrap();
    }

    #[test]
    fn an_empty_collection_streams_nothing() {
        let (_dir, db) = docs_db();
        let stream = db.execute("SELECT author FROM docs;").expect("compiles");
        // Labels still land — a header is knowable without any rows.
        assert_eq!(stream.labels(), ["author"]);
        assert!(collect(stream).is_empty());
        db.close().unwrap();
    }

    // -- WHERE, end to end --------------------------------------------------

    /// Four rows whose `published_at` spans a usable range, and whose vectors
    /// put `bob` NEARER the query than `carol` while giving them different
    /// `published_at` values — the arrangement the prefilter test needs.
    fn seed_filterable(db: &Db) {
        for (author, x, n) in [
            ("alice", 1.0, 1),
            ("bob", 0.9, 5),
            ("carol", 0.0, 1),
            ("dave", 0.0, 9),
        ] {
            db.execute(&format!(
                "INSERT INTO docs (author, vector, title, published_at) \
                 VALUES ('{author}', [{x}, 0.0, 0.0, 0.0], 'doc', {n});"
            ))
            .expect("seeded row inserts");
        }
    }

    /// The `author` of each returned row, in stream order.
    fn authors(stream: RowStream) -> Vec<String> {
        collect(stream)
            .into_iter()
            .map(|row| match &row[0] {
                RegValue::Str(s) => s.clone(),
                other => panic!("expected an author string, got {other:?}"),
            })
            .collect()
    }

    /// `SELECT author FROM docs WHERE <predicate>`, as authors.
    fn where_authors(db: &Db, predicate: &str) -> Vec<String> {
        let sql = format!("SELECT author FROM docs WHERE {predicate};");
        authors(db.execute(&sql).expect("the filtered select runs"))
    }

    /// A query answers as of ONE instant, negated predicates included.
    ///
    /// `!=` compiles to `BitmapEq` followed by `BitmapNot`, and the complement
    /// is taken within the live set. Those are two separate reads of liveness
    /// unless the query pins a snapshot — and if they disagree, the result can
    /// describe a row set that existed at no moment in time.
    ///
    /// Opens the stream, pulls one row so the snapshot is resolved, then
    /// deletes a row the predicate matches. The stream must still produce it:
    /// it was live when the query started.
    #[test]
    fn a_negated_where_answers_as_of_one_snapshot() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        let mut stream = db
            .execute("SELECT author FROM docs WHERE published_at != 5;")
            .expect("compiles");

        // First pull resolves the snapshot; the rest of the scan is pinned to it.
        let first = stream.next().expect("at least one row").expect("a row");
        assert!(matches!(&first.0[0], RegValue::Str(s) if s == "alice"));

        // Retire two of the remaining matches out from under the open stream.
        db.delete(0, 2).expect("delete carol"); // published_at = 1
        db.delete(0, 3).expect("delete dave"); // published_at = 9

        let mut got = vec!["alice".to_string()];
        for row in stream {
            match &row.expect("a row").0[0] {
                RegValue::Str(s) => got.push(s.clone()),
                other => panic!("expected an author, got {other:?}"),
            }
        }
        assert_eq!(
            got,
            ["alice", "carol", "dave"],
            "the stream lost rows that were live when it opened"
        );

        // A query started AFTER the deletes sees the smaller set — the snapshot
        // held the old rows, it did not fail to notice the deletes.
        assert_eq!(where_authors(&db, "published_at != 5"), ["alice"]);

        db.close().unwrap();
    }

    /// A tautological `WHERE` must return exactly what an unfiltered scan does.
    ///
    /// `p OR NOT p` exercises both a lookup and a complement in one query, so
    /// it is the shortest statement that can catch the two disagreeing about
    /// what is live.
    #[test]
    fn a_tautological_where_matches_an_unfiltered_scan() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        db.delete(0, 1).expect("delete bob");

        let scanned = authors(db.execute("SELECT author FROM docs;").expect("compiles"));
        let tautology = where_authors(&db, "published_at = 5 OR published_at != 5");

        assert_eq!(tautology, scanned);
        assert_eq!(scanned, ["alice", "carol", "dave"]);

        db.close().unwrap();
    }

    #[test]
    fn select_where_filters_rows() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        assert_eq!(where_authors(&db, "published_at < 2"), ["alice", "carol"]);
        assert_eq!(where_authors(&db, "published_at = 5"), ["bob"]);
        assert_eq!(
            where_authors(&db, "published_at >= 5"),
            ["bob", "dave"],
            "an inclusive bound includes its endpoint"
        );
        assert_eq!(where_authors(&db, "author = 'carol'"), ["carol"]);
        db.close().unwrap();
    }

    #[test]
    fn and_intersects_or_unions() {
        // The two combining ops, checked against results rather than bitmaps:
        // AND narrows, OR widens.
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        assert_eq!(
            where_authors(&db, "published_at < 6 AND author = 'bob'"),
            ["bob"]
        );
        assert_eq!(
            where_authors(&db, "published_at = 5 OR published_at = 9"),
            ["bob", "dave"]
        );
        // Precedence survives the whole pipeline: `a OR b AND c` is
        // `a OR (b AND c)`, so `alice` (matching the OR's left side) comes back
        // even though she fails the AND.
        assert_eq!(
            where_authors(
                &db,
                "author = 'alice' OR published_at = 5 AND author = 'bob'"
            ),
            ["alice", "bob"]
        );
        db.close().unwrap();
    }

    #[test]
    fn not_equal_excludes_only_the_named_value() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        assert_eq!(
            where_authors(&db, "author != 'alice'"),
            ["bob", "carol", "dave"]
        );
        assert_eq!(
            where_authors(&db, "published_at != 1"),
            ["bob", "dave"],
            "both rows holding the excluded value are dropped"
        );
        db.close().unwrap();
    }

    #[test]
    fn a_predicate_matching_nothing_returns_an_empty_result_with_labels() {
        // An empty bitmap must produce a well-formed empty result, not an
        // error: `SeekFirst` jumps past the loop body, and the labels still
        // describe what came back empty.
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        let stream = db
            .execute("SELECT author FROM docs WHERE published_at = 424242;")
            .expect("a predicate matching nothing still runs");
        assert_eq!(stream.labels(), ["author"]);
        assert!(collect(stream).is_empty());
        db.close().unwrap();
    }

    #[test]
    fn search_where_is_a_prefilter_not_a_postfilter() {
        // THE semantic that is easy to get wrong and impossible to notice.
        //
        // `bob` (x=0.9) is NEARER the query than `carol` (x=0.0), but `bob`
        // fails `published_at < 2`. A prefilter therefore returns
        // [alice, carol] — two rows, the two nearest AMONG THOSE MATCHING.
        // Ranking first and filtering after would rank [alice, bob], drop
        // `bob`, and return a single row — fewer than the requested TOP 2,
        // while looking entirely plausible.
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        let stream = db
            .execute(
                "SEARCH TOP 2 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs \
                 WHERE published_at < 2 RETURNING author;",
            )
            .expect("the filtered search runs");
        assert_eq!(
            authors(stream),
            ["alice", "carol"],
            "TOP k counts rows that satisfy the predicate"
        );
        db.close().unwrap();
    }

    #[test]
    fn search_where_preserves_score_order() {
        // The filter narrows the candidate set; it must not disturb the
        // ranking within it.
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        let stream = db
            .execute(
                "SEARCH TOP 3 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs \
                 WHERE published_at != 9 RETURNING author;",
            )
            .expect("the filtered search runs");
        // alice (1.0) > bob (0.9) > carol (0.0); dave is excluded.
        assert_eq!(authors(stream), ["alice", "bob", "carol"]);
        db.close().unwrap();
    }

    #[test]
    fn an_ordered_comparison_on_text_is_refused_at_the_boundary() {
        // The metadata index answers a TEXT range with an EMPTY bitmap by
        // design, so accepting this would return zero rows and look exactly
        // like a collection with no matching data. It must be an error the
        // user sees, not a silent wrong answer.
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        // `RowStream` is not `Debug`, so `expect_err` is unavailable.
        let message = match db.execute("SELECT author FROM docs WHERE author < 'm';") {
            Ok(_) => panic!("an ordered TEXT comparison must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("author"),
            "the error names the column: {message}"
        );
        db.close().unwrap();
    }

    #[test]
    fn a_where_clause_naming_an_unknown_column_is_a_bind_error() {
        let (_dir, db) = docs_db();
        if db.execute("SELECT author FROM docs WHERE nope = 1;").is_ok() {
            panic!("an unknown column in WHERE must be refused");
        }
        db.close().unwrap();
    }

    #[test]
    fn a_deleted_row_never_satisfies_a_predicate() {
        // Every lookup is masked by `live`, and `!=` complements WITHIN it —
        // so a tombstone cannot be resurrected by negating a predicate. Rows
        // here are all live, so this pins the complement's BOUND: `!=` must
        // return only real rows, never the unused ordinal space up to
        // capacity.
        let (_dir, db) = docs_db();
        seed_filterable(&db);
        assert_eq!(
            where_authors(&db, "author != 'nobody'").len(),
            4,
            "the complement is bounded by the live set, not by capacity"
        );
        db.close().unwrap();
    }
    // -- DELETE, end to end through the front door ---------------------------

    /// Every author still visible, sorted — the oracle for what a DELETE left.
    fn remaining(db: &Db) -> Vec<String> {
        let mut all = authors(
            db.execute("SELECT author FROM docs WHERE published_at > 0;")
                .expect("the select runs"),
        );
        all.sort();
        all
    }

    /// Run a DELETE and insist it behaves like every mutation: no rows out.
    fn delete(db: &Db, sql: &str) {
        let stream = db.execute(sql).expect("the delete runs");
        assert!(collect(stream).is_empty(), "a DELETE yields no rows");
    }

    #[test]
    fn delete_removes_exactly_the_rows_its_where_admits() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        delete(&db, "DELETE FROM docs WHERE published_at = 1;");

        assert_eq!(remaining(&db), vec!["bob", "dave"]);
        db.close().unwrap();
    }

    #[test]
    fn delete_takes_compound_predicates() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        // AND binds tighter than OR: (published_at = 1 AND author = 'alice') OR author = 'dave'.
        delete(
            &db,
            "DELETE FROM docs WHERE published_at = 1 AND author = 'alice' OR author = 'dave';",
        );

        assert_eq!(remaining(&db), vec!["bob", "carol"]);
        db.close().unwrap();
    }

    /// V-SQL has no NOT; `!=` is its negation, built as a complement WITHIN
    /// the query's snapshot. A DELETE through it must hit only rows that were
    /// live — it must neither resurrect a row deleted earlier nor reach past
    /// the live set into ordinals no row has been given yet.
    #[test]
    fn a_negated_delete_stays_inside_the_live_set() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        delete(&db, "DELETE FROM docs WHERE author = 'alice';");
        delete(&db, "DELETE FROM docs WHERE author != 'bob';");
        assert_eq!(remaining(&db), vec!["bob"]);
        assert!(
            where_authors(&db, "author = 'alice'").is_empty(),
            "no resurrection"
        );

        // A row inserted afterwards lands at a fresh ordinal and is visible:
        // the complement did not pre-retire ordinal space beyond the live set.
        insert(&db, "eve", "late", 3);
        assert_eq!(remaining(&db), vec!["bob", "eve"]);
        db.close().unwrap();
    }

    #[test]
    fn a_delete_matching_nothing_succeeds_and_logs_nothing() {
        let (dir, db) = docs_db();
        seed_filterable(&db);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).expect("wal exists").len();
        delete(&db, "DELETE FROM docs WHERE author = 'nobody';");

        assert_eq!(std::fs::metadata(&wal).expect("wal exists").len(), before);
        assert_eq!(remaining(&db), vec!["alice", "bob", "carol", "dave"]);
        db.close().unwrap();
    }

    #[test]
    fn a_delete_survives_reopen() {
        let (dir, db) = docs_db();
        seed_filterable(&db);
        delete(&db, "DELETE FROM docs WHERE published_at = 1;");
        drop(db); // no closing checkpoint: the WAL must carry the delete

        let db = Db::open(dir.path(), &[], DbOptions::default()).expect("reopen");
        assert_eq!(remaining(&db), vec!["bob", "dave"]);
        db.close().unwrap();
    }

    #[test]
    fn malformed_and_misaddressed_deletes_are_errors_not_panics() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        assert!(matches!(
            db.execute("DELETE FROM docs;"),
            Err(Error::Parse(_))
        ));
        assert!(matches!(
            db.execute("DELETE FROM nope WHERE author = 'alice';"),
            Err(Error::Bind(BindError::CollectionNotFound(_)))
        ));
        assert_eq!(remaining(&db).len(), 4, "a refused DELETE removes nothing");
        db.close().unwrap();
    }
    // -- WHERE <key> = <value> -----------------------------------------------

    #[test]
    fn a_row_can_be_named_by_its_key() {
        let (_dir, db) = docs_db();
        seed_filterable(&db); // authors alice/bob/carol/dave — `author` is the key

        assert_eq!(where_authors(&db, "author = 'carol'"), vec!["carol"]);
        assert_eq!(where_authors(&db, "author = 'alice'"), vec!["alice"]);
        // No such key: no row, no error.
        assert!(where_authors(&db, "author = 'nobody'").is_empty());
        db.close().unwrap();
    }

    /// An ordinal names a row, not a slot: once the row is retired, naming it
    /// matches nothing. Otherwise naming the key would be a back door around
    /// the liveness snapshot every other predicate goes through.
    #[test]
    fn a_retired_row_cannot_be_named_by_its_key() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        delete(&db, "DELETE FROM docs WHERE author = 'carol';");
        assert!(where_authors(&db, "author = 'carol'").is_empty());
        assert_eq!(remaining(&db).len(), 3);
        db.close().unwrap();
    }

    #[test]
    fn a_key_predicate_composes_with_a_column_one() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        assert_eq!(
            where_authors(&db, "author = 'bob' AND published_at = 5"),
            vec!["bob"]
        );
        assert!(where_authors(&db, "author = 'bob' AND published_at = 9").is_empty());
        db.close().unwrap();
    }

    // -- UPDATE, end to end through the front door ---------------------------

    /// Run an UPDATE and insist it behaves like every mutation: no rows out.
    fn update(db: &Db, sql: &str) {
        let stream = db.execute(sql).expect("the update runs");
        assert!(collect(stream).is_empty(), "an UPDATE yields no rows");
    }

    /// `(author, published_at)` of every live row, sorted.
    fn rows_of(db: &Db) -> Vec<(String, i64)> {
        let stream = db
            .execute("SELECT author, published_at FROM docs WHERE published_at > 0;")
            .expect("the select runs");
        let mut out: Vec<(String, i64)> = collect(stream)
            .into_iter()
            .map(|row| match (&row[0], &row[1]) {
                (RegValue::Str(a), RegValue::Int(n)) => (a.clone(), *n),
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn update_rewrites_the_rows_its_where_admits() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        update(
            &db,
            "UPDATE docs SET author = 'archived' WHERE published_at = 1;",
        );

        assert_eq!(
            rows_of(&db),
            vec![
                ("archived".to_string(), 1),
                ("archived".to_string(), 1),
                ("bob".to_string(), 5),
                ("dave".to_string(), 9),
            ],
            "alice and carol were rewritten; their other columns survived"
        );
        db.close().unwrap();
    }

    /// The embedding is copied across untouched, which only a vector search can
    /// show: the rewritten row must still be found by its original vector.
    #[test]
    fn update_copies_the_embedding_unchanged() {
        let (_dir, db) = docs_db();
        seed_filterable(&db); // alice's vector is [1.0, 0, 0, 0]

        update(
            &db,
            "UPDATE docs SET author = 'archived' WHERE published_at = 1;",
        );

        let found = authors(
            db.execute("SEARCH TOP 1 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
                .expect("the search runs"),
        );
        assert_eq!(
            found,
            vec!["archived"],
            "the rewritten row is still reachable by its original embedding"
        );
        db.close().unwrap();
    }

    #[test]
    fn an_update_matching_nothing_succeeds_and_logs_nothing() {
        let (dir, db) = docs_db();
        seed_filterable(&db);

        let wal = dir.path().join("wal.log");
        let before = std::fs::metadata(&wal).expect("wal exists").len();
        update(
            &db,
            "UPDATE docs SET author = 'x' WHERE published_at = 404;",
        );

        assert_eq!(std::fs::metadata(&wal).expect("wal exists").len(), before);
        assert_eq!(rows_of(&db).len(), 4);
        db.close().unwrap();
    }

    #[test]
    fn an_update_survives_reopen() {
        let (dir, db) = docs_db();
        seed_filterable(&db);
        update(
            &db,
            "UPDATE docs SET author = 'archived' WHERE published_at = 9;",
        );
        drop(db); // no closing checkpoint: the WAL must carry it

        let db = Db::open(dir.path(), &[], DbOptions::default()).expect("reopen");
        assert_eq!(
            rows_of(&db),
            vec![
                ("alice".to_string(), 1),
                ("archived".to_string(), 9),
                ("bob".to_string(), 5),
                ("carol".to_string(), 1),
            ]
        );
        db.close().unwrap();
    }

    /// Two threads updating the same row at the same time. Both statements
    /// must TAKE EFFECT — the loser re-resolves and tries again inside
    /// `execute` — and the row must stay one row.
    ///
    /// The liveness version is the witness: an applied Mutate bumps it, a
    /// conflicted one does not. So a statement whose conflict was swallowed
    /// instead of retried shows up as a missing bump, which is exactly the
    /// silent lost update this has to rule out.
    /// Full replacement, the statement from the roadmap. Proving the embedding
    /// REALLY changed takes two searches: the row must be found by its new
    /// vector, and a search for its old one must now prefer a different row.
    #[test]
    fn update_replaces_a_named_rows_embedding() {
        let (_dir, db) = docs_db();
        seed_filterable(&db); // alice [1.0,0,0,0] at id 0, bob [0.9,0,0,0] at id 1

        update(
            &db,
            "UPDATE docs SET vector = [0.0, 0.0, 0.0, 1.0], title = 'v2' WHERE author = 'alice';",
        );

        // Found by the NEW embedding...
        assert_eq!(
            authors(
                db.execute("SEARCH TOP 1 NEAREST TO [0.0, 0.0, 0.0, 1.0] FROM docs;")
                    .expect("the search runs")
            ),
            vec!["alice"]
        );
        // ...and no longer by the old one: bob is now the nearest to it.
        assert_eq!(
            authors(
                db.execute("SEARCH TOP 1 NEAREST TO [1.0, 0.0, 0.0, 0.0] FROM docs;")
                    .expect("the search runs")
            ),
            vec!["bob"],
            "alice's old embedding is gone, not merely shadowed"
        );
        db.close().unwrap();
    }

    /// The embedding may only be assigned to a row named by its KEY. Any other
    /// predicate could match many rows, and one statement carries one vector —
    /// so handing it to all of them is never what was meant.
    #[test]
    fn setting_a_vector_needs_a_key_predicate() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        // `title` is an ordinary column: every seeded row has title 'doc', so
        // this is exactly the "one vector, many rows" mistake.
        assert!(matches!(
            db.execute("UPDATE docs SET vector = [0.0, 0.0, 0.0, 1.0] WHERE title = 'doc';"),
            Err(Error::Bind(BindError::CannotSetVector { column })) if column == "vector"
        ));
        // A range over the key is not an identity either.
        assert!(matches!(
            db.execute("UPDATE docs SET vector = [0.0, 0.0, 0.0, 1.0] WHERE published_at < 9;"),
            Err(Error::Bind(BindError::CannotSetVector { .. }))
        ));
        db.close().unwrap();
    }

    /// The ordinal is no longer addressable, and that is the point of the key:
    /// an ordinal moves when a row is rewritten and is renumbered wholesale by
    /// compaction, so a statement that named one could silently hit a different
    /// row later. `id` is now an ordinary name with nothing behind it.
    #[test]
    fn a_row_cannot_be_named_by_its_ordinal() {
        let (_dir, db) = docs_db();
        seed_filterable(&db);

        assert!(matches!(
            db.execute("SELECT author FROM docs WHERE id = 0;"),
            Err(Error::Bind(BindError::ColumnNotFound(name))) if name == "id"
        ));
        assert!(matches!(
            db.execute("DELETE FROM docs WHERE id = 0;"),
            Err(Error::Bind(BindError::ColumnNotFound(name))) if name == "id"
        ));
        db.close().unwrap();
    }

    #[test]
    fn a_replaced_embedding_survives_reopen() {
        let (dir, db) = docs_db();
        seed_filterable(&db);
        update(
            &db,
            "UPDATE docs SET vector = [0.0, 0.0, 0.0, 1.0] WHERE author = 'alice';",
        );
        drop(db); // no closing checkpoint: the WAL must carry it

        let db = Db::open(dir.path(), &[], DbOptions::default()).expect("reopen");
        assert_eq!(
            authors(
                db.execute("SEARCH TOP 1 NEAREST TO [0.0, 0.0, 0.0, 1.0] FROM docs;")
                    .expect("the search runs")
            ),
            vec!["alice"]
        );
        db.close().unwrap();
    }

    #[test]
    fn concurrent_updates_through_execute_all_take_effect() {
        use std::sync::{Arc, Barrier};

        const ROUNDS: usize = 8;
        let (_dir, db) = docs_db();
        insert(&db, "alice", "doc", 1); // the single contended row

        let db = Arc::new(db);
        let barrier = Arc::new(Barrier::new(2));
        let before = db.live_snapshot(0).expect("collection open").version();

        let workers: Vec<_> = (0..2)
            .map(|tag| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    // Failures are COLLECTED, not panicked: every round must
                    // still reach its barrier, or a failing thread would leave
                    // the other one blocked forever and the test would hang
                    // instead of failing.
                    let mut failures = Vec::new();
                    for round in 0..ROUNDS {
                        barrier.wait();
                        if let Err(e) = db.execute(&format!(
                            "UPDATE docs SET title = 't{tag}-{round}' WHERE author = 'alice';"
                        )) {
                            failures.push(e.to_string());
                        }
                    }
                    failures
                })
            })
            .collect();
        let failures: Vec<String> = workers
            .into_iter()
            .flat_map(|w| w.join().expect("worker thread"))
            .collect();
        assert!(failures.is_empty(), "updates failed: {failures:?}");

        assert_eq!(
            db.live_snapshot(0).expect("collection open").version(),
            before + (2 * ROUNDS) as u64,
            "every statement applied: none was silently dropped on a conflict"
        );
        assert_eq!(rows_of(&db).len(), 1, "one row stays one row");
    }
}
