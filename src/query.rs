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
use crate::sql::bind::BoundStatement;
use crate::sql::{BindError, ParseError, Statement, analyze, parse, plan};
use crate::vm::exec::{ExecError, OutputRow, Vm};

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
            Error::Unsupported { .. } => None,
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
        let mutates = matches!(
            bound,
            BoundStatement::Insert(_) | BoundStatement::CreateCollection(_)
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

        // 6. BUILD — `Vm::new` runs `Program::validate`.
        let mut vm = Vm::new(program)?;

        if mutates {
            // EAGER: drain inside `execute`. A mutation yields no rows, and its
            // side effect must be durable before this returns.
            while vm.step(self)?.is_some() {}
            Ok(RowStream::empty())
        } else {
            // LAZY: open the cursors, then detach. Everything after this pulls
            // without a `Db`.
            vm.open_cursors(self)?;
            Ok(RowStream::over(labels, vm))
        }
    }
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
        // A bare SEARCH projects every scalar, so its labels come from the same
        // place a `SELECT *`'s do.
        BoundStatement::Search(search) => {
            search.projection.iter().map(|c| c.name.clone()).collect()
        }
        BoundStatement::Insert(_) | BoundStatement::CreateCollection(_) => Vec::new(),
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
    use super::{Error, RowStream};
    use crate::compiler::bytecode::{Addr, Cursor, Op, Program, Reg};
    use crate::compiler::constants::ConstPool;
    use crate::engine::{Db, DbOptions};
    use crate::error::Error as EngineError;
    use crate::sql::BindError;
    use crate::vm::exec::{RegValue, Vm};

    const CREATE: &str = "CREATE COLLECTION docs (
        author TEXT,
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
            db.execute("CREATE COLLECTION broken (v VECTOR(0), a INT) WITH (capacity = 10);"),
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
        // it pre-empts everything it does know. Two exist:
        //
        //   * `id` is the ordinal pseudo-column SEARCH returns, so a stored
        //     column may not shadow it. A storage rule, invisible to the binder.
        assert!(matches!(
            db.execute("CREATE COLLECTION reserved (v VECTOR(2), id INT) WITH (capacity = 4);"),
            Err(Error::Engine(EngineError::ReservedColumn(name))) if name == "id"
        ));
        //   * capacity is a RUNTIME property — how many rows are already there —
        //     so only the engine can know the collection is full. This one also
        //     shows why the eager/lazy split matters: the INSERT ran INSIDE
        //     `execute`, so its failure is an outer error, not a stream item.
        db.execute("CREATE COLLECTION tiny (v VECTOR(2), a INT) WITH (capacity = 1);")
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
            "CREATE COLLECTION alpha (emb VECTOR(2), title TEXT, tag TEXT) WITH (capacity = 8);",
        )
        .expect("alpha is created");
        db.execute(
            "CREATE COLLECTION beta (tag TEXT, emb VECTOR(2), title TEXT) WITH (capacity = 8);",
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
}
