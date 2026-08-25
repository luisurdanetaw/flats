//! `flats` — an interactive V-SQL shell.
//!
//! A read-eval-print loop over [`Db::execute`]. It owns no query logic: every
//! statement goes through the same pipeline the library exposes, so anything
//! that works here works from Rust and vice versa. What lives here is only what
//! a TERMINAL needs — reading lines, splitting them into statements, drawing a
//! table, and never dying on bad input.
//!
//! # Why statement splitting lives in the shell
//!
//! `execute` takes exactly ONE statement and rejects trailing tokens. That is
//! the right library contract — a caller passing two statements and getting one
//! `RowStream` back would have nowhere to put the second result. But a person
//! typing at a prompt writes `SELECT a FROM t; SELECT b FROM t;` on one line and
//! expects two tables. Reconciling those is a shell concern, so [`split`] does
//! it here rather than widening the library's contract.
//!
//! The splitter is deliberately NOT a second lexer. It tracks exactly the two
//! constructs in which a `;` is not a terminator — a `'…'` string (where `''`
//! is an escaped quote) and a `-- …` line comment — because those are the only
//! two the real lexer has. Anything else it hands to the parser verbatim and
//! lets the real error surface. Keeping it this small is what keeps it honest:
//! it cannot disagree with the lexer about a construct it does not model.
//!
//! # Two modes, one loop
//!
//! Whether stdin is a TTY changes four things, and nothing else:
//!
//! | | interactive | script (piped) |
//! |---|---|---|
//! | prompts + banner | shown | silent, so output is clean |
//! | timing / row counts | shown | omitted |
//! | a statement error | printed, loop continues | printed, **stops**, exit 1 |
//! | input | edited, with history | read a line at a time |
//!
//! A human retypes a bad line; a script cannot. Continuing past a failed
//! `CREATE` would run every dependent `INSERT` against a collection that does
//! not exist, turning one real error into a cascade of confusing ones — so a
//! script stops at the first failure and says so in its exit code.
//!
//! # Ctrl-C interrupts work, it does not end the session
//!
//! The default `SIGINT` disposition kills the process where it stands, which
//! skips [`Db::close`] and leaves a WAL that should have been checkpointed for
//! the next open to replay. (Nothing is lost either way — writes are fsync'd
//! before they are acked — but paying a recovery pass for an ordinary Ctrl-C
//! is silly.) So Ctrl-C is caught, and what it cancels depends on what is
//! happening:
//!
//! - **while typing** — abandons the line, keeps the session. The editor sees
//!   it as a byte, not a signal: `ISIG` is off in raw mode.
//! - **while a statement streams rows** — stops the output, keeps the session.
//! - **in a script** — ends the run with exit 130, still through the normal
//!   path, so the final checkpoint happens.
//!
//! # Output is bounded, not materialized
//!
//! A table cannot be drawn until its column widths are known, and widths are
//! not known until the cells are. The naive reading of that — collect every
//! row, then measure — makes memory scale with the RESULT, so `SELECT * FROM`
//! a large collection would push the whole thing through the shell's heap on
//! the way to a terminal that only ever shows the last screenful.
//!
//! [`Table`] measures a bounded sample instead ([`SAMPLE_ROWS`]), commits the
//! header once it has one, and streams everything after that straight to the
//! writer. Memory is O(sample); the cost is that an unusually wide cell
//! arriving after the header overflows its column instead of widening it.
//!
//! # Output is written, not `println!`ed
//!
//! Every byte of result output goes through an [`io::Write`] whose errors are
//! handled. `println!` PANICS if stdout is gone, and stdout is routinely gone —
//! `flats db | head`, or quitting a pager early, closes it mid-table. A shell
//! that aborts there is as broken as a library that panics on typable input, so
//! [`ErrorKind::BrokenPipe`] is an ordinary, silent, successful exit. Threading
//! the writer also means the table renderer is unit-testable against a `Vec<u8>`
//! rather than only inspectable by eye.
//!
//! # Usage
//!
//! ```text
//! cargo run -- [PATH]        # PATH defaults to ./flats-data
//! ```
//!
//! Statements end with `;` and may span lines. `.help` lists the meta-commands.

mod editor;

use std::io::{self, BufRead, ErrorKind, IsTerminal, Lines, StdinLock, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use flats::engine::{Db, DbOptions};
use flats::metadata::common::{CollectionConfig, ColumnType};
use flats::platform::signal;
use flats::platform::tty::RawMode;
use flats::vm::exec::RegValue;

use editor::{Editor, Input, Keys};

/// Where a database lives when the command line does not say.
const DEFAULT_PATH: &str = "./flats-data";

const PROMPT: &str = "flats> ";
/// Shown while a statement is still open (no `;` yet). Same width as [`PROMPT`]
/// so typed text stays in one column across a continuation.
const CONTINUATION: &str = "   ...> ";

/// The terminal's stdin. Named because the editor needs the raw descriptor,
/// and `0` at a call site says nothing.
const STDIN_FD: i32 = 0;

/// Exit code when a script's statement failed. Distinct from 2 (bad usage) and
/// 1 (could not open the database), so a caller can tell the three apart.
const EXIT_STATEMENT_FAILED: i32 = 1;

/// Exit code for a run ended by Ctrl-C. 128 + SIGINT, the shell convention —
/// `$?` is how a script runner tells "the query failed" from "someone stopped
/// it".
const EXIT_INTERRUPTED: i32 = 130;

/// How many rows are held to measure column widths before the header is
/// committed and the rest of the result streams straight through.
///
/// Large enough that any realistic terminal-sized result is measured exactly,
/// small enough that the buffer is irrelevant next to the rows themselves.
const SAMPLE_ROWS: usize = 256;

fn main() {
    // FIRST, before the database is touched. `Db::open` replays the WAL, which
    // is exactly the kind of pause during which someone presses Ctrl-C — and
    // with the default disposition that kills the process mid-recovery, which
    // is the failure this handler exists to prevent. Arming it after `open`
    // would leave that window open.
    //
    // Best effort: if it cannot be installed we keep the default disposition,
    // which costs a WAL replay on the next open but loses nothing — every
    // acked write is already fsync'd.
    let _ = signal::install();

    let path = match args() {
        Ok(path) => path,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let db = match Db::open(&path, &[], DbOptions::default()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("cannot open database at {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let interactive = io::stdin().is_terminal();
    let code = {
        // Buffered: a large result set is thousands of small writes, and an
        // unbuffered stdout syscalls on every one. Flushed before each prompt
        // and once at the end — see `repl`.
        let stdout = io::stdout();
        let mut out = io::BufWriter::new(stdout.lock());
        let code = repl(&db, &mut out, &path, interactive);
        // A final flush can itself hit a broken pipe; that is still a clean
        // exit, not a new failure.
        match out.flush() {
            Ok(()) => code,
            Err(e) if e.kind() == ErrorKind::BrokenPipe => code,
            Err(e) => {
                eprintln!("error writing output: {e}");
                1
            }
        }
    };

    // Close explicitly: `close` takes `self`, runs a final checkpoint, and
    // reports a durability failure that `drop` could only swallow. Note this is
    // an OPTIMIZATION, not the durability guarantee — every write is already
    // fsync'd through the WAL before it is acked, so a kill -9 here costs a WAL
    // replay on next open, never data.
    if let Err(e) = db.close() {
        eprintln!("error closing database: {e}");
        std::process::exit(1);
    }
    std::process::exit(code);
}

/// Parse the command line. One optional positional argument: the database path.
fn args() -> Result<PathBuf, String> {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        None => PathBuf::from(DEFAULT_PATH),
        Some(a) if a == "-h" || a == "--help" => {
            return Err(format!(
                "usage: flats [PATH]\n\n\
                 Opens the Flats database in PATH, creating it if absent.\n\
                 PATH defaults to {DEFAULT_PATH}."
            ));
        }
        Some(a) if a.starts_with('-') => return Err(format!("unknown option: {a}")),
        Some(a) => PathBuf::from(a),
    };
    match args.next() {
        Some(extra) => Err(format!("unexpected argument: {extra}")),
        None => Ok(path),
    }
}

/// The loop. Returns the process exit code.
///
/// Reads lines, accumulating them until the buffer holds at least one complete
/// statement, then runs each one. Ends on EOF, on `.quit`, on a closed stdout,
/// or — in script mode only — on the first statement error.
fn repl(db: &Db, out: &mut impl Write, path: &Path, interactive: bool) -> i32 {
    if interactive {
        // Best-effort: if the banner cannot be written, the loop below will
        // discover the same broken pipe and exit cleanly.
        let _ = writeln!(
            out,
            "flats — V-SQL shell   (.help for commands, .quit to exit)\ndatabase: {}",
            path.display()
        );
    }

    let mut source = Source::open(interactive);
    // Holds a partial statement across iterations, so a statement can span
    // lines.
    let mut buffer = String::new();

    let code = loop {
        let prompt = if buffer.trim().is_empty() {
            PROMPT
        } else {
            CONTINUATION
        };

        let line = match source.next_line(prompt, out, interactive) {
            Ok(Input::Line(line)) => line,
            // EOF (Ctrl-D, or the end of a piped script).
            Ok(Input::Eof) => {
                // A statement left open at EOF was never terminated. Say so
                // rather than silently discarding what was typed.
                if !buffer.trim().is_empty() {
                    eprintln!("error: unterminated statement at end of input (missing `;`)");
                    break EXIT_STATEMENT_FAILED;
                }
                break 0;
            }
            // Ctrl-C at the prompt. The half-typed statement goes away — the
            // whole point of the key — but the session does not.
            Ok(Input::Interrupted) => {
                if !interactive {
                    break EXIT_INTERRUPTED;
                }
                buffer.clear();
                continue;
            }
            Err(e) => {
                eprintln!("error reading input: {e}");
                break 1;
            }
        };

        // A meta-command is only a meta-command at the START of a statement.
        // Inside an open one, `.` is the parser's business — a lone `.` is not
        // valid V-SQL, but reporting THAT is the parser's job, not a silent
        // reinterpretation of the user's half-typed INSERT.
        if buffer.trim().is_empty() {
            let trimmed = line.trim();
            if trimmed.starts_with('.') {
                match meta(db, out, trimmed) {
                    Ok(Meta::Handled) => continue,
                    Ok(Meta::Quit) => break 0,
                    Err(e) => break exit_for(e),
                }
            }
        }

        buffer.push_str(&line);
        // The newline the reader stripped. Load-bearing: `-- comment` ends at a
        // line break, so without this a comment would swallow the next line.
        buffer.push('\n');

        // Run every COMPLETE statement in the buffer; whatever trails the last
        // `;` stays for the next line to finish.
        let (statements, rest) = split(&buffer);
        buffer = rest;
        let mut stop = None;
        for statement in statements {
            match run(db, out, &statement, interactive) {
                Ok(Outcome::Ok) => {}
                // The statement failed. A human just retypes it; a script
                // cannot, so it stops here with a meaningful exit code.
                Ok(Outcome::Failed) if !interactive => stop = Some(EXIT_STATEMENT_FAILED),
                // Ctrl-C mid-result. The rows already printed are real; a
                // script stops, a person gets their prompt back.
                Ok(Outcome::Canceled) if !interactive => stop = Some(EXIT_INTERRUPTED),
                Ok(_) => {}
                Err(e) => stop = Some(exit_for(e)),
            }
            if stop.is_some() {
                // Whatever followed the failed statement on the same line is
                // abandoned with it.
                break;
            }
        }
        if let Some(code) = stop {
            break code;
        }
    };

    source.finish();
    code
}

/// Where the shell's input comes from.
///
/// The two arms are not a preference, they are a capability check: raw mode is
/// what makes editing possible, and it is only available on a terminal. Piped
/// input gets [`Source::Plain`] because there is no cursor to move.
enum Source {
    /// A terminal, with editing and history.
    Editing {
        editor: Editor,
        /// Boxed because `Keys` carries a read buffer, which would otherwise
        /// make this variant hundreds of bytes larger than `Plain` — and an
        /// enum is as big as its largest variant.
        keys: Box<Keys<StdinLock<'static>>>,
    },
    /// A pipe, a file, or a terminal that refused raw mode.
    Plain { lines: Lines<StdinLock<'static>> },
}

impl Source {
    fn open(interactive: bool) -> Self {
        // Probe once, here, rather than discovering per keystroke: enabling
        // raw mode is the only way to know the terminal allows it, and a shell
        // that switched modes halfway through a session would be baffling. The
        // guard is dropped immediately, restoring the mode it found.
        if interactive && RawMode::enable(STDIN_FD).is_ok() {
            Source::Editing {
                editor: Editor::new(),
                keys: Box::new(Keys::new(io::stdin().lock())),
            }
        } else {
            Source::Plain {
                lines: io::stdin().lock().lines(),
            }
        }
    }

    fn next_line(
        &mut self,
        prompt: &str,
        out: &mut impl Write,
        interactive: bool,
    ) -> io::Result<Input> {
        match self {
            Source::Editing { editor, keys } => {
                // Raw mode is held for exactly the length of the read. While a
                // statement RUNS the terminal must be normal again: that is
                // what lets Ctrl-C become a signal the row loop can see, and
                // what keeps ordinary output looking ordinary.
                let _raw = RawMode::enable(STDIN_FD)?;
                editor.read_line(prompt, keys, out)
            }
            Source::Plain { lines } => {
                if interactive {
                    // A prompt has no newline, so it does not reach the
                    // terminal until the buffer is flushed — which must happen
                    // before we block on input, or the user stares at a blank
                    // line.
                    write!(out, "{prompt}")?;
                    out.flush()?;
                }
                loop {
                    return match lines.next() {
                        None => {
                            if interactive {
                                writeln!(out)?;
                            }
                            Ok(Input::Eof)
                        }
                        Some(Ok(line)) => Ok(Input::Line(line)),
                        // The read was interrupted by a signal. Our SIGINT
                        // handler is installed WITHOUT `SA_RESTART` so that it
                        // surfaces here instead of silently resuming — which
                        // is what lets Ctrl-C reach a script blocked on a pipe
                        // that never delivers.
                        Some(Err(e)) if e.kind() == ErrorKind::Interrupted => {
                            if signal::take() {
                                Ok(Input::Interrupted)
                            } else {
                                continue;
                            }
                        }
                        Some(Err(e)) => Err(e),
                    };
                }
            }
        }
    }

    /// Persist anything the session accumulated. History is the only such
    /// thing today.
    fn finish(&self) {
        if let Source::Editing { editor, .. } = self {
            editor.save();
        }
    }
}

/// Turn a write failure into an exit code.
///
/// A closed stdout is how `| head` and a quit pager END a program — the reader
/// got what it wanted. It is a SUCCESSFUL, silent exit; complaining about it
/// would put noise on every `flats … | head`.
fn exit_for(e: io::Error) -> i32 {
    if e.kind() == ErrorKind::BrokenPipe {
        0
    } else {
        eprintln!("error writing output: {e}");
        1
    }
}

/// What a meta-command wants the loop to do next.
enum Meta {
    /// Command ran (or was rejected); keep looping.
    Handled,
    /// Exit the shell.
    Quit,
}

/// Handle a `.`-prefixed shell command.
///
/// A bad meta-command is NOT a statement failure: it never reached the database,
/// and stopping a script over a shell typo would be surprising. It reports to
/// stderr and the loop continues.
fn meta(db: &Db, out: &mut impl Write, line: &str) -> io::Result<Meta> {
    let mut parts = line.split_whitespace();
    let command = parts.next().unwrap_or(".");
    let argument = parts.next();

    match command {
        ".quit" | ".exit" => return Ok(Meta::Quit),
        ".help" => print_help(out)?,
        ".tables" => {
            let collections = db.collections();
            if collections.is_empty() {
                writeln!(out, "no collections")?;
            } else {
                for c in collections {
                    writeln!(out, "{}", c.name)?;
                }
            }
        }
        ".schema" => {
            let collections = db.collections();
            let shown: Vec<&CollectionConfig> = match argument {
                Some(name) => collections.iter().filter(|c| c.name == name).collect(),
                None => collections.iter().collect(),
            };
            if shown.is_empty() {
                match argument {
                    Some(name) => eprintln!("error: no such collection: {name}"),
                    None => writeln!(out, "no collections")?,
                }
            } else {
                for (i, c) in shown.iter().enumerate() {
                    if i > 0 {
                        writeln!(out)?;
                    }
                    print_schema(out, c)?;
                }
            }
        }
        other => eprintln!("error: unknown command: {other}   (.help for a list)"),
    }
    Ok(Meta::Handled)
}

fn print_help(out: &mut impl Write) -> io::Result<()> {
    write!(
        out,
        "\
meta-commands
  .help              show this message
  .tables            list collections
  .schema [NAME]     show a collection's columns (all collections if omitted)
  .quit, .exit       leave the shell  (Ctrl-D also works)

keys
  up / down          previous and next statement from history
  left / right       move by character   (ctrl+left / ctrl+right by word)
  ctrl-a / ctrl-e    start / end of line
  ctrl-w / ctrl-u    delete the previous word / to the start of the line
  ctrl-k / ctrl-l    delete to end of line / clear the screen
  ctrl-c             abandon the line, or stop a running statement

statements end with `;` and may span several lines
  CREATE COLLECTION docs (vector VECTOR(3), author TEXT) WITH (capacity = 1000);
  INSERT INTO docs (vector, author) VALUES ([0.1, 0.2, 0.3], 'alice');
  SELECT * FROM docs;
  SEARCH TOP 5 NEAREST TO [0.1, 0.2, 0.3] FROM docs RETURNING id, score, author;
"
    )
}

/// Print one collection as a `CREATE COLLECTION`-shaped listing.
///
/// Columns are printed in DECLARATION order — the order they were typed —
/// which means interleaving the vector back among the scalars. The stored
/// schema keeps them apart (the embedding lives in a different store and has no
/// scalar `ColumnId`), so declaration order is reconstructed from the ordinals
/// both sides carry for exactly this purpose.
fn print_schema(out: &mut impl Write, c: &CollectionConfig) -> io::Result<()> {
    writeln!(out, "{}   (capacity {})", c.name, c.capacity)?;

    let mut columns: Vec<(usize, String, String)> = c
        .schema
        .columns
        .iter()
        .map(|col| {
            let ty = match col.ty {
                ColumnType::Int => "INT",
                ColumnType::Float => "FLOAT",
                ColumnType::Text => "TEXT",
            };
            (col.ordinal.get(), col.name.clone(), ty.to_string())
        })
        .collect();
    let vector = &c.schema.vector;
    columns.push((
        vector.ordinal.get(),
        vector.name.clone(),
        format!("VECTOR({})", vector.dim.get()),
    ));
    columns.sort_by_key(|(ordinal, _, _)| *ordinal);

    let width = columns.iter().map(|(_, n, _)| n.len()).max().unwrap_or(0);
    for (_, name, ty) in columns {
        writeln!(out, "  {name:width$}  {ty}")?;
    }
    Ok(())
}

/// How a statement ended.
enum Outcome {
    /// It ran, and any rows it produced were printed.
    Ok,
    /// It was rejected, or faulted partway through its rows.
    Failed,
    /// Ctrl-C stopped it. Distinct from `Failed` because nothing is WRONG —
    /// the user asked it to stop — and because a script should report it as
    /// an interrupt rather than a query error.
    Canceled,
}

/// Run one statement and print its result or its error.
///
/// Statement errors are printed and swallowed rather than propagated — a shell
/// that exited on a typo would be unusable, and it is the caller that decides
/// whether a failure ends a script. The `io::Result` is a separate axis: it
/// reports the OUTPUT channel dying, which no statement can recover from.
fn run(db: &Db, out: &mut impl Write, sql: &str, interactive: bool) -> io::Result<Outcome> {
    let started = Instant::now();
    // Drop any Ctrl-C pressed before this statement began. Without this, an
    // interrupt aimed at the PREVIOUS query would cancel this one the moment
    // it produced a row.
    signal::clear();

    let stream = match db.execute(sql) {
        Ok(stream) => stream,
        Err(e) => {
            eprintln!("error: {e}");
            return Ok(Outcome::Failed);
        }
    };

    let labels: Vec<String> = stream.labels().to_vec();
    // A mutation reports no columns and yields no rows — it has already run by
    // the time `execute` returns (see the `query` module's eager-mutation rule).
    if labels.is_empty() {
        if interactive {
            writeln!(out, "ok  ({:.2?})", started.elapsed())?;
        }
        return Ok(Outcome::Ok);
    }

    let mut table = Table::new(labels);
    let mut rows = 0usize;
    let mut failure = None;
    let mut canceled = false;
    for row in stream {
        // Checked per row rather than per statement: the reason to press
        // Ctrl-C is a result that is still arriving.
        if signal::take() {
            canceled = true;
            break;
        }
        match row {
            Ok(row) => {
                let cells: Vec<String> = row.0.iter().map(render).collect();
                table.push(out, cells)?;
                rows += 1;
            }
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    // Emits the header for a result that never reached the sample size, so an
    // empty or interrupted result still shows its columns.
    table.finish(out)?;

    if interactive {
        let unit = if rows == 1 { "row" } else { "rows" };
        writeln!(out, "{rows} {unit}  ({:.2?})", started.elapsed())?;
    }
    // AFTER the table: the rows that arrived before the fault are real results,
    // and printing them first makes it obvious how far the scan got.
    if let Some(e) = failure {
        eprintln!("error after {rows} rows: {e}");
        return Ok(Outcome::Failed);
    }
    if canceled {
        eprintln!("canceled after {rows} rows");
        return Ok(Outcome::Canceled);
    }
    Ok(Outcome::Ok)
}

/// One cell's text.
fn render(value: &RegValue) -> String {
    match value {
        RegValue::Int(i) => i.to_string(),
        // A whole-numbered float prints as `1.0`, not `1`. `{}` gives the
        // latter, which is indistinguishable from an INT in a column of
        // right-aligned-looking numbers — and telling FLOAT from INT is most of
        // what someone reads a score column for. Non-finite values keep the
        // default spelling (`NaN`, `inf`), which have no fractional part to add.
        RegValue::Real(f) => {
            if f.is_finite() && f.fract() == 0.0 {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        RegValue::Str(s) => s.clone(),
        // Printed as a summary, not 768 floats across the terminal. Unreachable
        // today — projecting an embedding is refused at the boundary — but a
        // cell renderer that panicked on a variant would be a landmine for
        // whoever lands `VectorFetch`.
        RegValue::Vector(v) => format!("<vector dim={}>", v.len()),
        RegValue::Record(_) => "<record>".to_string(),
        // A WHERE intermediate. Never projected — it exists between the
        // predicate's ops and the cursor that opens over it.
        RegValue::Bitmap(b) => format!("<bitmap {} rows>", b.len()),
        // Never emitted: reading an unset register is an ExecError, so a row
        // carrying one cannot reach here.
        RegValue::Unset => "<unset>".to_string(),
    }
}

/// Draws a table incrementally, with bounded memory.
///
/// The tension it resolves: a column's width is the width of its widest cell,
/// which is not known until the last row has been seen — but holding every row
/// to find out makes memory scale with the result set.
///
/// So it holds at most [`SAMPLE_ROWS`]. Once that many have arrived (or the
/// result ends, whichever comes first) the header is written, the widths are
/// FROZEN, and every subsequent row goes straight to the writer. A later cell
/// wider than its column overflows rather than reflowing the table — which is
/// unavoidable for output already written, and is why the sample is generous
/// enough that no screenful-sized result ever reaches the streaming path.
///
/// Width is counted in CHARACTERS, not bytes, so a non-ASCII value does not
/// skew the column it sits in. (Full display-width handling — CJK, emoji —
/// needs a Unicode width table, which is a dependency; CLAUDE.md §2 says no,
/// and a slightly ragged emoji column is the right price.)
struct Table {
    labels: Vec<String>,
    widths: Vec<usize>,
    /// Rows held back to measure widths. Emptied by [`Table::commit`], and
    /// never longer than [`SAMPLE_ROWS`].
    pending: Vec<Vec<String>>,
    /// Whether the header has been written — after which `widths` is fixed.
    committed: bool,
}

impl Table {
    fn new(labels: Vec<String>) -> Self {
        // A column starts at least as wide as its own name, so a header is
        // never truncated by a table of narrow values.
        let widths = labels.iter().map(|l| l.chars().count()).collect();
        Table {
            labels,
            widths,
            pending: Vec::new(),
            committed: false,
        }
    }

    /// Add one row, writing it out if the header has already gone.
    fn push(&mut self, out: &mut impl Write, cells: Vec<String>) -> io::Result<()> {
        if self.committed {
            return writeln!(out, "{}", join_padded(&cells, &self.widths));
        }
        for (i, cell) in cells.iter().enumerate() {
            if i < self.widths.len() {
                self.widths[i] = self.widths[i].max(cell.chars().count());
            }
        }
        self.pending.push(cells);
        if self.pending.len() >= SAMPLE_ROWS {
            self.commit(out)?;
        }
        Ok(())
    }

    /// Write the header, the rule, and everything held back.
    fn commit(&mut self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "{}", join_padded(&self.labels, &self.widths))?;
        let rule: Vec<String> = self.widths.iter().map(|w| "-".repeat(*w)).collect();
        writeln!(out, "{}", join_padded(&rule, &self.widths))?;
        for row in std::mem::take(&mut self.pending) {
            writeln!(out, "{}", join_padded(&row, &self.widths))?;
        }
        self.committed = true;
        Ok(())
    }

    /// Finish the table. Idempotent, and required even for a result with no
    /// rows — an empty result must still show WHICH columns came back empty.
    fn finish(&mut self, out: &mut impl Write) -> io::Result<()> {
        if !self.committed {
            self.commit(out)?;
        }
        Ok(())
    }
}

/// Join `cells` with two spaces, padding each to its column width. The final
/// column is not padded, so no line carries trailing whitespace.
fn join_padded(cells: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    for (i, cell) in cells.iter().enumerate() {
        if i > 0 {
            out.push_str("  ");
        }
        out.push_str(cell);
        if i + 1 < cells.len() {
            let width = widths.get(i).copied().unwrap_or(0);
            let pad = width.saturating_sub(cell.chars().count());
            out.extend(std::iter::repeat_n(' ', pad));
        }
    }
    out
}

/// Split `input` into complete statements plus the unterminated remainder.
///
/// Each returned statement KEEPS its `;` — that is what `execute` expects, and
/// re-adding it here would be a second place that has to agree about the
/// terminator.
///
/// A statement that is only whitespace or comments is DROPPED rather than run:
/// a blank line and a `-- note` line are things people type constantly, and
/// `execute("")` is (correctly) a parse error. Filtering them is the shell's
/// job precisely because the library is right to refuse them.
fn split(input: &str) -> (Vec<String>, String) {
    let mut statements = Vec::new();
    let mut current = String::new();
    // The two states in which `;` is not a terminator. See the module header:
    // these are the only two the real lexer has.
    let mut in_string = false;
    let mut in_comment = false;

    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if in_comment {
            current.push(c);
            if c == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            current.push(c);
            if c == '\'' {
                // `''` is an escaped quote and the string CONTINUES. Consuming
                // the second quote here is what stops it from being read as a
                // fresh opening quote.
                if chars.peek() == Some(&'\'') {
                    current.push(chars.next().expect("peeked"));
                } else {
                    in_string = false;
                }
            }
            continue;
        }
        match c {
            '\'' => {
                in_string = true;
                current.push(c);
            }
            '-' if chars.peek() == Some(&'-') => {
                in_comment = true;
                current.push(c);
                current.push(chars.next().expect("peeked"));
            }
            ';' => {
                current.push(c);
                if !is_blank(&current) {
                    statements.push(current.clone());
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    // An unterminated string or comment stays in the remainder: both end at a
    // boundary the next line supplies (a closing quote, a line break), so the
    // buffer is simply not finished yet.
    if is_blank(&current) {
        current.clear();
    }
    (statements, current)
}

/// Whether `s` holds no statement text — only whitespace and `--` comments.
fn is_blank(s: &str) -> bool {
    let mut rest = s.trim();
    loop {
        if rest.is_empty() {
            return true;
        }
        if rest == ";" {
            // A bare `;` — the user pressed enter on an empty statement.
            return true;
        }
        if let Some(after) = rest.strip_prefix("--") {
            // Skip to the end of the comment line, then re-test what follows.
            rest = match after.find('\n') {
                Some(i) => after[i + 1..].trim(),
                None => return true,
            };
            continue;
        }
        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EXIT_INTERRUPTED, EXIT_STATEMENT_FAILED, SAMPLE_ROWS, Table, exit_for, is_blank,
        join_padded, render, split,
    };
    use flats::vm::exec::RegValue;
    use std::io::{self, ErrorKind, Write};
    use std::sync::Arc;

    /// `split` returns owned statements; comparing against `&str` is easier.
    fn parts(input: &str) -> (Vec<String>, String) {
        let (statements, rest) = split(input);
        (statements, rest)
    }

    /// Push every row through a [`Table`] and return what it drew.
    fn table(labels: &[&str], rows: &[&[&str]]) -> String {
        let mut out = Vec::new();
        let mut table = Table::new(labels.iter().map(|s| s.to_string()).collect());
        for row in rows {
            let cells = row.iter().map(|s| s.to_string()).collect();
            table.push(&mut out, cells).expect("a Vec never fails to write");
        }
        table.finish(&mut out).expect("a Vec never fails to write");
        String::from_utf8(out).expect("output is utf-8")
    }

    // -- statement splitting -------------------------------------------------

    #[test]
    fn one_statement_per_terminator() {
        let (statements, rest) = parts("SELECT a FROM t; SELECT b FROM t;");
        assert_eq!(statements, ["SELECT a FROM t;", " SELECT b FROM t;"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn an_unterminated_statement_stays_in_the_remainder() {
        // What makes a statement span lines: the tail is handed back, and the
        // next line is appended to it.
        let (statements, rest) = parts("SELECT a\n");
        assert!(statements.is_empty());
        assert_eq!(rest, "SELECT a\n");

        let (statements, rest) = parts(&format!("{rest}FROM t;\n"));
        assert_eq!(statements, ["SELECT a\nFROM t;"]);
        assert!(rest.trim().is_empty());
    }

    #[test]
    fn a_semicolon_inside_a_string_is_not_a_terminator() {
        // The reason the splitter models strings at all. Splitting naively here
        // would hand the parser two fragments, each with an unterminated
        // literal.
        let (statements, rest) = parts("INSERT INTO t (a) VALUES ('x;y');");
        assert_eq!(statements, ["INSERT INTO t (a) VALUES ('x;y');"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn a_doubled_quote_does_not_end_the_string() {
        // `''` is the lexer's escape (`'it''s'` lexes to `it's`), so the string
        // continues — and the `;` inside it is still not a terminator.
        let (statements, rest) = parts("INSERT INTO t (a) VALUES ('it''s; ok');");
        assert_eq!(statements, ["INSERT INTO t (a) VALUES ('it''s; ok');"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn a_semicolon_inside_a_comment_is_not_a_terminator() {
        let (statements, rest) = parts("SELECT a FROM t -- ; not this\n;");
        assert_eq!(statements, ["SELECT a FROM t -- ; not this\n;"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn a_comment_ends_at_the_line_break() {
        // The newline the reader strips has to be pushed back, or the comment
        // would swallow the statement typed on the following line. This is that
        // guarantee, from the splitter's side.
        let (statements, rest) = parts("-- a note\nSELECT a FROM t;");
        assert_eq!(statements, ["-- a note\nSELECT a FROM t;"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn blank_and_comment_only_input_yields_no_statements() {
        // These are what a person types constantly, and `execute("")` is
        // correctly a parse error — so the shell must never send them.
        for input in ["", "   ", "\n\n", "-- just a note\n", "  -- note\n\n", ";"] {
            let (statements, rest) = parts(input);
            assert!(statements.is_empty(), "{input:?} produced {statements:?}");
            assert!(rest.is_empty(), "{input:?} left {rest:?}");
        }
    }

    #[test]
    fn an_unterminated_string_stays_open_across_lines() {
        // The remainder keeps the open literal, so the next line continues it
        // rather than starting a new statement.
        let (statements, rest) = parts("INSERT INTO t (a) VALUES ('start\n");
        assert!(statements.is_empty());
        assert_eq!(rest, "INSERT INTO t (a) VALUES ('start\n");
    }

    #[test]
    fn a_statement_after_a_complete_one_is_kept_as_the_remainder() {
        let (statements, rest) = parts("SELECT a FROM t; SELECT b");
        assert_eq!(statements, ["SELECT a FROM t;"]);
        assert_eq!(rest, " SELECT b");
    }

    #[test]
    fn is_blank_sees_through_comments_but_not_statements() {
        assert!(is_blank(""));
        assert!(is_blank("  \n "));
        assert!(is_blank("-- note"));
        assert!(is_blank("-- note\n  -- another\n"));
        assert!(!is_blank("-- note\nSELECT 1"));
        assert!(!is_blank("SELECT 1"));
    }

    // -- table rendering -----------------------------------------------------

    #[test]
    fn a_table_has_a_header_a_rule_and_one_line_per_row() {
        // Column `a` is one character wide (label and cells all fit), so it is
        // separated from `bb` by exactly the two-space gutter and no padding.
        let out = table(&["a", "bb"], &[&["1", "2"], &["3", "4"]]);
        assert_eq!(out, "a  bb\n-  --\n1  2\n3  4\n");
    }

    #[test]
    fn a_table_with_no_rows_still_prints_its_header() {
        // An empty result must show WHICH columns came back empty — that is the
        // difference between "no matches" and "wrong query".
        assert_eq!(table(&["author"], &[]), "author\n------\n");
    }

    #[test]
    fn columns_widen_to_their_widest_cell() {
        let out = table(&["a"], &[&["longest value"], &["x"]]);
        assert_eq!(out, "a\n-------------\nlongest value\nx\n");
    }

    #[test]
    fn no_line_carries_trailing_whitespace() {
        // Otherwise every row ends in padding, which shows up in diffs and
        // copy-paste.
        let out = table(&["a", "b"], &[&["wide value", "x"]]);
        for line in out.lines() {
            assert!(!line.ends_with(' '), "trailing space in {line:?}");
        }
    }

    #[test]
    fn the_last_column_is_not_padded() {
        let widths = vec![5, 5];
        let line = join_padded(&["a".to_string(), "b".to_string()], &widths);
        assert_eq!(line, "a      b");
        assert!(!line.ends_with(' '));
    }

    #[test]
    fn padding_counts_characters_not_bytes() {
        // A multi-byte value must not skew its column: `é` is two BYTES but one
        // character, so padding it by byte length would under-pad by one.
        let widths = vec![3, 1];
        let line = join_padded(&["é".to_string(), "x".to_string()], &widths);
        assert_eq!(line, "é    x");

        // The same property through the real renderer.
        let out = table(&["c", "n"], &[&["é", "1"], &["abc", "2"]]);
        assert_eq!(out, "c    n\n---  -\né    1\nabc  2\n");
    }

    // -- cell rendering ------------------------------------------------------

    #[test]
    fn a_whole_numbered_float_is_distinguishable_from_an_int() {
        // `SEARCH … RETURNING id, score` puts an INT and a FLOAT side by side;
        // a score of exactly 1 must not read as the integer 1.
        assert_eq!(render(&RegValue::Real(1.0)), "1.0");
        assert_eq!(render(&RegValue::Int(1)), "1");
        assert_eq!(render(&RegValue::Real(0.5)), "0.5");
    }

    #[test]
    fn no_register_variant_panics_when_rendered() {
        // Including the ones no query can emit today. A cell renderer that
        // panicked on a variant would be a landmine for whoever lands
        // `VectorFetch`.
        let values = [
            RegValue::Unset,
            RegValue::Int(-7),
            RegValue::Real(f64::NAN),
            RegValue::Real(f64::INFINITY),
            RegValue::Str(String::new()),
            RegValue::Str("ünïcødé".to_string()),
            RegValue::Vector(Arc::from(vec![0.0f32, 1.0].as_slice())),
        ];
        for v in &values {
            assert!(!render(v).is_empty() || matches!(v, RegValue::Str(_)));
        }
        assert_eq!(
            render(&RegValue::Vector(Arc::from(vec![0.0f32; 768].as_slice()))),
            "<vector dim=768>"
        );
    }

    // -- write failures ------------------------------------------------------

    /// A writer that fails every write with a chosen kind — stands in for a
    /// closed pipe.
    struct Failing(ErrorKind);

    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(self.0, "synthetic"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(self.0, "synthetic"))
        }
    }

    #[test]
    fn a_broken_pipe_is_a_clean_successful_exit() {
        // `flats db | head` closes stdout mid-table. The reader got what it
        // wanted; that is not a failure, and it must not be reported as one.
        assert_eq!(exit_for(io::Error::new(ErrorKind::BrokenPipe, "x")), 0);
    }

    #[test]
    fn a_real_write_failure_is_an_error_exit() {
        assert_eq!(
            exit_for(io::Error::new(ErrorKind::PermissionDenied, "x")),
            1
        );
        assert_eq!(EXIT_STATEMENT_FAILED, 1);
        // 128 + SIGINT, so `$?` distinguishes a stopped run from a failed one.
        assert_eq!(EXIT_INTERRUPTED, 130);
    }

    #[test]
    fn printing_a_table_propagates_a_write_failure_instead_of_panicking() {
        // The property that keeps `| head` from aborting: the renderer returns
        // the error rather than reaching a `println!` that would panic.
        let mut sink = Failing(ErrorKind::BrokenPipe);
        let mut table = Table::new(vec!["a".to_string()]);
        let err = table.finish(&mut sink).expect_err("must fail");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    }

    // -- bounded output ------------------------------------------------------

    #[test]
    fn rows_beyond_the_sample_are_not_held_in_memory() {
        // The property, stated directly: memory is O(sample), not O(result).
        // A shell that buffered everything would grow `pending` without bound
        // here, and `SELECT *` over a real collection would push the whole
        // thing through the heap on its way to a terminal.
        let mut out = Vec::new();
        let mut table = Table::new(vec!["n".to_string()]);
        for i in 0..(SAMPLE_ROWS * 4) {
            table
                .push(&mut out, vec![i.to_string()])
                .expect("a Vec never fails to write");
            assert!(
                table.pending.len() < SAMPLE_ROWS,
                "held {} rows after {i}",
                table.pending.len()
            );
        }
        table.finish(&mut out).expect("a Vec never fails to write");

        // Every row still reaches the output — bounding memory must not lose
        // data.
        let text = String::from_utf8(out).expect("output is utf-8");
        assert_eq!(text.lines().count(), SAMPLE_ROWS * 4 + 2, "header + rule");
        assert!(text.lines().nth(2) == Some("0"));
        assert!(text.lines().last() == Some(&format!("{}", SAMPLE_ROWS * 4 - 1)));
    }

    #[test]
    fn the_header_is_written_before_the_result_is_complete() {
        // What makes the output STREAM: a long result starts appearing while
        // it is still being produced, rather than after. Once the sample is
        // full the header must already be on the wire.
        let mut out = Vec::new();
        let mut table = Table::new(vec!["n".to_string()]);
        for i in 0..SAMPLE_ROWS {
            table
                .push(&mut out, vec![i.to_string()])
                .expect("a Vec never fails to write");
        }
        assert!(
            !out.is_empty(),
            "nothing was written before the result ended"
        );
        let text = String::from_utf8(out).expect("utf-8");
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("n"), "the header is out");
        assert!(
            lines.next().is_some_and(|l| l.chars().all(|c| c == '-')),
            "and its rule"
        );
    }

    #[test]
    fn a_cell_wider_than_its_frozen_column_is_not_truncated() {
        // The accepted cost of streaming: widths are fixed once the header is
        // out, so a late wide value overflows its column. It must still print
        // in FULL — a ragged table is a cosmetic problem, a silently truncated
        // value is a wrong answer.
        let mut out = Vec::new();
        let mut table = Table::new(vec!["a".to_string(), "b".to_string()]);
        for _ in 0..SAMPLE_ROWS {
            table
                .push(&mut out, vec!["x".to_string(), "y".to_string()])
                .expect("a Vec never fails to write");
        }
        table
            .push(&mut out, vec!["a much wider value".to_string(), "z".to_string()])
            .expect("a Vec never fails to write");
        table.finish(&mut out).expect("a Vec never fails to write");

        let text = String::from_utf8(out).expect("output is utf-8");
        assert!(text.contains("a much wider value  z"), "{text}");
    }
}
