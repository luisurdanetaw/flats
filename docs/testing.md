# How Flats is tested

Besides the ordinary unit and integration tests, five randomized harnesses
run on every pull request (briefly) and every night (for hours, on fresh
seeds), from `.github/workflows/fuzz.yml`. Every one of them is seeded: a
failure names its seed, and the seed replays it exactly.

| Harness | What it does | Catches |
|---|---|---|
| `tests/differential.rs` | Random workloads against the database AND a reference model | Wrong rows, wrong ranking, wrong errors |
| `fuzz/sqlite_diff.py` | The same `WHERE` text and parameters through Flats and SQLite, via the Python SDK | Predicate semantics, SDK parameter and batch paths |
| `tests/crash_kill.rs` | SIGKILLs a child process running a workload at random moments | Torn files, non-atomic steps, recovery bugs |
| `tests/failpoints.rs` | Fails, then crashes or carries on, at every step of every durability path | Crash-safety and error-handling bugs at exact points |
| `tests/fuzz.rs` | Hostile statement text, damaged and forged files | Panics, poisoning, silent misreads |
| `fuzz/run.sh` | Coverage-guided libFuzzer with AddressSanitizer: parser, full query pipeline, every decoder | Panics, memory errors, unbounded allocation or time |

## The reference model (`tests/sim`)

Flats does exact search, so every statement has exactly one right answer. A
few hundred lines of plain Rust compute it: rows in a map, predicates
evaluated row by row, scores as f64 dot products. The simulator generates
workloads of CREATE, DROP, INSERT (multi-row, shuffled column lists), UPDATE
(vector updates included), DELETE, SEARCH, SELECT with LIMIT/OFFSET,
checkpoints, clean reopens, crashes, compactions and backups. About one op in
five is malformed somewhere, and the model predicts exactly which ones the
binder must refuse. Each value is written inline or as a `?` parameter, at
random.

What's compared:

- **Rows:** exact, bit for bit, -0.0 included.
- **SEARCH:**
  - the right number of rows, all of them passing the filter;
  - in score order;
  - no row left out that scores higher than one let in.

  Each of these allows only the gap between the engine's f32 scores and the
  model's f64 ones, so ties may break either way.
- **Metamorphic checks:** a LIMIT/OFFSET result is a slice of the same query
  without it; a wider TOP k extends the narrower one; backups, compactions,
  reopens and crashes change nothing.

A failure is shrunk by delta debugging to a minimal workload and written to
`target/sim-failures/` (or `$FLATS_SIM_OUT`) with the replay command.

The model is only as good as its ability to notice bugs, so it was checked
by planting some. Each of these was caught, and shrunk to 3–6 statements:
- a `<=` that behaved like `<`;
- a `>=` off by one in the metadata index;
- a top-k heap that kept a row it should have replaced whenever the
  replacement scored within 0.01.

## Crashes

`crash_kill` kills a real process, so nothing is flushed and no destructor
runs. Every acknowledged write must survive. The one write in flight may or
may not have.

`failpoints` (`cargo test --release --features failpoints --test failpoints`)
uses named fault points compiled in only with that feature
(`src/failpoint.rs`). The sites are:
- the WAL write: before it, half of it, before and after its fsync;
- WAL truncation;
- each store's step in a checkpoint;
- the snapshot and catalog writes;
- both sides of a compaction's directory swap.

At each site it either crashes or carries on, and holds the engine to its
contract: an error means the op did not happen, now or after a restart.
The one exception is an error saying the record is durable.

Bugs these found, each now a regression test:

- **A crash between a checkpoint's two snapshots** left the metadata index
  ahead. Replay then refused rows as duplicates of themselves, and a row's
  values were lost. The metadata index is now checkpointed last.
- **A failed fsync** left the write in the log, so it came back on restart
  even though its caller had been told it failed.
- **A torn write** in the middle of the log made recovery drop every later
  acknowledged write. After a failed write, the log is now cut back to where
  the batch started.

## Coverage-guided fuzzing (`fuzz/`)

A standalone crate with its own workspace, so nothing in it is a dependency
of Flats. It adds `libfuzzer-sys`, which bundles libFuzzer. `fuzz/run.sh`
builds with the flags `cargo fuzz` would use (sanitizer coverage, comparison
tracing, ASan), seeds each corpus with real inputs (`fuzz/make_seeds.rs`),
and runs:

| Target | Input |
|---|---|
| `sql_parse` | any text into the lexer and parser |
| `sql_execute` | any SELECT/SEARCH through bind, plan, compile and the VM, on a live database |
| `wal_record` | a WAL frame payload, past its CRC |
| `metadata_snapshot`, `tuple_snapshot`, `catalog` | a snapshot body, past magic, version, length and CRC |

The decoders sit behind CRCs a fuzzer can never forge, so the hooks
(`fuzz_decode_*`, under the `internals` feature) re-frame the fuzzer's
bytes as a valid file. That lets it spend its time in the decoding itself.

```sh
fuzz/run.sh                    # every target, 60 s each (nightly toolchain)
fuzz/run.sh sql_execute 3600   # one target, an hour
fuzz/run.sh catalog 0 fuzz/artifacts/catalog/crash-…   # replay a crash
```

Nightly CI runs each target for 40 minutes on 2 cores, and caches the
corpus so it keeps growing across nights.

## Coverage

`scripts/coverage.sh [--html]` measures line, region and branch coverage of
`src/` under the whole suite, randomized harnesses included. It uses rustc's
`-C instrument-coverage` and the `llvm-tools` component, with no cargo plugin.
`COVERAGE_MIN=<pct>` makes it fail below a floor.

## Replaying and scaling

```sh
FLATS_SIM_SEED=1234 cargo test --release --test differential     # one seed
FLATS_SIM_SEEDS=500 FLATS_SIM_OPS=800 cargo test --release --test differential
FLATS_CRASH_ITERS=500 cargo test --release --test crash_kill
FLATS_FP_SEEDS=300 cargo test --release --features failpoints --test failpoints
python3 fuzz/sqlite_diff.py --seeds 500 --ops 400 --seed-base 1234
FLATS_SIM_STATS=1 cargo test --release --test differential -- --nocapture   # coverage
```

`FLATS_SIM_SEED_BASE` moves every harness to a different seed range; the
nightly job uses it to cover new ground each night.

## Not covered yet

- **Power loss**, meaning writes the OS never flushed are lost. A SIGKILL
  keeps the page cache, so it can't test fsync discipline. The tool for that
  is LazyFS, a FUSE filesystem that drops unsynced data on command. It needs
  `libfuse3-dev` on the runner and hasn't been wired in.
- **Concurrency**: the harnesses drive the database from one thread. The
  concurrency suites (`swmr`, `concurrent_stores`, `thread_safety`) cover
  races separately.
