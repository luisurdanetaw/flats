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
| `tests/concurrent_differential.rs` | Writer and reader threads, with the checkpointer and compactor busy | Torn or mixed-version reads, lost writes under races |
| `tests/sync_order.rs` | Runs a workload under `strace` and checks the fsync/rename order against what a power cut needs | Missing or misplaced syncs |
| `sdk/c/tests/abi_fuzz.rs` | Random sequences of C ABI calls with in-contract hostile arguments, handles freed in random order (under ASan in CI) | Memory errors, crashes, undocumented statuses at the FFI boundary |
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
- **A kill during CREATE COLLECTION** could leave a sized but zero-filled
  vector index, and then the database refused to open at all ("bad magic
  number: 0"). The index file is now built under a temporary name and
  renamed into place once its header is durable.

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

Baseline (2026-11-19): **94.8% of lines, 93.0% of regions, 82.9% of
branches** of `src/`. The nightly job fails below 93% lines and publishes the
HTML report. The weakest spots when measured were the scalar kernel's
dispatch path on AVX2 machines (the aarch64 CI job covers NEON natively),
`vm/exec.rs` regions (80%) and `query.rs` branches (63%).

## Power loss, from the syscall trace

A SIGKILL keeps the page cache, so a missing fsync never shows up in
`crash_kill`. `tests/sync_order.rs` runs a workload under `strace -f -y`.
It then checks the order of the durability syscalls, following ALICE
(OSDI '14):

- an acknowledgment never leaves un-fsynced bytes in the WAL;
- a file is fsynced before it is renamed into place;
- a rename, a new file and a new directory are each followed by an fsync of
  their directory before the next acknowledgment.

Its first runs found three real power-loss gaps, all fixed:
- the database directory was created without syncing its parent;
- the WAL truncation's rename wasn't followed by a directory fsync, so a
  power cut could bring back the old log and lose every write acknowledged
  since;
- a new `wal.log` wasn't followed by a directory fsync.

Planted bugs were caught: a missing WAL `fdatasync` (rule 1) and an
un-synced snapshot rename (rule 2).

It checks ORDER, not recovery: whether replay works on what a power cut
leaves is the job of deterministic simulation
(`docs/deterministic-simulation.md`). The vector index's writes go through
`mmap`, which strace doesn't see; its durability rests on the `msync`
before each header update (`index.rs`).

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

- **Recovery after a power cut.** `sync_order` checks the sync order but
  never replays a power-cut state. Deterministic simulation would.
- **Deterministic concurrency**: `concurrent_differential` runs real threads,
  so a race it finds may not replay from its seed. A seeded scheduler
  driving all I/O and threads, as Turso does, would make every race replay
  exactly. It needs the storage layer's I/O behind one interface.
