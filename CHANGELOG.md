# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
follow [docs/versioning.md](docs/versioning.md).

## Unreleased (since `d64e6d8`, "reclaim dead rows with background compaction")

### Added

- **SDKs**:
  - a C ABI (`sdk/c`, `flats.h`);
  - Python over ctypes (`sdk/python`), with an asyncio API (`flats.aio`),
    type stubs and a wheel build;
  - Node.js over N-API (`sdk/node`), with a Promise API, TypeScript types and
    prebuilt-addon packaging.
- **V-SQL**:
  - `?` parameters;
  - `LIMIT` / `OFFSET`;
  - multi-row `INSERT`;
  - `UPSERT INTO … VALUES …`;
  - `DROP COLLECTION`;
  - float exponents (`1e-5`).
- **Rust API**:
  - `Db::execute_with`, `Db::insert_many`, `Db::upsert_many`;
  - `Db::backup`, `Db::open_salvage`, `Db::stats`, `Db::row_count`;
  - `Db::interrupt` and statement timeouts (`Limits`);
  - `flats::diagnostics::set_log_hook`.
- **SDK conveniences**: `upsert`, `get`/`delete` by key, `collections`,
  `describe`; cursors that stream large results.
- **Safety limits**: statement size, expression depth and term caps;
  collection size checks before anything is logged.
- **A directory lock**: a database can be open in one handle at a time.
- **Testing**: differential testing against a reference model and against
  SQLite; SIGKILL and fault-point crash tests; a syscall-order power-loss
  check; coverage-guided fuzzing; concurrency tests; upgrade tests against a
  database written by the previous build; coverage measurement. Nightly CI
  runs all of it on fresh seeds and files issues (`docs/testing.md`).

### Changed

- **The public API is curated:** internal modules are private unless the
  `internals` feature is on (tests only, no semver).
- **A damaged or missing store file is refused** when the WAL can't rebuild
  it, instead of silently opening empty. The same goes for a collection whose
  directory is gone. `Db::open_salvage` is the explicit way past.
- **Batches coerce INT into FLOAT columns**, as V-SQL does.
- **A column named `score` is filterable;** `RETURNING score` in a SEARCH is
  still the similarity.
- **Upserts never conflict:** the commit thread decides what each one
  replaces.

### Fixed

- **Crash safety:**
  - a crash between a checkpoint's two metadata snapshots could open with
    rows whose values were missing (the metadata index is now checkpointed
    last);
  - a failed or torn WAL write could bring a refused write back on restart,
    or hide every later acknowledged write (the log is now cut back);
  - a kill during CREATE COLLECTION could leave the database unable to open
    (the vector index is now created atomically).
- **Power loss** (from the sync-order check): the database directory,
  a new WAL and the WAL truncation's rename were not made durable.
- **Lost or blocked writes:**
  - an idle collection held WAL truncation back forever;
  - a CREATE that couldn't make its files (a full disk) halted the database
    (the files are now made before it is logged);
  - two concurrent bulk inserts could insert the same primary key.
- **Hostile input:** deeply nested or very long predicates crashed the
  process.
- **Silent failures:** background checkpoint and compaction failures were
  dropped; they are now reported.

### Formats

All readable from earlier versions (`tests/upgrade.rs`):

- **WAL v7:** adds `DropCollection` (v6) and `Upsert` (v7). v5 and v6 logs
  replay.
- **Catalog v5:** adds `next_id` (v4) and the "files before registration"
  promise (v5). v3 and v4 catalogs open.
