# Security notes and threat model

Flats is an embedded library: no network listener, no auth, no server. Its
attack surface is the input the embedding application hands it.

| Input | Threat | Defence | Verified by |
|---|---|---|---|
| Statement text | Injection through string-built SQL | `?` parameters, substituted after lexing; values never become syntax or names | `tests/params.rs`, SDK suites |
| Statement text | Crash or exhaustion through deep or huge predicates | caps: nesting 64, 4,096 comparisons (chains rebalanced), 256 MiB of text | `tests/limits.rs` (largest legal predicate on a 256 KiB stack) |
| Statement text | Runaway queries | `interrupt()`, statement timeout | `tests/limits.rs` |
| CREATE | A collection that can't be materialized, which used to poison the log forever | dim, capacity and size caps checked before logging | `tests/limits.rs` |
| Database files | Corrupt or crafted WAL, catalog, snapshots, index | CRCs, version checks, size-capped frames, no panics | `tests/fuzz.rs`, including forged-checksum cases |
| Concurrency | Two writers on one directory | exclusive OS lock | `tests/lock.rs` (cross-process) |
| FFI handles | Use after close | shared, reference-counted handles; close waits for in-flight calls | `sdk/c/tests/abi.rs` |
| Crashes and I/O errors | Lost or resurrected writes at any step of a commit, checkpoint, truncation or compaction | metadata index checkpointed last; a failed WAL write is cut back out of the log (or the log is poisoned) | `tests/crash_kill.rs` (SIGKILL), `tests/failpoints.rs` (every step); see `docs/testing.md` |
| Store files | A damaged or missing snapshot silently dropping rows | open rebuilds from the log only when it provably holds everything, refuses otherwise; `Db::open_salvage` is the explicit opt-in | `tests/snapshot_corruption.rs` |
| `unsafe` code | Memory errors | `// SAFETY:` on every block; AddressSanitizer over the lib, C ABI, SIMD, mmap and concurrency suites | CI `sanitizers` job |

## Known gaps (decisions for the maintainer)

- A damaged snapshot is now a refusal, not silent loss, but the rows it held
  can't be recovered without a backup: salvage keeps only what the log
  still has. Keeping the previous snapshot per store would make recovery
  lossless.
- Power loss (unflushed writes lost) is not simulated yet; SIGKILL keeps the
  page cache. LazyFS in CI would cover it (`docs/testing.md`).
- A collection whose whole directory was deleted opens empty: with no files
  left, it looks like one being created for the first time.
- A damaged active header slot in `vectors.idx` falls back to the older slot
  (that is how a torn header write is survived). Combined with a damaged
  snapshot in the same collection, that older watermark could wrongly pass
  the "the log still has everything" test.
- A CREATE whose files fail to materialize for reasons other than the caps
  (for example, a full disk) still poisons the log.
- A column named `score` cannot be filtered on: it collides with the
  `score` pseudo-column.
- `backup(dest)` writes wherever it is told. Don't pass it user-controlled
  paths.
- The Miri CI job is untested; it was added at the end of the session.
