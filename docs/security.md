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
| `unsafe` code | Memory errors | `// SAFETY:` on every block; AddressSanitizer over the lib, C ABI, SIMD, mmap and concurrency suites | CI `sanitizers` job |

## Known gaps (decisions for the maintainer)

- **Snapshot bit-rot after a clean checkpoint is silent data loss.** A
  metadata or tuple snapshot that fails its checks makes that store start
  empty and rebuild from the WAL. After a clean close the WAL has been
  truncated, so the rows are gone with no error. Forged-snapshot fuzzing
  confirmed this. Refusing to open (as the catalog already does) would be
  safer.
- A CREATE whose files fail to materialize for reasons other than the caps
  (for example, a full disk) still poisons the log.
- A column named `score` cannot be filtered on: it collides with the
  `score` pseudo-column.
- `backup(dest)` writes wherever it is told. Don't pass it user-controlled
  paths.
- The Miri CI job is untested; it was added at the end of the session.
