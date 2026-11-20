# Deterministic simulation testing: a plan

**Status:** proposal; nothing built. It is a change to the storage core, so
it's the code owner's call.

## Why

Today's harnesses (`docs/testing.md`) find bugs, but two kinds don't replay:

- **Races.** `concurrent_differential` and `crash_kill` run real threads
  with real timing. The crash-mid-checkpoint bug failed about 1 run in 5 at
  the same seed.
- **Power loss.** A SIGKILL keeps the page cache, so nothing tests that every
  `fsync` and directory sync is really where it has to be.

Deterministic simulation, the approach used by FoundationDB, TigerBeetle and
Turso, removes both gaps. One seed decides everything: the order threads run
in, the clock, which write fails, and what a power cut throws away. A failure
then replays exactly, every time, and shrinks like the model test's do.

## What has to go behind an interface

Measured in `src/`:

| Surface | Uses | Simulated as |
|---|---|---|
| Files (`OpenOptions`, `read`, `write`, `rename`, `remove_*`, `read_dir`, `metadata`, `set_len`) | ~100 call sites | in-memory files, each with durable and cached contents |
| `sync_data` / `sync_all` / directory fsync | 23 | cached → durable; a power cut keeps only what's durable, plus a seeded random subset of the cache (torn writes) |
| `MmapMut` (the vector index) | 11 | a buffer whose `flush` (msync) is the sync point |
| `thread::spawn` (commit thread, flusher, compactor, cursors) | 14 | tasks run one at a time by a seeded scheduler |
| `Instant::now` (timeouts, flusher cadence) | 7 | a simulated clock the scheduler advances |

## Shape

1. **A `platform::Io` trait.** It's the one door to files, syncs, mmap, the
   clock and spawning. The real implementation is today's code, moved and
   unchanged. `unsafe` stays in `platform/`, as CLAUDE.md requires.
2. **A `Db` built on it.** `Db::open` gets the real `Io`, and a test-only
   constructor (behind `internals`) takes another one. No public API changes.
3. **`tests/sim` gains a simulated `Io`:**
   - files are an in-memory map;
   - syncs move bytes from cached to durable;
   - `crash()` keeps durable bytes plus a seeded subset of cached ones;
   - a cooperative scheduler interleaves the engine's threads at every I/O
     call and channel operation, choosing who runs next from the seed.
4. **The existing model and op generator run on top unchanged.** They gain
   two new op kinds: power cut, and "a sync fails".

The threads are the hard part. The engine uses `std::sync::mpsc` and
`Mutex`. A scheduler can only interleave at points it controls, so:
- channel sends and receives, and the I/O calls, go through `Io` (as
  `Io::channel`, say);
- locks held without I/O inside stay real. Their interleavings are what the
  concurrency suites cover.

## Cost

- **Code:** about 2–3 weeks. Most of it is mechanical, moving call sites
  behind the trait. The scheduler and the mmap shim are the real work.
- **Risk:** the real `Io` must behave exactly as today. Every existing suite
  runs on it unchanged, which is the check.
- **Gain:**
  - every race and every crash point replays from a seed;
  - power loss gets tested, not assumed;
  - runs are much faster: no fsyncs, so thousands of crash-recover cycles a
    second instead of a few.

## Cheaper interim step

LazyFS in CI (a FUSE filesystem that drops unsynced data on command) covers
power loss without touching the engine. Races stay non-replayable. It needs
`libfuse3-dev` on the runner. Couldn't be verified on the dev machine (no FUSE
headers, no sudo), so it hasn't been added.
