# ARCHITECTURE.md — How the gears move

This document explains the inner workings of the three core modules — the flat
index (`src/index/index.rs`), the write-ahead log (`src/wal/wal.rs`), and the
engine that wires them together (`src/engine/mod.rs`). SIMD is out of scope here;
see `CLAUDE.md` §8.

For *scope, conventions, and on-disk byte layouts*, read `CLAUDE.md`. This file is
about the **runtime** — who owns what, which thread mutates which bytes, and why
the durability and concurrency properties hold.

## The shape of it

Three layers, one rule each:

- **`index.rs`** — the flat vector store. One writer, many readers, no lock.
- **`wal.rs`** — the durable log. One thread owns the file and *is* the writer.
- **`engine/mod.rs`** — wires them together and owns the durability contract
  end-to-end.

The whole design collapses to a single sentence:

> **The index's one `Writer` lives on the WAL commit thread, and nothing else
> ever mutates an index.**

Everything below is a consequence of that.

---

## 1. The flat index (`index.rs`)

### Layout

An mmap'd file: two header pages (slot A on page 0, slot B on page 1), then a
tombstone bitset, then the page-aligned `f32 * dim * capacity` vector region. A
vector's **ordinal is its position** — `vectors_offset + ordinal*dim*4` — not an
append counter. That positional identity is what makes replay idempotent (see
below).

```text
page 0:  header slot A   (64 bytes used)
page 1:  header slot B   (64 bytes used)
page 2+: tombstone bitset (ceil(capacity/8) bytes)
page N:  vectors          (f32 * dim * capacity), page-aligned start
```

### SWMR — how it's lock-free and still sound

`FlatIndex::create`/`open` hand back a `(Writer, Reader)` pair over a shared
`Arc<FlatIndexInner>`. The `Writer` is **`!Clone`** — exactly one exists — so its
`&mut self` methods are *honestly* exclusive; no two writes can race because
there is only one writer, period. `Reader` is `Clone`, handed to any number of
query threads, and only ever takes `&self`.

The single synchronization point is the `AtomicU32` `count` (the high-water mark):

- **Writer** copies a vector into slot `ordinal`, *then*
  `count.fetch_max(ordinal+1, Release)`.
- **Reader** does `count.load(Acquire)` and scans only ordinals `< count`.

That Release/Acquire pair is the entire memory-safety argument
(`FlatIndexInner::committed_vectors`, and the `unsafe impl Send/Sync` SAFETY
block): a reader that *observes* a count also observes every byte written before
the store that published it. Writers only ever touch slots `>= count` — slots no
reader is allowed to look at yet — so there's no torn read and no `&mut`/`&`
aliasing on the vector region.

The inner keeps a raw `base` pointer (captured once from `as_mut_ptr()` with
write provenance) instead of `UnsafeCell<MmapMut>` precisely to avoid forming a
transient `&mut MmapMut` on each write that would alias readers' `&` — see the
module doc "Why a cached raw pointer."

**The one exception:** tombstones are in-place mutation *below* `count`, where a
reader genuinely can be looking. So that byte is touched *only* through
`AtomicU8` (`fetch_or`, Relaxed) — never a plain `&mut u8`. That keeps it
well-defined. The bit is an independent flag; the vector it refers to was already
published via `count`, so Relaxed is enough.

### Crash-safe checkpoint: double-buffered header

The watermark (`count`, `last_lsn`, `seq`) lives in *two* header slots on
separate pages. Checkpoint writes the **spare** slot, flushes it, then flips
which is active:

```text
begin_checkpoint → sync_data → stage_watermark → sync_header
```

On open, the slot with the highest valid `seq` (CRC-checked) wins. A torn header
flush therefore can't lose the watermark — you fall back to the previous slot.
The strict order matters: `sync_data` flushes the vector/tombstone pages
**before** the header that claims they're durable, so a crash mid-checkpoint
never advertises data that didn't land.

---

## 2. The WAL (`wal.rs`)

### One thread, one file

Every operation that touches the log — `Append`, `Truncate`, `Checkpoint` —
flows through one mpsc `Command` channel to a single `wal-commit` thread. That
thread is the *sole* file owner, so control ops are cleanly ordered against
append batches (handled only *between* batches, never interleaved — see
`pending_control` in the commit loop). It also owns the `IndexApplier` — and thus
the index `Writer`s — which is how the WAL upholds the index's single-writer
invariant "for free."

### The ordering invariant (the heart of it)

```text
assign LSN  →  write frame  →  FSYNC  →  apply  →  ACK
```

(`commit_batch`). Each step's placement is load-bearing:

- **fsync strictly before apply** keeps WAL ≥ index at all times. That's what
  makes recovery "replay the tail onto the index" correct — the log can never be
  behind what's in the index.
- **ack strictly after fsync** makes durability *honest*: the caller is told
  "durable" only once it actually is. The caller acks its own client only after
  `append` returns `Ok`.
- **Group commit**: the loop drains every queued record (up to
  `MAX_BATCH = 1024`), writes all their frames, and fsyncs **once** for the whole
  batch before acking any. fsync is the expensive part; batching it is the only
  path to real throughput. Never fsync per record.

Because apply happens on this same thread *after* fsync, "committed" and
"visible" coincide — read-your-writes works with no extra machinery.

### Frame format & recovery

Each frame: `[len u32][crc32 u32][lsn u64][payload]`, with the crc covering
`lsn || payload`. Recovery walks frames in order and **stops at the first short
read or bad CRC** — that torn tail was a write caught mid-fsync, never acked, so
discarding it is correct. Everything before it is the valid log. LSNs are 1-based
and gap-free; the counter resumes at `max(seen)+1`.

### Failure handling

If the frame buffer or fsync fails, `fail_batch` acks *every* waiter with the
error and reclaims the LSNs so the log stays gap-free. The writes did not commit,
so callers must not ack their clients — and they won't, since `append` returned
`Err`.

---

## 3. The engine (`engine/mod.rs`) — where checkpoint & recovery integrate

### Split ownership

Per collection: the `Reader` + an `AtomicU64` ordinal allocator go in the
`Catalog` (read path); the `Writer` goes into the `IndexApplier` on the WAL
thread. The catalog is an `Arc<HashMap>`, read-only after open, cloned into the
read path — it never holds a writer.

### Write path (`insert`)

1. Validate `dim` **before logging** — a record that can't apply must never reach
   the durable WAL, or it would fail apply forever on every replay.
2. `alloc_ordinal()` — CAS loop on the allocator, assigning the ordinal *once*,
   on the logging side.
3. `wal.append(Record::Insert { collection, ordinal, vector })` — blocks until
   durable **and** applied.

The record is **positional** ("write vector at ordinal N"), never "append next."
That's the idempotency key: `IndexApplier::apply` calls `write_at`/`delete` (both
no-ops on re-application), so replaying an already-applied record changes
nothing.

**The phantom-ordinal subtlety:** if the WAL append fails, the allocator already
burned that ordinal, leaving a zero-filled slot that a *later* successful insert
would pull into search range (surfacing with score 0). The fix: tombstone it in
memory via `Reader::tombstone_uncommitted` — a pure atomic bit-flip, *not* a WAL
`Delete` (nothing durable to replay). The known residual hole (transient failure
→ success → crash before checkpoint) is documented in-code as deferred.

### Checkpoint (`IndexApplier::checkpoint`)

The flusher thread pokes the WAL thread on a timer; the WAL thread runs
checkpoint *itself* (it owns the writers). Per collection:
`sync_data → stage_watermark → sync_header`, then the commit thread truncates the
WAL — strictly after. Ordering rule: **msync the index before truncating the
WAL.** Truncate-first-then-crash would throw away records the index hadn't
durably absorbed = data loss.

The truncation point is the **minimum** durable watermark across collections, not
the max. A frame at LSN L is redundant only once *every* collection that might
own it is durable past L.

### Recovery (`Db::open`)

Each `Writer::checkpoint_lsn()` says which prefix is already folded into that
index. `skip_through` is the **min** across collections — anything ≤ the slowest
collection's watermark is durable everywhere, so skipping it is always safe;
frames above it get replayed and idempotent apply absorbs any a faster collection
already saw. The code is emphatic: **never use `max`** — that would skip frames a
lagging collection still needs = silent loss. Recovery runs on the caller's
thread *before* the commit thread starts, so the index is caught up before any
new write can race it. Then allocators are re-seeded from the post-recovery
high-water mark.

---

## What is and isn't guaranteed

| Guaranteed | How |
|---|---|
| **Durability** — `insert` returns `Ok` ⟹ data survives crash | fsync before ack; WAL is source of truth |
| **Read-your-writes** | apply on the same thread as commit, post-fsync |
| **Crash consistency** | torn WAL tail dropped via CRC; double-buffered index header |
| **Lock-free parallel reads** | SWMR via Release/Acquire on `count`; readers never serialize |
| **Idempotent recovery** | positional records + `write_at`/`delete` no-ops |
| **Gap-free monotonic LSNs** | assigned only by the single commit thread; reclaimed on failure |

**Not** guaranteed:

- Real power-loss durability on macOS (`sync_data` ≠ `F_FULLFSYNC`; noted in
  `commit_batch`).
- The transient-failure phantom-ordinal hole (documented as deferred in
  `Db::insert`).
- Per-collection recovery precision — currently the conservative global-min; a
  real per-frame watermark is the documented future fix.
- Anything in `CLAUDE.md` §2 non-goals (deletes-with-compaction, ANN, metadata
  filtering are v2).

The one mental model to keep: **a single thread owns the WAL file, the index
`Writer`s, apply, and checkpoint.** Readers run lock-free off `count`. Every
durability and concurrency property falls out of that.
