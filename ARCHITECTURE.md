# ARCHITECTURE.md — How the gears move

This document explains the inner workings of the core modules — the flat index
(`src/index/index.rs`), the write-ahead log (`src/wal/wal.rs`), the metadata
layer (`src/metadata/`), and the engine that wires them together
(`src/engine/mod.rs`). SIMD is out of scope here; see `CLAUDE.md` §8.

For *scope, conventions, and on-disk byte layouts*, read `CLAUDE.md`. This file is
about the **runtime** — who owns what, which thread mutates which bytes, and why
the durability and concurrency properties hold.

## The shape of it

Per collection there are **three stores**, all keyed by the same ordinal:

- **`index/index.rs`** — the flat vector store. *Ordinal → vector.* One writer,
  many readers, no lock.
- **`metadata/index.rs`** — the metadata index. *(Column, value) → bitmap of
  ordinals* — answers WHERE.
- **`metadata/tuples.rs`** — the tuple store. *Ordinal → row values* — answers
  RETURNING.

Around them, two singletons:

- **`wal.rs`** — the durable log. One thread owns the file and *is* the writer.
- **`engine/mod.rs`** — wires everything together, owns the durability contract
  end-to-end, and keeps the **catalog** (which collections exist) durable in
  `catalog.snap`.

The whole design collapses to a single sentence:

> **Each collection's writers (all three) live on the WAL commit thread, and
> nothing else ever mutates an index.**

Everything below is a consequence of that.

On disk:

```text
<db dir>/
├── catalog.snap             which collections exist (id, dim, capacity, schema)
├── wal.log                  ONE log for all collections, records interleaved
└── collection-{id}/
    ├── vectors.idx          flat index (mmap)
    ├── metadata.snap        metadata index snapshot
    └── tuples.snap          tuple store snapshot
```

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

Per collection: the three **read handles** (flat `Reader`, metadata `Reader`,
tuple `Reader`) + the config + an `AtomicU64` ordinal allocator go in the
`Catalog` (read path); the three **writers** travel together as
`CollectionWriters` inside the `IndexApplier` on the WAL thread. The catalog is
an `Arc<HashMap>`, read-only after open, cloned into the read path — it never
holds a writer.

### Write path (`insert`)

1. Validate `dim` **and the metadata row** (schema, types, NaN) **before
   logging** — a record that can't apply must never reach the durable WAL, or it
   would fail apply forever on every replay.
2. `alloc_ordinal()` — CAS loop on the allocator, assigning the ordinal *once*,
   on the logging side.
3. `wal.append(Record::Insert { collection, ordinal, vector, metadata })` —
   ONE record carries both the vector and the row; blocks until durable **and**
   applied to all three stores.

The record is **positional** ("write vector at ordinal N"), never "append next."
That's the idempotency key: `IndexApplier::apply` fans the record out to all
three writers (`write_at` / `insert_row` / `write_row`, all no-ops on
re-application), so replaying an already-applied record changes nothing.
Watermarks advance **last**, only after every store's write succeeded.

**The phantom-ordinal subtlety:** if the WAL append fails, the allocator already
burned that ordinal, leaving a zero-filled slot that a *later* successful insert
would pull into search range (surfacing with score 0). The fix: tombstone it in
memory via `Reader::tombstone_uncommitted` — a pure atomic bit-flip, *not* a WAL
`Delete` (nothing durable to replay). The known residual hole (transient failure
→ success → crash before checkpoint) is documented in-code as deferred.

### Checkpoint (`IndexApplier::checkpoint`)

The flusher thread pokes the WAL thread on a timer; the WAL thread runs
checkpoint *itself* (it owns the writers). Per collection, all three stores:
the flat index does `sync_data → stage_watermark → sync_header`, the two
metadata stores each write their serialize-and-rename snapshot. Then the commit
thread truncates the WAL — strictly after. Ordering rule: **make the stores
durable before truncating the WAL.** Truncate-first-then-crash would throw away
records a store hadn't durably absorbed = data loss.

The truncation point is the **minimum** durable watermark across every store of
every collection, not the max. A frame at LSN L is redundant only once *every*
store that might own it is durable past L. If any one store's checkpoint fails,
the whole round errs and truncation is skipped — retried next tick.

### Recovery (`Db::open`)

Each store's persisted watermark says which prefix is already folded into it.
`skip_through` is the **min** across all stores of all collections — anything ≤
the slowest store's watermark is durable everywhere, so skipping it is always
safe; frames above it get replayed and idempotent apply absorbs any a faster
store already saw. The code is emphatic: **never use `max`** — that would skip
frames a lagging store still needs = silent loss. Recovery runs on the caller's
thread *before* the commit thread starts, so the stores are caught up before any
new write can race them. Then allocators are re-seeded from the post-recovery
high-water mark.

---

## 4. The metadata layer (`metadata/`) — two inverse access paths

### The mental map

The metadata index and the tuple store answer opposite questions about the same
rows, keyed by the same ordinal the flat index assigned:

| | question | shape | serves |
|---|---|---|---|
| **metadata index** (4a) | *which ordinals match?* | `(column, value) → RoaringBitmap` | `WHERE` |
| **tuple store** (4b) | *what values at ordinal N?* | `ordinal → Vec<Value>` | `RETURNING` |

Neither can cheaply answer the other's question — a posting list knows
membership, not the row; a row knows values, not the set. They never talk to
each other; the executor (a later phase) composes them:

```text
SEARCH TOP 5 NEAREST TO [q] WHERE a < 3 RETURNING title

1. metadata index   lookup_range(a, Lt, 3)   → bitmap {0,1,5}    WHO matches
2. flat index       dot-product scan ∩ bitmap → top-k ordinals    WHO is nearest
3. tuple store      get(ordinal, [title])     → values            WHAT to return
```

### Durability: the WAL is the database; the stores are caches of it

Neither store is durable by itself — both are plain in-memory structures behind
a mutex. Durability comes from one place: the insert's `Record::Insert` carries
the metadata row, and `apply` (post-fsync, on the WAL thread) fans it out to all
three stores. Both stores are **rebuildable projections of the WAL**: wipe
either one and replay reconstructs it.

Rebuildable *as long as the WAL still has the records* — and the WAL doesn't
keep them forever. Checkpoint (on the same WAL thread) makes all three stores
durable first, and only **then** truncates the log, up to the **minimum of the
watermarks it just persisted** (§3). That ordering is the whole contract: a
record leaves the WAL only after every store that needs it has it durably on
disk, so replay can always cover the gap between any store's snapshot and the
log's tail.

Snapshots (`metadata.snap`, `tuples.snap`) exist only to bound replay time.
Each is written by checkpoint as a whole-file serialize-and-rename
(tmp → fsync → atomic rename → dir fsync), stamped with the `last_lsn` it
reflects — a freshness stamp. On open: load snapshot, replay everything above
the min watermark.

Two properties make the min-watermark scheme sound:

- **Idempotent applies** — re-delivering a record a store already has is a
  no-op (bitmap set-insert, slot overwrite-with-same).
- **Corrupt/missing snapshot ⇒ empty at LSN 0** — the store's 0 drags the
  global min down, forcing a full-tail replay that heals it while the healthy
  stores absorb the replay harmlessly. (Only crash-produced tears are covered:
  atomic rename means a torn checkpoint leaves the *old complete* snapshot,
  whose watermark bounds truncation. Bit-rot after the WAL truncated past the
  data is unrecoverable — accepted.)

Same SWMR posture as the flat index: each store is a `!Clone` `Writer` (lives
in the applier) + cloneable `Reader`s, though internally it's a plain Mutex for
now — real lock-free SWMR is deferred until profiling demands it.

Two details worth remembering: **deletes are lazy** in the metadata index (only
the `live` bitmap is cleared; every lookup masks with `& live`, postings are
never scrubbed), and **NaN is rejected at the boundary** (`validate_row`) so
float keys can hold a total order.

---

## 5. The catalog — who exists

The catalog answers one bootstrap question: *which collections exist, and what
shape are they?* Without it, `Db::open` wouldn't know which `collection-N/`
dirs to open or which schema to validate inserts against.

- **On disk**: `catalog.snap` in the root — the list of `CollectionConfig`s
  (id, dim, capacity, schema). Written with the same atomic-rename dance as
  every other snapshot.
- **In memory**: the `Catalog` map built at open — read handles + config +
  allocator per collection (writers go to the applier, as always).

`Db::open(dir, configs, opts)` treats `configs` as *"ensure these exist"*: a
new id is registered and the file rewritten; a known id must match its
persisted config exactly (`CollectionConfigMismatch` otherwise); and
`open(dir, &[], opts)` brings everything back with no configs at all.

**The catalog is the one durable thing the WAL does NOT protect.** Registration
happens at open time, before the WAL thread starts, so there's no acked-but-
unpersisted window — the atomic rename alone is the guarantee. The consequence:
a corrupt `catalog.snap` is a **loud error**, never an empty fallback. The
stores may fall back to empty because replay rebuilds them; nothing can rebuild
the catalog, and an empty fallback would silently hide every collection. This
asymmetry disappears when CREATE COLLECTION becomes a WAL record (next phase).

---

## What is and isn't guaranteed

| Guaranteed | How |
|---|---|
| **Durability** — `insert` returns `Ok` ⟹ vector AND metadata survive crash | one record, fsync before ack; WAL is source of truth |
| **Read-your-writes** | apply on the same thread as commit, post-fsync |
| **Crash consistency** | torn WAL tail dropped via CRC; double-buffered index header; CRC'd serialize-and-rename snapshots |
| **Three-store convergence after any crash** | idempotent applies + min-watermark replay (a wiped store self-heals) |
| **Lock-free parallel vector reads** | SWMR via Release/Acquire on `count`; readers never serialize |
| **Idempotent recovery** | positional records; `write_at`/`insert_row`/`write_row`/deletes all no-op on re-application |
| **Gap-free monotonic LSNs** | assigned only by the single commit thread; reclaimed on failure |
| **Catalog survives restarts** | `catalog.snap`, atomic-rename; conflicts refused loudly |

**Not** guaranteed:

- Real power-loss durability on macOS (`sync_data` ≠ `F_FULLFSYNC`; noted in
  `commit_batch`).
- The transient-failure phantom-ordinal hole (documented as deferred in
  `Db::insert`).
- Per-collection recovery precision — currently the conservative global-min; a
  real per-frame watermark is the documented future fix.
- Snapshot bit-rot *after* the WAL truncated past its data (outside the crash
  model; see §4).
- Catalog rebuild from the WAL — registration isn't logged yet; corrupt
  `catalog.snap` is a refusal, not a recovery (see §5).
- Anything in `CLAUDE.md` §2 non-goals (the SEARCH..WHERE executor, ANN,
  compaction are later phases).

The one mental model to keep: **the WAL is the database. A single thread owns
the log file, every store's writer, apply, and checkpoint; the three stores per
collection are rebuildable projections of the log, each with a freshness stamp
(`applied_lsn`), and snapshots just let them warm up faster.** Readers run
lock-free off `count` (vectors) or a mutex (metadata). Every durability and
concurrency property falls out of that.
