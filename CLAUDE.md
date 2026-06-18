# CLAUDE.md — Flats: a tiny vector DB

A minimalist, cheap, embeddable vector database. No GPU. No third-party runtime crates. Pure Rust, `std` only.

This file is the source of truth for design decisions, scope, and conventions. Read it before making changes.
 
---

## 1. Project Goals

- **Tiny footprint** — zero third-party runtime dependencies for v1. `std` only. Platform syscalls via hand-rolled `extern "C"` blocks where needed (mmap, CreateFileMapping).
- **Cross-platform** — Linux, macOS, Windows. Both `x86_64` and `aarch64`.
- **Durable** — WAL-first writes. `fsync` on every batch drain.
- **Fast reads** — `mmap`'d flat vector index + hand-tuned SIMD dot product.
- **Embeddable** — single-process Rust library. Not a server. Not a daemon.
## 2. Non-Goals (v1)

- ❌ **No GPU.** No CUDA, Metal, ROCm, Vulkan. CPU only.
- ❌ **No third-party crates.** Including no `crossbeam`, no `roaring`, no `memmap2`, no `libc`, no `windows-sys`. We declare FFI ourselves.
- ❌ **No ANN index** (HNSW, IVF-PQ, ScaNN). Flat brute-force only.
- ❌ **No clustering, sharding, replication, network protocol.**
- ❌ **No async runtime.** Plain threads + `std::sync::mpsc`.
- ❌ **No metadata / filtering.** That's v2.
- ❌ **No deletes.** Append-only in v1. Tombstones come with metadata in v2.
> If a feature isn't on the v1 milestone list below, it doesn't belong in v1. Push back on scope creep — that's the whole point of this project.
 
---

## 3. Supported Platforms

| OS      | Architectures      | SIMD path           |
| ------- | ------------------ | ------------------- |
| Linux   | x86_64, aarch64    | AVX2 / AVX-512 / NEON |
| macOS   | x86_64, aarch64    | AVX2 / NEON         |
| Windows | x86_64, aarch64    | AVX2 / AVX-512 / NEON |

Naive scalar fallback is always available and tested.
 
---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Public API:   db.insert(vec) -> LSN                         │
│                db.search(query, k) -> Vec<(id, score)>       │
│                db.open(path) / db.flush() / db.close()       │
└────────────────┬─────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │ Writes          │ Reads
        ▼                 ▼
┌─────────────────┐   ┌───────────────────────────────┐
│ std::sync::mpsc │   │ mmap'd vectors.dat            │
│ (unbounded)     │   │     │                         │
└────────┬────────┘   │     ▼                         │
         │            │ SIMD dot kernel (AVX2/NEON)   │
         ▼            │     │                         │
┌─────────────────┐   │     ▼                         │
│ Flusher thread  │   │ top-k heap                    │
│  1. drain batch │   └───────────────────────────────┘
│  2. assign LSNs │
│  3. append WAL  │
│  4. fsync(WAL)  │
│  5. append .dat │
│  6. fsync(.dat) │
│  7. ack waiters │
└─────────────────┘
```

Two threads is the whole concurrency story for v1: caller threads on one side, single flusher on the other. Readers go directly against the mmap.
 
---

## 5. Repository Layout

```
.
├── CLAUDE.md              ← this file
├── Cargo.toml             ← no [dependencies] section. enforce in CI.
├── src/
│   ├── lib.rs             ← public API, re-exports
│   ├── db.rs              ← Db struct, open/close/insert/search
│   ├── wal.rs             ← WAL writer, recovery
│   ├── index.rs           ← vectors.dat layout, mmap wrapper
│   ├── flusher.rs         ← drain loop, LSN allocator
│   ├── simd/
│   │   ├── mod.rs         ← runtime dispatch
│   │   ├── scalar.rs      ← naive fallback (always compiled)
│   │   ├── avx2.rs        ← x86_64
│   │   ├── avx512.rs      ← x86_64 (optional)
│   │   └── neon.rs        ← aarch64
│   ├── platform/
│   │   ├── mod.rs         ← cfg-gated re-exports
│   │   ├── unix.rs        ← mmap, munmap, fsync FFI
│   │   └── windows.rs     ← CreateFileMapping, MapViewOfFile, FlushFileBuffers
│   └── util/
│       ├── crc32.rs       ← table-driven CRC32 (~30 lines, no deps)
│       └── bytes.rs       ← LE encode/decode helpers
├── tests/
│   ├── crash_recovery.rs
│   ├── simd_parity.rs     ← scalar vs each SIMD path must match bit-for-bit
│   └── e2e.rs
└── benches/               ← criterion-style raw harness, no `criterion` dep
    └── simd.rs
```
 
---

## 6. File Formats

### 6.1 `wal.log` (append-only, no mmap)

**Why no mmap on WAL:** mmap'd writes don't give us deterministic `fsync` semantics. Page writeback ordering is at the OS's mercy. For durability we need explicit `write()` + `fsync()`.

**Header (32 bytes, written once at file create):**

```
offset  size  field
0       4     magic        "WAL0"
4       4     version      u32 LE
8       4     flags        u32 LE (reserved)
12      4     header_crc   u32 LE  (CRC32 of bytes 0..12)
16      16    reserved
```

**Record (variable length):**

```
offset  size       field
0       8          lsn          u64 LE, monotonic
8       4          payload_len  u32 LE
12      4          payload_crc  u32 LE  (CRC32 of payload bytes)
16      1          op           u8: 0x01 = INSERT
17      payload_len-1   op-specific payload
```

**INSERT payload:**

```
0       4          dim          u32 LE
4       dim*4      vector       f32 LE (little-endian on disk, even on BE hosts; v1 is LE-only)
```

**Recovery:** read header → loop reading records → verify CRC → stop at first bad CRC or EOF. The last good LSN is the high-water mark. Any records in `vectors.dat` with LSN > high-water are discarded (truncate the .dat back to the good record boundary).

### 6.2 `vectors.dat` (mmap'd, append-only)

**Header (4 KiB, page-aligned):**

```
offset  size  field
0       4     magic        "VEC0"
4       4     version      u32 LE
8       4     dim          u32 LE        (fixed at file creation)
12      4     flags        u32 LE
16      8     count        u64 LE        (number of valid records)
24      8     last_lsn     u64 LE        (high-water mark)
32      4     header_crc   u32 LE        (CRC32 of bytes 0..32)
36      4060  reserved (zero-filled)
```

**Records (contiguous, starting at offset 4096):**

```
record[i]:
  offset  size       field
  0       8          lsn          u64 LE
  8       dim*4      vector       f32 LE
 
record_size = 8 + dim * 4
record[i] starts at byte offset 4096 + i * record_size
```

The header is updated **after** each batch flush, and is the last thing written before `fsync`. This makes it the commit point: a torn header = pre-batch state on recovery (since CRC won't match).
 
---

## 7. WAL + Flusher Protocol

### 7.1 Insert path (caller thread)

```rust
// pseudo
let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
self.write_tx.send(WriteRequest::Insert { vec, ack: ack_tx })?;
let lsn = ack_rx.recv()??;
Ok(lsn)
```

Calls block until durable. v1 keeps the API synchronous and obvious. Async wrappers (futures, batch APIs) are v2.

### 7.2 Flusher loop

```text
loop:
    msg = write_rx.recv()              // block for first message
    batch = [msg]
    while let Ok(m) = write_rx.try_recv():   // opportunistically drain
        batch.push(m)
        if batch.len() >= MAX_BATCH: break
 
    for req in &batch:
        req.lsn = next_lsn(); next_lsn += 1
 
    // 1. WAL
    for req in &batch:
        wal_file.write_all(&encode(req))
    wal_file.sync_all()                // fsync #1
 
    // 2. Index
    for req in &batch:
        let off = 4096 + req.id * record_size
        mmap[off .. off+record_size].copy_from(&record_bytes)
    update_header(count, last_lsn)
    index_file.sync_all()              // fsync #2 (also flushes mmap'd pages on most OSes via msync semantics — see §9)
 
    // 3. Ack
    for req in batch:
        req.ack_tx.send(Ok(req.lsn))
```

**LSN is monotonic and gap-free.** Assigned only by the flusher under no contention. Crash recovery uses LSNs to reconcile WAL vs index.

**Batch limit:** start with `MAX_BATCH = 1024` or so. Tunable. Don't add a timer in v1 — the queue's `recv()` blocks until there's work, then `try_recv()` opportunistically drains whatever piled up while we were busy. That's enough.
 
---

## 8. SIMD Kernel

### 8.1 What we're computing

Dot product of two `f32` slices of length `dim`. That's it. Cosine similarity = caller normalizes inputs.

```rust
pub fn dot(a: &[f32], b: &[f32]) -> f32;
```

### 8.2 Dispatch strategy

Runtime CPU detection on first call (cached in an `AtomicU8`). No build-time feature flags — one binary runs everywhere.

```text
x86_64:
    if is_x86_feature_detected!("avx512f")  → avx512 path
    elif is_x86_feature_detected!("avx2")   → avx2 path
    else                                     → scalar
aarch64:
    NEON is mandatory on aarch64-*-* targets → neon path
else:
    scalar
```

`#[target_feature(enable = "avx2")]` on the AVX2 function, `unsafe` wrapper that's only callable after the detection check.

### 8.3 Correctness contract

The SIMD parity test (`tests/simd_parity.rs`) runs every implementation on the same inputs and asserts results match **bit-for-bit with the scalar reference**. If they don't, the SIMD path is wrong — fix it, don't relax the test.

Note: f32 sums are not associative, so a SIMD reduction with 8 lanes can legitimately differ from a scalar sum in the last ULP. Handle this by making the scalar reference reduce in the same lane-grouped order, **not** by adding a tolerance. We want bit-exact.

### 8.4 Numerical care

- AVX2: 8 lanes of f32, use `_mm256_fmadd_ps` if FMA is detected (it almost always is alongside AVX2; check `is_x86_feature_detected!("fma")`).
- Tail handling: process `dim - (dim % LANES)` with SIMD, then scalar for the remainder. Do not assume `dim` is a multiple of 8/16.
- Alignment: the vectors come from mmap'd memory at offsets that **are** 4-byte aligned but **not** guaranteed 32-byte aligned. Use unaligned loads (`_mm256_loadu_ps`). The performance delta vs aligned is negligible on modern CPUs and the alternative is painful.
---

## 9. Platform Layer (zero-dep mmap & fsync)

Hand-rolled FFI. No `libc`, no `windows-sys`.

### 9.1 Unix (`src/platform/unix.rs`)

```rust
#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut core::ffi::c_void, len: usize, prot: i32,
            flags: i32, fd: i32, offset: i64) -> *mut core::ffi::c_void;
    fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut core::ffi::c_void, len: usize, flags: i32) -> i32;
    fn ftruncate(fd: i32, length: i64) -> i32;
}
// PROT_READ=1, PROT_WRITE=2, MAP_SHARED=1, MAP_FAILED=-1isize as *mut _
// MS_SYNC=4
```

Use `std::os::unix::io::AsRawFd` to get the `fd` from a `std::fs::File`.

For `fsync` use `File::sync_all()` — std handles the per-platform call.

### 9.2 Windows (`src/platform/windows.rs`)

```rust
#[cfg(windows)]
extern "system" {
    fn CreateFileMappingW(hFile: isize, lpAttrs: *mut core::ffi::c_void,
                          flProtect: u32, dwMaximumSizeHigh: u32,
                          dwMaximumSizeLow: u32, lpName: *const u16) -> isize;
    fn MapViewOfFile(hMapping: isize, dwAccess: u32,
                     dwOffsetHigh: u32, dwOffsetLow: u32,
                     dwNumberOfBytesToMap: usize) -> *mut core::ffi::c_void;
    fn UnmapViewOfFile(lpBaseAddress: *const core::ffi::c_void) -> i32;
    fn FlushViewOfFile(lpBaseAddress: *const core::ffi::c_void, dwNumberOfBytesToFlush: usize) -> i32;
    fn CloseHandle(hObject: isize) -> i32;
}
```

Use `std::os::windows::io::AsRawHandle` for the `HANDLE`.

### 9.3 Sync ordering rules (read carefully)

mmap durability is subtle. The rules for our index file:

1. `mmap` is `MAP_SHARED` so writes are visible to the file.
2. Before we consider a batch durable, we **must** call:
    - Unix: `msync(MS_SYNC)` then `File::sync_all()`.
    - Windows: `FlushViewOfFile` then `FlushFileBuffers` (via `File::sync_all()`).
3. `msync` / `FlushViewOfFile` alone is **not** sufficient — they push dirty pages to the writeback queue but don't guarantee the disk has them.
4. The WAL is the authoritative durability log. The index is rebuildable from it. If you ever doubt index integrity, replay the WAL.
---

## 10. Public API Sketch

```rust
pub struct Db { /* … */ }
 
#[derive(Debug)]
pub struct OpenOptions {
    pub dim: u32,
    pub create_if_missing: bool,
    pub max_batch: usize,
}
 
impl Db {
    pub fn open(path: impl AsRef<Path>, opts: OpenOptions) -> Result<Self>;
    pub fn insert(&self, vec: &[f32]) -> Result<u64>;       // returns LSN
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>>;
    pub fn flush(&self) -> Result<()>;                       // forces a drain
    pub fn close(self) -> Result<()>;                        // graceful shutdown
}
```

`id` for a vector = its record index in `vectors.dat` (which equals the order it was inserted). Returned by `search` alongside the score.

**Search implementation (v1):** brute-force scan all `count` records, compute dot products via SIMD kernel, maintain a top-k binary heap. No prefetching tricks in v1 — measure first.
 
---

## 11. Coding Conventions

- **No `unwrap()` in library code.** Bubble errors with `Result<T, Error>`. Tests can `unwrap`.
- **One error type** (`crate::Error`) with variants. No `anyhow`, no `thiserror`. Hand-roll `Display` and `Error`.
- **`unsafe` is allowed but quarantined.** Only in `simd/*` and `platform/*`. Every `unsafe` block has a `// SAFETY:` comment naming the preconditions.
- **No `Box<dyn Error>` in public API.**
- **`#![forbid(unsafe_op_in_unsafe_fn)]`** at crate root. Be explicit about unsafe scopes.
- **`#![warn(missing_docs)]`** on public API.
- **Little-endian on disk, always.** Use explicit `to_le_bytes` / `from_le_bytes`. Do not rely on host order.
- **CI enforces zero-dep:** a check that greps `Cargo.toml` for any line under `[dependencies]` or `[dev-dependencies]` other than `[dev-dependencies]` entries we allow (none in v1).
---

## 12. Milestones

> **NOTE:** The first milestone is the SIMD kernel (with naive fallback if the architecture doesn't support SIMD). Get the math right and benchmarked before touching any I/O.

### Milestone 1 — SIMD Kernel ★ start here

- [ ] `src/simd/scalar.rs` — naive `dot(&[f32], &[f32]) -> f32` reference.
- [ ] `src/simd/avx2.rs` — AVX2 + FMA path behind `#[target_feature]`.
- [ ] `src/simd/neon.rs` — NEON path.
- [ ] `src/simd/avx512.rs` — optional, only if free time.
- [ ] `src/simd/mod.rs` — runtime dispatch with cached detection.
- [ ] `tests/simd_parity.rs` — bit-exact match between scalar and each enabled SIMD path across `dim ∈ {1, 7, 8, 9, 15, 16, 31, 32, 127, 128, 768, 1536}`.
- [ ] `benches/simd.rs` — measure GB/s per path. Establish baseline.
  **Done when:** parity test green on every supported platform in CI, and AVX2 path is at least 4× faster than scalar on dim=768.

### Milestone 2 — In-memory flat index + brute-force search

- [ ] `Db` struct holding `Vec<f32>` (flat, row-major) + `dim` + `count`.
- [ ] `insert`/`search` ignoring durability entirely.
- [ ] Top-k via `BinaryHeap`.
- [ ] Tests for correctness (known small vectors).
### Milestone 3 — `vectors.dat` with mmap

- [ ] Platform layer (`platform/unix.rs`, `platform/windows.rs`).
- [ ] File creation with header, `ftruncate` to capacity (grow in chunks of e.g. 1 MiB to avoid per-insert truncate).
- [ ] mmap on open, remap on grow.
- [ ] `search` reads from mmap'd region.
- [ ] Tests on all three OSes (CI matrix).
### Milestone 4 — WAL + flusher + recovery

- [ ] CRC32 implementation in `util/crc32.rs`.
- [ ] WAL writer (append-only file).
- [ ] Flusher thread reading from `mpsc::Receiver`.
- [ ] Crash-recovery on `Db::open`: scan WAL, find last good LSN, truncate `vectors.dat` to match, replay missing records.
- [ ] `tests/crash_recovery.rs` — kill process mid-write, reopen, assert state.
### Milestone 5 — End-to-end polish

- [ ] Graceful shutdown (`Db::close` drains queue, joins flusher).
- [ ] Error types finalized.
- [ ] README with examples.
- [ ] CI matrix: linux/macos/windows × x86_64/aarch64 where available.
---

## 13. Testing Strategy

- **Unit tests** colocated with modules.
- **`simd_parity.rs`** — non-negotiable, bit-exact across paths.
- **`crash_recovery.rs`** — uses `std::process::Command` to fork a child that writes and `abort()`s mid-batch. Parent reopens, verifies state.
- **Property tests** (hand-rolled, no `proptest`/`quickcheck`): random vectors, random `dim`, random batch sizes. Compare against in-memory reference.
- **No flaky-allowed list.** A flaky test is a bug.
---

## 14. Things We Are Postponing to v2

Do not implement these in v1. If a PR adds them, push back.

- **Metadata bitmap index.** A separate single file (e.g. `meta.idx`) mapping vector IDs → category bitsets. For v1's dense, sequential ID space, a simple `Vec<u64>` bitfield (1 bit per (id, category) pair) is plenty — no Roaring needed unless categories get sparse.
- **Set-operation query pruning.** For filtered queries, intersect the candidate set from the metadata bitmap with the full ID range first, then iterate only those IDs (read the corresponding offsets in `vectors.dat`) and compute dot products only on the survivors. This is where the bitmap pays for itself: turning a full scan into a sparse scan.
- **Deletes** (tombstones in WAL, compaction).
- **ANN index** (HNSW or IVF-PQ over the flat backing store).
- **Async / batched public API.**
- **Snapshot + checkpoint** (truncate WAL after index is durable up to LSN X).
- **Quantization** (int8, binary).
---

## 15. Quick Reference for Contributors

- Editing the WAL or index format? Bump the `version` field and update recovery to refuse unknown versions cleanly.
- Adding a new SIMD path? Add it to `simd_parity.rs` first, then implement.
- Touching `unsafe`? Write the `// SAFETY:` comment before the code.
- Tempted to add a dependency? Read §2 again. The answer is no for v1.