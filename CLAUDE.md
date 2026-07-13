# CLAUDE.md — Flats: a tiny vector DB

A lightweight, cheap, embeddable vector database. No GPU. Minimal third-party runtime crates.

This file is the source of truth for design decisions, scope, and conventions. Read it before making changes.
 
---

## 1. Project Goals

- **Tiny footprint** — zero third-party runtime dependencies for v1. `std` only. Only can be allowed after discussion with code owner (Luis). Platform syscalls via hand-rolled `extern "C"` blocks where needed (mmap, CreateFileMapping).
- **Cross-platform** — Linux, macOS, Windows. Both `x86_64` and `aarch64`. V1 is linux only. 
- **Durable** — WAL-first writes. `fsync` on every batch drain.
- **Fast reads** — `mmap`'d flat vector index + hand-tuned SIMD dot product.
- **Embeddable** — single-process Rust library. Not a server. Not a daemon.

## 2. Non-Goals (v1)

- ❌ **No GPU.** No CUDA, Metal, ROCm, Vulkan. CPU only!.
- ❌ **No third-party crates.** Including no `crossbeam`, no `memmap2`, no `libc`, no `windows-sys`. We declare FFI ourselves. Exceptions need to be discussed with code owner.
- ❌ **No ANN index** (HNSW, IVF-PQ, ScaNN). Flat brute-force only.
- ❌ **No clustering, sharding, replication, network protocol.**
- ❌ **No async runtime.** Plain threads + `std::sync::mpsc`.
> If a feature isn't on the v1 milestone list below, it doesn't belong in v1. Push back on scope creep — that's the whole point of this project.
 
---

## 3. Supported Platforms

| OS      | Architectures      | SIMD path           |
| ------- | ------------------ | ------------------- |
| Linux   | x86_64, aarch64    | AVX2 / AVX-512 / NEON |
| macOS   | x86_64, aarch64    | AVX2 / NEON         |
| Windows | x86_64, aarch64    | AVX2 / AVX-512 / NEON |

Naive scalar fallback is always available and tested.

NOTE: V1 IS LINUX ONLY
 
---

## 4. High-level Architecture Map

```
0. Client layer
1. Query
2. Parse
3. Translate → logical plan → optimize (rule-based)
4. Compile logical plan → bytecode/opcodes (physical execution plan)
5. Engine validates the bytecode, translates into WAL mutation records, split by OP kind:

   ── READ (search) ──────────────────────────
     • Metadata index (roaring bitmap) filtered
     • Vector index searched; blocks pruned
         → dot-product kernel
     • Results returned.  (WAL never touched)

   ── WRITE (insert/delete) ──────────────────
     • Execution VALIDATES + builds the logical mutation
       (dim check, capacity check) → produces a clean record
     • Record → mpsc WAL queue          ◄── intent, post-validation
     • WAL drains, assigns LSN, fsyncs  ◄── COMMIT POINT
     • THEN apply (derived from the one record):
         - Metadata index inserted to
         - Vector index inserted to
     • Ack the write.

   ── WRITE: CREATE COLLECTION ──────────────────
  • Execute VALIDATES (schema well-formed, name available)
  • Build Record::CreateCollection { name, schema, capacity }
  • Record → mpsc WAL queue
  • WAL drains, assigns LSN, fsyncs           ◄── COMMIT POINT
  • Apply (on WAL writer thread):
      - If catalog already has `name`: no-op (idempotent replay)
      - Else: create dir, create FlatIndex files, create MetadataIndex snapshot,
              fsync parent dir, register collection in catalog
  • Ack the write
  • (Subsequent inserts use the normal flusher schedule for index msync)



6. Background: every X seconds a flusher msyncs both indexes,
   truncates the WAL, advances the checkpoint.

V-SQL: SUBSET OF SQL DEDICATED TO FLATS

-- Types: only VECTOR(dim), TEXT, INT, FLOAT... thats it!

-- Vector similarity search queries (core)
SEARCH TOP 5 NEAREST TO [...] FROM docs;                              
SEARCH TOP 5 NEAREST TO [...] FROM docs WHERE z < 2;                  
SEARCH TOP 5 NEAREST TO [...] FROM docs RETURNING id, score; 
SEARCH TOP 5 NEAREST TO [...] FROM docs WHERE z < 2 RETURNING id, score;    

-- Create collection
CREATE COLLECTION docs (
    vector VECTOR(768),
    author TEXT,
    title TEXT,
    published_at INT
) WITH (capacity = 1000000);

-- Regular metadata queries:
SELECT x, y FROM docs; 
SELECT * FROM docs; -- NOTE: DOES NOT RETURN EMBEDDING BY DEFAULT!
SELECT x, y FROM docs WHERE a < 4 AND a = 2; -- NOTE: we will support < > <= >= != =  AND OR and thats it... maybe IN with range/set stuff if easy...


-- INSERT
INSERT INTO docs (vector, author, title, published_at) VALUES ([0.1, ...], 'alice', 'My doc', 1700000000);

-- DELETE
DELETE FROM <collection> WHERE <predicate>;

-- UPDATE
UPDATE <collection> SET <col> = <expr>, <col> = <expr>, ... WHERE <predicate>;


```

Concurrency: single writer, multiple readers. Please refer to ARCHITECTURE.md for a detailed specification.
 
---

## 5. Coding Conventions

- **No `unwrap()` in library code.** Bubble errors with `Result<T, Error>`. Tests can `unwrap`.
- **One error type** (`crate::Error`) with variants. No `anyhow`, no `thiserror`. Hand-roll `Display` and `Error`.
- **`unsafe` is allowed but quarantined.** Only in `simd/*` and `platform/*`. Every `unsafe` block has a `// SAFETY:` comment naming the preconditions.
- **No `Box<dyn Error>` in public API.**
- **`#![forbid(unsafe_op_in_unsafe_fn)]`** at crate root. Be explicit about unsafe scopes.
- **`#![warn(missing_docs)]`** on public API.
- **Little-endian on disk, always.** Use explicit `to_le_bytes` / `from_le_bytes`. Do not rely on host order.
---

## 6. Quick Reference for Contributors

- Editing the WAL or index format? Bump the `version` field and update recovery to refuse unknown versions cleanly.
- Adding a new SIMD path? Add it to `simd_parity.rs` first, then implement.
- Touching `unsafe`? Write the `// SAFETY:` comment before the code.
- Tempted to add a dependency? Read §2 again. The answer is most likely no.