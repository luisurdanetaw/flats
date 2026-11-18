# Flats

A tiny, embeddable vector database. Exact (brute-force) similarity search over
`mmap`'d vectors, a write-ahead log for durability, metadata filters through
roaring bitmaps, and V-SQL — a small SQL dialect — to drive it. A library, not
a server: open a directory and query it from the same process. CPU only.

- **Exact results.** Every search scans every live vector with a SIMD dot
  product (AVX2 or NEON, with a scalar fallback). Recall is 1.0.
- **Parameterized queries.** Values go in as `?` parameters, never as SQL
  text, so injection is impossible, and query vectors travel as binary.
- **Filters that keep `k`.** `WHERE` runs *before* ranking, so `TOP 10 … WHERE
  lang = 'en'` returns ten English rows whenever ten exist.
- **Durable.** A write is fsynced to the WAL before it is acknowledged; a
  multi-row statement is one record, so it lands whole or not at all.
- **Writes are visible at once.** No index build: a row is searchable the moment
  its insert returns.
- **Concurrent reads.** One writer, lock-free parallel readers, per-statement
  snapshots. Deleted rows are reclaimed by background compaction.
- **Linux, x86_64 and aarch64** for v1.

## Documentation

- **[User guide](docs/guide.md)**: the model, V-SQL, loading data, sizing,
  backups, performance, errors.
- SDK guides: [Python](sdk/python/README.md) · [Node.js](sdk/node/README.md) ·
  [C](sdk/c/README.md). Rust: the crate docs (`cargo doc --open`) and below.
- [Architecture](ARCHITECTURE.md): how it works and what is guaranteed.

## Quickstart

### Rust

```rust
use flats::{Db, DbOptions, Param};

let db = Db::open("./my-db", &[], DbOptions::default())?;
db.execute(
    "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), year INT)
     WITH (capacity = 10000);",
)?;
db.execute(
    "INSERT INTO docs (id, vector, year) VALUES
         ('a', [0.1, 0.9, 0.0], 2023),
         ('b', [0.8, 0.1, 1e-5], 2024);",
)?;
// Values go in as `?` parameters, never pasted into the text.
for row in db.execute_with(
    "SEARCH TOP 5 NEAREST TO ? FROM docs WHERE year >= ? RETURNING id, score;",
    &[vec![1.0f32, 0.0, 0.0].into(), Param::Int(2024)],
)? {
    println!("{:?}", row?.0);
}
db.close()?;
```

For bulk loads, `Db::insert_many(collection, rows)` takes vectors as `f32`
slices and commits a whole batch as one record. A runnable version of this
example is the crate-level doc test (`cargo test --doc`).

### Python

```python
import flats

with flats.Database("./my-db") as db:
    db.execute("CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3)) "
               "WITH (capacity = 10000);")
    db.insert("docs", [{"id": "a", "vector": [0.1, 0.9, 0.0]}])
    hits = db.search("docs", [1, 0, 0], k=5, where="id != ?", params=["b"],
                     returning=["id", "score"])
    print(hits.dicts())
```

ctypes over the C ABI, no third-party packages. See [`sdk/python`](sdk/python).

### Node.js

```js
const flats = require('flats');

const db = new flats.Database('./my-db');
db.execute('CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3)) WITH (capacity = 10000);');
db.insert('docs', [{ id: 'a', vector: [0.1, 0.9, 0.0] }]);
console.log(db.search('docs', [1, 0, 0], { k: 5, returning: ['id', 'score'] }).toObjects());
db.close();
```

An N-API addon over the C ABI, no npm dependencies. See [`sdk/node`](sdk/node).

### C

```c
#include "flats.h"

flats_db *db;
flats_open("./my-db", &db);
flats_execute(db, "SELECT id FROM docs;", &res);
```

The header [`sdk/c/include/flats.h`](sdk/c/include/flats.h) is the contract
every binding wraps. See [`sdk/c`](sdk/c).

## V-SQL

```sql
CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(768), author TEXT, year INT)
    WITH (capacity = 1000000);

INSERT INTO docs (id, vector, author, year) VALUES ('d1', [0.1, ...], 'alice', 2024), (...);

SEARCH TOP 10 NEAREST TO [...] FROM docs WHERE year > 2020 AND author != 'bob'
    RETURNING id, score, author;

SELECT id, author FROM docs WHERE year >= 2024;   -- `*` never includes the vector
UPDATE docs SET author = 'carol' WHERE id = 'd1';
DELETE FROM docs WHERE year < 2000;
DROP COLLECTION docs;
```

Types: `VECTOR(dim)`, `TEXT`, `INT`, `FLOAT`. Operators: `= != < <= > >=`,
`AND`, `OR`. Every collection declares one `PRIMARY KEY` (TEXT or INT). The
full grammar and its rationale are in [`CLAUDE.md`](CLAUDE.md).

## Operations

- **Backup:** `Db::backup(dir)` (or `flats_backup`, `db.backup()`) writes a
  consistent copy — whole statements only — while reads continue. Restore by
  opening the copy.
- **Compaction** runs in the background and reclaims deleted rows; ordinals
  are renumbered, primary keys are not.
- **Crash recovery** replays the WAL tail on open. See
  [`ARCHITECTURE.md`](ARCHITECTURE.md) for what is and is not guaranteed.

## Performance

Flat search costs one pass over every live vector, so it is bounded by memory
bandwidth. Measured on a Ryzen 7 8700F under WSL2 (7 GB VM), 768-dim `f32`,
top-10, release build:

| | 1M vectors |
|---|---|
| Search, one thread | p50 ~105 ms |
| Search, filter matching 10% | p50 ~27 ms |
| Bulk insert (`insert_many`) | ~57k rows/s |
| Reopen | 0.3 s |

Approximate (ANN) indexes are 20–100× faster at this size and give up exact
recall. Flats is built for collections up to a few hundred thousand vectors,
for exact filtered results, and for data that changes constantly.

## Building and testing

```sh
cargo test                           # the engine
cargo test --workspace               # + the C ABI (compiles and runs a C program)
cargo build --release -p flats-c -p flats-node
python3 -m unittest discover -s sdk/python/tests -t sdk/python
node --test sdk/node/test/
```
