# Flats user guide

This guide covers what every SDK shares: the model, V-SQL, and how to run a
Flats database well. For language specifics see the SDK guides:
[Python](../sdk/python/README.md) · [Node.js](../sdk/node/README.md) ·
[C](../sdk/c/README.md) · [Rust](../README.md#rust).

## The model in two minutes

- **A database is a directory.** Open it from your process; there is no
  server. Only one handle can have a directory open at a time: a second open
  — from another process or the same one — fails with a "locked" error
  rather than risk two writers corrupting the files. The lock is the
  operating system's, so it goes away with the process, crash included.
- **A collection** is a fixed-shape table with exactly one `VECTOR(dim)`
  column, any number of `TEXT`, `INT` and `FLOAT` columns, and exactly one
  `PRIMARY KEY` (`TEXT` or `INT`). Its `capacity` — the most rows it will
  ever hold — is fixed when you create it.
- **Search is exact.** `SEARCH` scores every live row against your query
  vector and returns the best `k`. No index to build, no recall to tune.
- **Score is the dot product.** Higher is more similar. For **cosine
  similarity**, normalize vectors to unit length before inserting them and
  before searching; the dot product of unit vectors *is* the cosine.
- **Every write is durable when the call returns** — it is in the
  write-ahead log and fsynced. A multi-row statement is all-or-nothing.
- **Writes are visible immediately.** A row can be searched the moment its
  insert returns.
- **No NULLs, no defaults.** Every insert gives every column a value.

## A first session

```sql
CREATE COLLECTION articles (
    id        TEXT PRIMARY KEY,
    vector    VECTOR(384),
    lang      TEXT,
    year      INT,
    rating    FLOAT
) WITH (capacity = 500000);

INSERT INTO articles (id, vector, lang, year, rating) VALUES
    ('a1', [0.012, -0.08, ...], 'en', 2023, 4.5),
    ('a2', [0.044,  0.01, ...], 'es', 2024, 3.9);

SEARCH TOP 10 NEAREST TO [0.02, -0.05, ...] FROM articles
    WHERE lang = 'en' AND year >= 2023
    RETURNING id, score, rating;
```

That is V-SQL as written by hand — at a shell, say. From code, never put
values into the text: write `?` and pass them as parameters (see
[Parameters](#parameters-how-values-get-in)), and load rows through your
SDK's `insert` call, which sends vectors as binary and is several times
faster (see [Loading data](#loading-data)).

## V-SQL

V-SQL is a deliberately small SQL. One statement per call, ending in `;`.
Keywords are case-insensitive.

### Types and literals

| Type | Literal | Notes |
|---|---|---|
| `INT` | `42`, `-7` | 64-bit signed |
| `FLOAT` | `0.5`, `-1e-5`, `2.5E+10` | 64-bit; NaN is refused |
| `TEXT` | `'hello'`, `'it''s'` | UTF-8; double a quote to escape it |
| `VECTOR(n)` | `[0.1, -2, 3e-7]` | `n` 32-bit floats; ints are fine |

Scientific notation is accepted everywhere. A value too large for its type
(`1e999`, or a vector component past `3.4e38`) is an error, never a silent
infinity.

### Parameters: how values get in

From a program, **every value goes in as a parameter**. Write `?` where the
value goes and pass the values in order (`execute(sql, params)` in every
SDK, `flats_execute_params` in C):

```sql
SEARCH TOP 10 NEAREST TO ? FROM articles WHERE lang = ? AND year >= ? RETURNING id, score;
--                      params: [query_vector, "en", 2023]
```

A parameter is substituted after the statement has been split into tokens,
as a value. It is never statement text, so nothing inside it — a quote, a
`;`, a keyword — can change what the statement does. That makes injection
impossible, not merely unlikely, and it is why you should never assemble
V-SQL by pasting strings together.

- A `?` can go anywhere a literal can: `WHERE` comparands, `INSERT` and
  `SET` values, `NEAREST TO ?`, `TOP ?`.
- It cannot stand for a name — `FROM ?` or `SELECT ?` is a syntax error —
  so collection and column names stay in the statement text. The SDKs check
  the names you pass to `search` and `insert` and refuse anything that
  is not a plain identifier.
- The number of `?` must equal the number of parameters.
- A parameter is checked exactly like the literal it stands for: an INT
  parameter is fine for a FLOAT column, a TEXT one against an INT column is a
  type error. Floats must be finite.
- A vector parameter is passed as binary 32-bit floats: exact, and cheaper
  than formatting hundreds of numbers into text.

### Statements

**Create and drop**

```sql
CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(768), title TEXT)
    WITH (capacity = 1000000);

DROP COLLECTION docs;   -- rows, files and all; the name is free again at once
```

**Insert** — every column, the key included. Several rows in one statement
land together or not at all; a key that is already taken, or repeated in the
statement, refuses the whole statement.

```sql
INSERT INTO docs (id, vector, title) VALUES (1, [...], 'one'), (2, [...], 'two');
```

**Search** — the core query.

```sql
SEARCH TOP 5 NEAREST TO [...] FROM docs;                          -- every scalar column
SEARCH TOP 5 NEAREST TO [...] FROM docs RETURNING id, score;      -- pick columns
SEARCH TOP 5 NEAREST TO [...] FROM docs RETURNING id, vector;     -- include embeddings
SEARCH TOP 5 NEAREST TO [...] FROM docs WHERE title != 'draft' RETURNING id, score;
```

Results come nearest first. `score` exists only in `RETURNING`. The query
vector must have the collection's dimension.

**Select** — read rows without ranking them. Row order is unspecified.

```sql
SELECT * FROM docs;                       -- every scalar column, never the vector
SELECT id, vector FROM docs WHERE id = 7; -- name the vector column to get it
```

`LIMIT n` and `OFFSET m` page through a SELECT (both optional, `LIMIT`
first, each a non-negative integer or a `?`):

```sql
SELECT id, title FROM docs WHERE lang = 'en' LIMIT 100 OFFSET 200;
```

The scan stops as soon as the limit is met, so `LIMIT` is cheap even on a
huge collection. Row order is stable for a given database state, so pages
tile the result as long as nothing is written in between. A `SEARCH` takes no
`LIMIT`: `TOP k` already says how many rows it wants.

**Update** — `WHERE` is required. The primary key cannot be changed (delete
and re-insert instead). Replacing the embedding needs the key, because one
statement carries one vector and so must name one row.

```sql
UPDATE docs SET title = 'renamed' WHERE id = 7;
UPDATE docs SET title = 'archived' WHERE id < 100;
UPDATE docs SET vector = [...], title = 'v2' WHERE id = 7;
```

**Delete** — `WHERE` is required; there is no "delete everything" (drop the
collection instead).

```sql
DELETE FROM docs WHERE title = 'draft';
```

### WHERE clauses

Comparisons `=  !=  <  <=  >  >=`, combined with `AND`, `OR` and
parentheses.

```sql
WHERE lang = 'en' AND (year < 2010 OR rating >= 4.5)
```

- Ordered comparisons (`<`, `>`, …) work on `INT` and `FLOAT`; `TEXT`
  supports `=` and `!=` only.
- `IN` and `NOT` are not supported, and say so: write `IN` as an `OR` chain
  and `NOT a = x` as `a != x`.
- There is no `ORDER BY`, `GROUP BY` or join. `SEARCH … TOP k` is
  the way to get "the best k".

**Filtered search keeps k.** The filter runs *before* ranking: `TOP 10 …
WHERE lang = 'en'` returns ten English rows whenever ten exist, however rare
English is. A selective filter also makes the search faster, since fewer
rows are scored.

## Loading data

Use your SDK's `insert` call rather than `INSERT` text: it passes vectors as
binary floats and commits in large batches. Measured at 768 dimensions:

| Path | Rows / second |
|---|---|
| SDK `insert` with float32 buffers (Python `array('f')`/NumPy, JS `Float32Array`) | ~45,000 |
| Rust `Db::insert_many` | ~75,000 |
| `INSERT … VALUES` with 1,000 rows per statement | ~7,000 |
| One row per `INSERT` statement | ~400 |

SDK inserts commit every 10,000 rows. Each batch is atomic; a large load is
not atomic as a whole, so if it fails midway, earlier batches are in. Make
loads idempotent by keying on your own ids and checking what exists first if
you need to resume.

## Sizing: capacity, dimension, memory

- **Disk and memory** are about `capacity × dim × 4` bytes for vectors —
  1M × 768 is ~3 GB — plus the metadata. Vectors are memory-mapped, so the OS
  pages them; for fast search the working set should fit in RAM.
- **Capacity is fixed.** Inserting past it fails with a capacity error. Size
  it for growth. Deleted and updated rows keep their slot until **compaction**
  reclaims it, which runs automatically in the background; an update is a
  delete plus an insert, so heavy update traffic consumes capacity until then.
- **Dimension is fixed** per collection. Changing embedding models means a
  new collection.

## Operations

**Durability and crashes.** A write is acknowledged after it is fsynced to
the log. On open, Flats replays whatever the log holds past the last
checkpoint — after a crash, just open the directory again. Closing cleanly
(`close()`) checkpoints, which makes the next open faster.

**Backups.** Each SDK has `backup(dest)`: a consistent copy, taken while the
database stays open. Reads continue during the copy; writes wait for it. The
copy holds whole statements only. To restore, open the copy. `dest` must not
exist or must be empty.

**Concurrency.** One database handle can be shared by many threads. Reads run
in parallel and never block; writes are serialized. Each statement sees one
consistent snapshot. There are no multi-statement transactions. Async apps
have non-blocking APIs: `flats.aio` in Python and `await flats.open()` in
Node.js. Both run each call off the event loop.

**Listing collections.** V-SQL has no `SHOW` statement. The Rust API has
`Db::collections()`; the bundled shell (`cargo run --release -- DIR`) has
`.tables` and `.schema`.

## Limits and cancellation

**Stopping a statement.** Any statement can be cut short:

- **Interrupt.** `interrupt()` stops every statement running on the database
  at that moment, from any thread. A search stops within a fraction of a
  millisecond, and a scan or cursor before its next row. Statements started
  afterwards are unaffected. It's the tool for a "cancel" button or a
  watchdog.
- **Timeout.** Setting a statement timeout (`set_timeout`,
  `setStatementTimeout`, `flats_set_statement_timeout_ms`,
  `Limits::statement_timeout`) stops any statement that runs longer than
  that. It is off by default.

A stopped statement fails with an *interrupted* error. Writes are one record
each and are logged only once complete, so a stopped statement never leaves
anything half-written.

**Hard caps.** These protect the process from a single hostile or runaway
statement, whatever the settings:

| Cap | Limit | Beyond it |
|---|---|---|
| Parenthesis nesting in a `WHERE` | 64 levels | query error |
| Comparisons in one statement | 4,096 | query error |
| Statement text | 256 MiB by default (`Limits::max_statement_bytes`) | query error |
| Vector dimension | 65,536 | refused at `CREATE` |
| Vector storage per collection (`capacity × dim × 4`) | 4 TiB | refused at `CREATE` |
| Capacity | 2³² − 1 rows | refused at `CREATE` |

Long `AND`/`OR` chains, the way V-SQL spells `IN`, are fine up to the
comparison cap: they are rebalanced internally so they stay cheap.

## Performance

Search time is proportional to `rows × dim`: it reads every live vector once.
Rough figures on a desktop CPU at 768 dimensions:

| Live rows | One search |
|---|---|
| 20,000 | ~2 ms |
| 1,000,000 | ~105 ms |

- **Filter when you can.** A `WHERE` matching 10% of rows searched 1M rows in
  ~27 ms.
- **Throughput** stops scaling past a few concurrent searches: they compete
  for memory bandwidth.
- **Smaller dimensions are proportionally faster.** A 384-dim model searches
  twice as fast as a 768-dim one.
- Flats suits collections up to a few hundred thousand vectors at
  interactive latency, and larger ones for batch or low-QPS work. Past that,
  an approximate (ANN) index is the right tool.

## Errors

Every SDK reports the same four kinds:

| Kind | Means | Typical cause |
|---|---|---|
| Query error | The statement is wrong; nothing changed | typo, unknown collection/column, unsupported syntax, missing column in INSERT |
| Engine error | The database refused or failed it | duplicate primary key, collection full, backup target not empty, I/O |
| Invalid argument | A value or handle the call cannot take | wrong vector length, wrong type for a column, NaN, closed database |
| Interrupted | The statement was stopped | `interrupt()` was called, or it ran past its timeout |
| Internal error | A bug in Flats | please report it |

Messages are specific — they name the column, value or construct at fault —
so they are worth surfacing to your users or logs as they are.
