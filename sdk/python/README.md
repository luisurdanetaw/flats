# Flats for Python

Exact vector search with metadata filters, embedded in your Python process.
Pure ctypes over the Flats C library: Python 3.8+, no third-party packages.

For V-SQL and the concepts shared by every SDK, see the
[user guide](../../docs/guide.md).

## Setup

Build the shared library once, then point Python at it:

```sh
cargo build --release -p flats-c
export FLATS_LIBRARY=$PWD/target/release/libflats_c.so
export PYTHONPATH=$PWD/sdk/python
python3 -c "import flats; print(flats.version())"
```

Instead of `FLATS_LIBRARY` you can copy `libflats_c.so` into
`sdk/python/flats/`. Inside a Flats checkout neither is needed: the package
finds `target/release` (or `target/debug`) on its own.

## Quickstart

```python
import flats

with flats.Database("./my-db") as db:
    db.execute("""
        CREATE COLLECTION articles (
            id TEXT PRIMARY KEY,
            vector VECTOR(4),
            lang TEXT,
            year INT
        ) WITH (capacity = 10000);
    """)

    db.insert("articles", [
        {"id": "a1", "vector": [0.9, 0.1, 0.0, 0.0], "lang": "en", "year": 2023},
        {"id": "a2", "vector": [0.1, 0.9, 0.0, 0.0], "lang": "en", "year": 2024},
        {"id": "a3", "vector": [0.8, 0.2, 0.1, 0.0], "lang": "es", "year": 2024},
    ])

    hits = db.search("articles", [1, 0, 0, 0], k=2,
                     where="lang = ?", params=["en"],
                     returning=["id", "score", "year"])
    for row in hits.dicts():
        print(row)
    # {'id': 'a1', 'score': 0.8999999761581421, 'year': 2023}
    # {'id': 'a2', 'score': 0.10000000149011612, 'year': 2024}
```

(Vectors and scores are 32-bit floats, hence `0.8999999761581421` for 0.9.)

The directory is created on first open. Reopening it later brings everything
back.

## Opening and closing

```python
db = flats.Database("./my-db")     # opens, or creates, and runs crash recovery
...
db.close()                          # checkpoints; safe to call twice
```

Prefer `with flats.Database(...) as db:`, which closes for you. A database
that is never closed is still safe: everything acknowledged is in the
write-ahead log and is recovered on the next open. Closing only makes that
next open faster. `db.closed` tells you whether it has been closed; any call
on a closed database raises `FlatsError`.

## Creating collections

Collections are created with V-SQL:

```python
db.execute("""
    CREATE COLLECTION docs (
        doc_id TEXT PRIMARY KEY,
        vector VECTOR(768),
        source TEXT,
        published INT,
        quality FLOAT
    ) WITH (capacity = 1000000);
""")
```

Pick the dimension of your embedding model and a capacity with room to grow;
both are fixed. Drop a collection with `db.execute("DROP COLLECTION docs;")`.

## Inserting

`insert(collection, rows)` takes any iterable of dicts and returns how many
rows it inserted:

```python
n = db.insert("docs", (
    {"doc_id": d.id, "vector": d.embedding, "source": d.source,
     "published": d.year, "quality": d.score}
    for d in documents
))
```

- Every column must be present in every row, including the primary key.
- Python types map directly: `int` → INT, `float` → FLOAT, `str` → TEXT.
  `bool` and `None` are refused.
- The vector is the row's one sequence value: a `list`, `tuple`,
  `array.array`, NumPy array or anything else that is a sequence of numbers.
- Rows are sent in batches of `flats.Database.BATCH_ROWS` (10,000). **Each
  batch is atomic**: a duplicate key or a bad value anywhere in a batch
  inserts none of it, and the error tells you which value. Batches already
  committed stay committed.

**Fast path.** Pass float32 buffers and each vector is copied in one move
instead of float by float, about 4× faster (~45,000 rows/s at 768 dimensions):

```python
import array
vec = array.array("f", embedding)     # or: numpy.asarray(embedding, dtype=numpy.float32)
```

## Searching

```python
hits = db.search("docs", query_vector, k=10,
                 where="source = ? AND published >= ?", params=[source, 2020],
                 returning=["doc_id", "score", "quality"])
```

- `k` — how many rows to return (default 10). Results come nearest first.
- `where` — a V-SQL predicate, applied *before* ranking, so you get `k`
  matching rows whenever `k` exist. Comparisons `= != < <= > >=` with
  `AND`/`OR`; see the [guide](../../docs/guide.md#where-clauses). Write each
  value as `?` and pass it in `params` — see
  [Passing values](#passing-values-always-use-parameters).
- `returning` — the columns to return. `score` is the similarity (dot
  product, higher is closer); `vector` returns the stored embedding. Without
  it you get every scalar column and no score.

The score is a dot product. For cosine similarity, normalize your vectors to
unit length before inserting and before searching.

## Results

`execute` and `search` return a `flats.Result`, a read-only sequence of
tuples:

```python
res = db.execute("SELECT doc_id, published FROM docs WHERE quality > 0.9;")
res.columns          # ('doc_id', 'published')
len(res)             # number of rows
res[0]               # ('d17', 2021)
for doc_id, year in res: ...
res.dicts()          # [{'doc_id': 'd17', 'published': 2021}, ...]
```

Cells are `int`, `float`, `str`, or `list[float]` for a vector. Write
statements return an empty `Result`.

## Passing values: always use parameters

Never build a statement by pasting values into the SQL string. Write `?`
where each value goes and pass the values, in order, as `params`:

```python
# NO — user_lang can rewrite the statement:  "en' OR '1'='1"
db.execute(f"SELECT doc_id FROM docs WHERE source = '{user_lang}';")

# YES — user_lang is only ever a value
db.execute("SELECT doc_id FROM docs WHERE source = ?;", [user_lang])
```

A parameter is substituted after the statement is parsed into tokens, as a
value. Nothing inside it — quotes, `;`, keywords — can change what the
statement does. Parameters work anywhere a literal can go: `WHERE`
comparands, `INSERT` values, `SET` values, `NEAREST TO ?`, `TOP ?`.

| Python value | V-SQL type |
|---|---|
| `int` | INT (also accepted for FLOAT columns) |
| `float` | FLOAT (must be finite) |
| `str` | TEXT |
| list / tuple / `array.array` / NumPy array of numbers | VECTOR, passed as binary floats |

`bool` and `None` are refused. A parameter can't stand for a collection or
column name; `search` and `insert` check the names you pass and refuse
anything that isn't a plain identifier.

## Updating and deleting

V-SQL through `execute`, with values as parameters:

```python
db.execute("UPDATE docs SET quality = ? WHERE source = ?;", [0.1, "spam"])
db.execute("UPDATE docs SET vector = ? WHERE doc_id = ?;", [new_embedding, "d17"])
db.execute("DELETE FROM docs WHERE published < ?;", [2000])
```

Both need a `WHERE`. The primary key can't be changed, and a new vector can
only be set on one row at a time, named by its key.

## Backups

```python
db.backup("/backups/my-db-2024-06-01")
restored = flats.Database("/backups/my-db-2024-06-01")
```

The copy is consistent (whole statements only) and taken while the database
stays open. The destination must not exist or must be empty.

## Errors

Every error is a `flats.FlatsError`; catch the subclass you care about:

| Exception | When |
|---|---|
| `flats.QueryError` | Bad or unsupported V-SQL, an unknown collection or column, a missing column in an INSERT, a `?` count that doesn't match `params` |
| `flats.EngineError` | Duplicate primary key, collection full, backup target not empty, I/O failure |
| `flats.InvalidArgumentError` | Wrong vector length, a value of the wrong type for its column, NaN or infinity, unknown column in `insert` |
| `flats.InternalError` | A bug in Flats |

```python
try:
    db.insert("docs", rows)
except flats.EngineError as e:
    print("refused:", e)        # e.g. a live row already holds primary key 'd17' ...
```

`e.code` holds the numeric status from the C library. Passing a non-`str`
where text is expected raises `TypeError`. A non-finite query vector, or a
collection or column name that isn't a plain identifier, raises `ValueError`.

## Threads

A `Database` can be shared by any number of threads. Every call releases the
GIL while Flats works, so searches on different threads really run in
parallel, and writes are serialized internally. Keep one `Database` per
directory per process.

## Running the tests

```sh
cargo build --release -p flats-c
python3 -m unittest discover -s sdk/python/tests -t sdk/python
```
