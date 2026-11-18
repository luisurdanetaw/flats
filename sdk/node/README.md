# Flats for Node.js

Exact vector search with metadata filters, embedded in your Node process. A
native addon over the Flats C library: Node 18+, no npm dependencies, no
node-gyp. TypeScript types included (`index.d.ts`).

For V-SQL and the concepts shared by every SDK, see the
[user guide](../../docs/guide.md).

## Setup

Build the addon once and put it next to `index.js`:

```sh
cargo build --release -p flats-node
cp target/release/libflats_node.so sdk/node/flats.node
```

Then `require('./sdk/node')` (or add `sdk/node` to your project as a local
dependency: `npm install ./path/to/flats/sdk/node`). Instead of copying, you
can set `FLATS_NODE_ADDON` to the `.so`; inside a Flats checkout the package
finds `target/release` on its own.

## Quickstart

```js
const flats = require('flats');

const db = new flats.Database('./my-db');

db.execute(`
  CREATE COLLECTION articles (
    id TEXT PRIMARY KEY,
    vector VECTOR(4),
    lang TEXT,
    year INT
  ) WITH (capacity = 10000);
`);

db.insert('articles', [
  { id: 'a1', vector: [0.9, 0.1, 0, 0], lang: 'en', year: 2023 },
  { id: 'a2', vector: [0.1, 0.9, 0, 0], lang: 'en', year: 2024 },
  { id: 'a3', vector: [0.8, 0.2, 0.1, 0], lang: 'es', year: 2024 },
]);

const hits = db.search('articles', [1, 0, 0, 0], {
  k: 2,
  where: 'lang = ?',
  params: ['en'],
  returning: ['id', 'score', 'year'],
});
console.log(hits.toObjects());
// [ { id: 'a1', score: 0.8999999761581421, year: 2023 },
//   { id: 'a2', score: 0.10000000149011612, year: 2024 } ]

db.close();
```

(Vectors and scores are 32-bit floats, hence `0.8999999761581421` for 0.9.)

## Opening and closing

```js
const db = new flats.Database('./my-db'); // opens, or creates, and runs crash recovery
db.close();                               // checkpoints; safe to call twice
```

A database that is never closed is still safe: everything acknowledged is in
the write-ahead log and is recovered on the next open, and the garbage
collector releases the handle eventually. Close explicitly anyway, on
shutdown, so the next open is fast. `db.closed` tells you whether it has
been closed; any call on a closed database throws `FlatsError`.

## Creating collections

```js
db.execute(`
  CREATE COLLECTION docs (
    doc_id TEXT PRIMARY KEY,
    vector VECTOR(768),
    source TEXT,
    published INT,
    quality FLOAT
  ) WITH (capacity = 1000000);
`);
```

The dimension and capacity are fixed; pick capacity with room to grow. Drop a
collection with `db.execute('DROP COLLECTION docs;')`.

## Inserting

`insert(collection, rows)` takes any iterable of plain objects and returns
the number inserted:

```js
const n = db.insert('docs', documents.map((d) => ({
  doc_id: d.id,
  vector: d.embedding,       // Array, Float32Array or Float64Array
  source: d.source,
  published: d.year,
  quality: d.score,
})));
```

- Every column must be present in every row, the primary key included.
- Numbers go where the schema says: into an INT column they must be whole
  (`1.5` is refused), into a FLOAT column anything finite. Use a `BigInt`
  for INT values beyond ±2^53. Strings go into TEXT. Booleans, `null` and
  `undefined` are refused.
- Rows are sent in batches of `flats.BATCH_ROWS` (10,000). **Each batch is
  atomic**: a duplicate key or a bad value anywhere in it inserts none of it,
  and the error names the value. Batches already committed stay committed.

**Fast path.** `Float32Array` vectors are copied in one move: about 47,000
rows/s at 768 dimensions.

## Searching

```js
const hits = db.search('docs', queryVector, {
  k: 10,
  where: 'source = ? AND published >= ?',
  params: [source, 2020],
  returning: ['doc_id', 'score', 'quality'],
});
```

- `k` — how many rows (default 10). Results come nearest first.
- `where` — a V-SQL predicate applied *before* ranking, so you get `k`
  matching rows whenever `k` exist. See the
  [guide](../../docs/guide.md#where-clauses). Write each value as `?` and
  pass it in `params` — see
  [Passing values](#passing-values-always-use-parameters).
- `returning` — the columns to return. `score` is the similarity (dot
  product, higher is closer); `vector` returns the stored embedding. Without
  it you get every scalar column and no score.

For cosine similarity, normalize vectors to unit length before inserting and
before searching.

## Results

`execute` and `search` return a `Result`:

```js
const res = db.execute('SELECT doc_id, published FROM docs WHERE quality > 0.9;');
res.columns;     // ['doc_id', 'published']
res.rows;        // [['d17', 2021], ...]
res.length;      // number of rows
for (const [docId, year] of res) { /* ... */ }
res.toObjects(); // [{ doc_id: 'd17', published: 2021 }, ...]
```

Cells are numbers, strings, arrays of numbers (vectors), or `BigInt` for an
INT too large to be a JS number exactly. Write statements return an empty
`Result`.

## Passing values: always use parameters

Never build a statement by pasting values into the SQL string. Write `?`
where each value goes and pass the values, in order:

```js
// NO — userLang can rewrite the statement:  "en' OR '1'='1"
db.execute(`SELECT doc_id FROM docs WHERE source = '${userLang}';`);

// YES — userLang is only ever a value
db.execute('SELECT doc_id FROM docs WHERE source = ?;', [userLang]);
```

A parameter is substituted after the statement is parsed into tokens, as a
value. Nothing inside it — quotes, `;`, keywords — can change what the
statement does. Parameters work anywhere a literal can go: `WHERE`
comparands, `INSERT` values, `SET` values, `NEAREST TO ?`, `TOP ?`.

| JS value | V-SQL type |
|---|---|
| a whole `number` (safe integer) | INT (also accepted for FLOAT columns) |
| any other finite `number` | FLOAT |
| `bigint` | INT, exactly |
| `string` | TEXT |
| `Array`, `Float32Array`, `Float64Array` of numbers | VECTOR, passed as binary floats |

Booleans, `null`, `undefined` and objects are refused. A parameter can't
stand for a collection or column name; `search` and `insert` check the names
you pass and refuse anything that isn't a plain identifier.

## Updating and deleting

V-SQL through `execute`, with values as parameters:

```js
db.execute('UPDATE docs SET quality = ? WHERE source = ?;', [0.1, 'spam']);
db.execute('UPDATE docs SET vector = ? WHERE doc_id = ?;', [newEmbedding, 'd17']);
db.execute('DELETE FROM docs WHERE published < ?;', [2000]);
```

Both need a `WHERE`. The primary key can't be changed, and a new vector can
only be set on one row at a time, named by its key.

## Backups

```js
db.backup('/backups/my-db-2024-06-01');
const restored = new flats.Database('/backups/my-db-2024-06-01');
```

The copy is consistent (whole statements only) and taken while the database
stays open. The destination must not exist or must be empty.

## Errors

Every error is a `flats.FlatsError`; check the subclass:

| Class | When |
|---|---|
| `flats.QueryError` | Bad or unsupported V-SQL, an unknown collection or column, a missing column in an INSERT, a `?` count that doesn't match `params` |
| `flats.EngineError` | Duplicate primary key, collection full, backup target not empty, I/O failure |
| `flats.InvalidArgumentError` | Wrong vector length, a value of the wrong type for its column, NaN or infinity, unknown column in `insert` |
| `flats.InternalError` | A bug in Flats |

```js
try {
  db.insert('docs', rows);
} catch (e) {
  if (e instanceof flats.EngineError) console.error('refused:', e.message);
  else throw e;
}
```

`e.code` holds the numeric status from the C library. Wrong argument types
(a number for the SQL string, say), or a collection or column name that isn't
a plain identifier, throw `TypeError`; a non-finite query vector in `search`
throws `RangeError`.

## Blocking and threads

Calls are **synchronous**, like `better-sqlite3`: a search holds the event
loop for its duration — about 2 ms over 20,000 vectors of 768 dimensions,
~100 ms over a million. For a server under load, open the database inside a
`worker_threads` worker and send it queries by message, or keep collections
small enough that searches stay in the low milliseconds. Either way, open
each directory from exactly one place in the process: one `Database` per
directory.

## Running the tests

```sh
cargo build --release -p flats-node
node --test sdk/node/test/
```
