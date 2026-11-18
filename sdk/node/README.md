# Flats — Node.js

An N-API addon over the C ABI. Node 18+, no npm dependencies, no node-gyp: the
addon is a Rust `cdylib` that declares the N-API functions it uses by hand.

```sh
cargo build --release -p flats-node
cp target/release/libflats_node.so sdk/node/flats.node     # or set FLATS_NODE_ADDON
```

```js
const flats = require('./sdk/node');

const db = new flats.Database('./my-db');
db.execute('CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), year INT) WITH (capacity = 100000);');

// Batched through the binary insert path; Float32Array is the fastest vector.
db.insert('docs', [
  { id: 'a', vector: new Float32Array([0.1, 0.9, 0.0]), year: 2023 },
  { id: 'b', vector: [0.8, 0.1, 1e-5], year: 2024 },
]);

const hits = db.search('docs', [1, 0, 0], { k: 5, where: 'year >= 2024', returning: ['id', 'score'] });
console.log(hits.toObjects());

db.backup('./my-db-backup');
db.close();
```

Calls are synchronous, like better-sqlite3. INT cells beyond 2^53 come back
as `BigInt` (and a `BigInt` goes into an INT column exactly). Errors are
`QueryError`, `EngineError`, `InvalidArgumentError`, `InternalError`, all
`FlatsError` with the C status in `.code`. TypeScript types: `index.d.ts`.

Tests: `node --test sdk/node/test/`.
