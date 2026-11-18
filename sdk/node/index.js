// Flats — a tiny embeddable vector database. Node.js bindings.
//
// A thin layer over the native addon (sdk/node/src/lib.rs), which is itself a
// thin layer over the C ABI (sdk/c/include/flats.h). No npm dependencies.
//
//   const flats = require('flats');
//   const db = new flats.Database('./my-db');
//   db.execute("CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3)) WITH (capacity = 1000);");
//   db.insert('docs', [{ id: 'd1', vector: [0.1, 0.2, 0.3] }]);
//   db.search('docs', [0.1, 0.2, 0.3], { k: 5, returning: ['id', 'score'] }).rows;
//   db.close();
//
// Calls are synchronous, like better-sqlite3: a search blocks the event loop
// for its duration (a brute-force scan — milliseconds at 100k vectors).
//
// Finding the addon: FLATS_NODE_ADDON wins; otherwise flats.node next to this
// file, then a Flats checkout's target/release and target/debug.

'use strict';

const fs = require('node:fs');
const path = require('node:path');

function locate() {
  if (process.env.FLATS_NODE_ADDON) return [process.env.FLATS_NODE_ADDON];
  const root = path.resolve(__dirname, '..', '..');
  return [
    path.join(__dirname, 'flats.node'),
    path.join(root, 'target', 'release', 'libflats_node.so'),
    path.join(root, 'target', 'debug', 'libflats_node.so'),
  ];
}

function load() {
  const tried = locate();
  for (const file of tried) {
    if (fs.existsSync(file)) {
      const m = { exports: {} };
      process.dlopen(m, file);
      return m.exports;
    }
  }
  throw new Error(
    'flats: native addon not found (build it with `cargo build --release -p flats-node`, ' +
      `or set FLATS_NODE_ADDON). Looked in: ${tried.join(', ')}`,
  );
}

const native = load();

// -- errors: one class per status code in flats.h ---------------------------

class FlatsError extends Error {
  constructor(message, code) {
    super(message);
    this.name = new.target.name;
    this.code = code;
  }
}
class InvalidArgumentError extends FlatsError {}
class QueryError extends FlatsError {}
class EngineError extends FlatsError {}
class InternalError extends FlatsError {}

const CLASSES = { 1: InvalidArgumentError, 2: QueryError, 3: EngineError, 4: InternalError };

/** Call into the addon, rethrowing its errors as the matching class. */
function call(fn, ...args) {
  try {
    return fn(...args);
  } catch (e) {
    if (e instanceof TypeError) throw e;
    const code = Number(e.code);
    const Cls = CLASSES[code];
    if (Cls) throw new Cls(e.message, code);
    throw e;
  }
}

// -- results ------------------------------------------------------------------

/** The rows of one statement: `columns` and `rows` (arrays of cells). */
class Result {
  constructor(columns, rows) {
    this.columns = columns;
    this.rows = rows;
  }
  get length() {
    return this.rows.length;
  }
  [Symbol.iterator]() {
    return this.rows[Symbol.iterator]();
  }
  /** Every row as a `{column: value}` object. */
  toObjects() {
    return this.rows.map((row) => Object.fromEntries(this.columns.map((c, i) => [c, row[i]])));
  }
}

// -- formatting values into V-SQL ---------------------------------------------

function floatSql(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) throw new RangeError(`vector components must be finite, got ${x}`);
  // String() is the shortest exact spelling, exponent included ('1e-7').
  return String(n);
}

function vectorSql(vector) {
  return `[${Array.from(vector, floatSql).join(', ')}]`;
}

// -- the database -------------------------------------------------------------

/** Rows sent per insert statement; each batch is atomic. */
const BATCH_ROWS = 10000;

class Database {
  /** Open (creating if missing) the database in directory `path`. */
  constructor(dbPath) {
    if (typeof dbPath !== 'string') throw new TypeError('path must be a string');
    this._db = call(native.open, dbPath);
  }

  get closed() {
    return this._db === null;
  }

  /** Checkpoint and close. Idempotent. */
  close() {
    const db = this._db;
    this._db = null;
    if (db) call(native.close, db);
  }

  _handle() {
    if (this._db === null) throw new FlatsError('database is closed', 1);
    return this._db;
  }

  /** Run one V-SQL statement and return its Result. */
  execute(sql) {
    if (typeof sql !== 'string') throw new TypeError('sql must be a string');
    const [columns, rows] = call(native.execute, this._handle(), sql);
    return new Result(columns, rows);
  }

  /**
   * Top-k rows of `collection` nearest `vector` (dot product).
   * Options: `k` (default 10), `where` (a V-SQL predicate), `returning`
   * (column names; `score` and `vector` included).
   */
  search(collection, vector, { k = 10, where, returning } = {}) {
    let sql = `SEARCH TOP ${Math.trunc(k)} NEAREST TO ${vectorSql(vector)} FROM ${collection}`;
    if (where) sql += ` WHERE ${where}`;
    if (returning && returning.length) sql += ` RETURNING ${returning.join(', ')}`;
    return this.execute(`${sql};`);
  }

  /**
   * Insert `rows` — objects of column name to value — and return the count.
   * The embedding is an Array, Float32Array or Float64Array; INT columns take
   * integers (or BigInt), FLOAT columns numbers, TEXT columns strings. Rows go
   * in batches of BATCH_ROWS, each all-or-nothing.
   */
  insert(collection, rows) {
    const batch = call(native.batchNew, this._handle(), collection);
    let total = 0;
    try {
      for (const row of rows) {
        if (row === null || typeof row !== 'object') throw new TypeError('each row must be an object');
        for (const [column, value] of Object.entries(row)) {
          call(native.batchSet, batch, column, value);
        }
        call(native.batchEndRow, batch);
        if (call(native.batchLen, batch) >= BATCH_ROWS) total += call(native.batchCommit, batch);
      }
      if (call(native.batchLen, batch) > 0) total += call(native.batchCommit, batch);
    } finally {
      native.batchFree(batch);
    }
    return total;
  }

  /** Write a consistent copy into `dest` (absent or empty). Restore with `new Database(dest)`. */
  backup(dest) {
    if (typeof dest !== 'string') throw new TypeError('dest must be a string');
    call(native.backup, this._handle(), dest);
  }
}

/** The library version, e.g. "0.1.0". */
function version() {
  return native.version();
}

module.exports = {
  Database,
  Result,
  FlatsError,
  InvalidArgumentError,
  QueryError,
  EngineError,
  InternalError,
  BATCH_ROWS,
  version,
};
