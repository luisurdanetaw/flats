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
/** The statement was stopped: interrupt(), or its timeout. Nothing it would
 * have written was written. */
class InterruptedError extends FlatsError {}

const CLASSES = { 1: InvalidArgumentError, 2: QueryError, 3: EngineError, 4: InternalError, 5: InterruptedError };

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

/**
 * The rows of one statement, read lazily in chunks — see Database#iterate.
 * Iterate it once (`for…of`, spread, or `next()`); breaking out of a loop, or
 * `close()`, releases it early. `columns` is known before the first row.
 */
class RowIterator {
  constructor(cursor, chunkRows) {
    this._cursor = cursor;
    this._chunkRows = chunkRows;
    this._buffer = [];
    this._at = 0;
    this._fill();
  }

  _fill() {
    const [columns, rows] = call(native.cursorNext, this._cursor, this._chunkRows);
    this.columns = columns;
    this._buffer = rows;
    this._at = 0;
    if (rows.length === 0) this._release();
  }

  get closed() {
    return this._cursor === null;
  }

  /** Release the cursor. Idempotent; iteration then simply ends. */
  close() {
    this._buffer = [];
    this._at = 0;
    this._release();
  }

  _release() {
    const cursor = this._cursor;
    this._cursor = null;
    if (cursor) native.cursorFree(cursor);
  }

  next() {
    for (;;) {
      if (this._at < this._buffer.length) return { value: this._buffer[this._at++], done: false };
      if (this._cursor === null) return { value: undefined, done: true };
      this._fill();
    }
  }

  /** Called by `for…of` on `break`/`throw`: release early. */
  return() {
    this.close();
    return { value: undefined, done: true };
  }

  [Symbol.iterator]() {
    return this;
  }
}

// -- names --------------------------------------------------------------------

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** A collection or column name, checked: names are statement text, so they
 * must be plain identifiers — values go in parameters. */
function ident(name, what) {
  if (typeof name !== 'string' || !IDENT.test(name)) {
    throw new TypeError(`${what} must be a plain identifier, got ${JSON.stringify(name)}`);
  }
  return name;
}

/** The SEARCH statement `search` runs; its first `?` is the query vector. */
function searchSql(collection, vector, { k, where, returning }) {
  if (!Number.isInteger(k)) throw new TypeError('k must be an integer');
  for (const x of vector) {
    if (!Number.isFinite(x)) throw new RangeError(`vector components must be finite, got ${x}`);
  }
  let sql = `SEARCH TOP ${k} NEAREST TO ? FROM ${ident(collection, 'collection')}`;
  if (where) sql += ` WHERE ${where}`;
  if (returning && returning.length) {
    sql += ` RETURNING ${returning.map((c) => ident(c, 'column')).join(', ')}`;
  }
  return `${sql};`;
}

/** Add one row object to a native batch. */
function addRow(batch, row) {
  if (row === null || typeof row !== 'object') throw new TypeError('each row must be an object');
  for (const [column, value] of Object.entries(row)) {
    call(native.batchSet, batch, column, value);
  }
  call(native.batchEndRow, batch);
}

// -- the database -------------------------------------------------------------

/** Rows sent per insert statement; each batch is atomic. */
const BATCH_ROWS = 10000;

/** `options.salvage`, checked: undefined/omitted means false. */
function salvageOption(options) {
  if (options === undefined) return false;
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('options must be an object');
  }
  const { salvage = false } = options;
  if (typeof salvage !== 'boolean') throw new TypeError('options.salvage must be a boolean');
  return salvage;
}

class Database {
  /**
   * Open (creating if missing) the database in directory `path`.
   *
   * If a collection's store file is damaged in a way the log can't repair,
   * this throws EngineError naming the file, and changes nothing. Restore a
   * backup, or open once with `{ salvage: true }`: each damaged collection
   * then comes back with only the rows written since its last checkpoint.
   */
  constructor(dbPath, options) {
    if (typeof dbPath !== 'string') throw new TypeError('path must be a string');
    const salvage = salvageOption(options);
    this._db = call(salvage ? native.openSalvage : native.open, dbPath);
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

  /**
   * Run one V-SQL statement and return its Result.
   *
   * Write `?` wherever a value goes and pass the values in `params`, in
   * order. Always pass outside values this way: a parameter is a value, never
   * statement text, so nothing in it can change what the statement does.
   */
  execute(sql, params = []) {
    if (typeof sql !== 'string') throw new TypeError('sql must be a string');
    if (!Array.isArray(params)) throw new TypeError('params must be an array');
    const [columns, rows] = call(native.execute, this._handle(), sql, params);
    return new Result(columns, rows);
  }

  /**
   * Run one V-SQL statement and iterate its rows lazily: they are read from
   * storage `chunkRows` at a time as you go, so memory stays flat however
   * many there are. The iterator reads one snapshot — rows written after this
   * call are not in it. Use it for large reads; execute() collects everything.
   */
  iterate(sql, params = [], { chunkRows = 1000 } = {}) {
    if (typeof sql !== 'string') throw new TypeError('sql must be a string');
    if (!Array.isArray(params)) throw new TypeError('params must be an array');
    if (!Number.isInteger(chunkRows) || chunkRows < 1) throw new RangeError('chunkRows must be a positive integer');
    const cursor = call(native.query, this._handle(), sql, params);
    return new RowIterator(cursor, chunkRows);
  }

  /**
   * Top-k rows of `collection` nearest `vector` (dot product).
   * Options: `k` (default 10); `where`, a V-SQL predicate whose values are
   * `?` placeholders filled from `params`; `returning`, column names
   * (`score` and `vector` included). The vector is passed as a binary
   * parameter, exactly.
   */
  search(collection, vector, { k = 10, where, params = [], returning } = {}) {
    return this.execute(searchSql(collection, vector, { k, where, returning }), [vector, ...params]);
  }

  /**
   * Insert `rows` — objects of column name to value — and return the count.
   * The embedding is an Array, Float32Array or Float64Array; INT columns take
   * integers (or BigInt), FLOAT columns numbers, TEXT columns strings. Rows go
   * in batches of BATCH_ROWS, each all-or-nothing.
   */
  insert(collection, rows) {
    const batch = call(native.batchNew, this._handle(), ident(collection, 'collection'));
    let total = 0;
    try {
      for (const row of rows) {
        addRow(batch, row);
        if (call(native.batchLen, batch) >= BATCH_ROWS) total += call(native.batchCommit, batch);
      }
      if (call(native.batchLen, batch) > 0) total += call(native.batchCommit, batch);
    } finally {
      native.batchFree(batch);
    }
    return total;
  }

  /**
   * Stop every statement running on this database right now with
   * InterruptedError. Statements started afterwards are unaffected.
   */
  interrupt() {
    call(native.interrupt, this._handle());
  }

  /** Stop any statement running longer than `ms` milliseconds; null disables. */
  setStatementTimeout(ms) {
    if (ms !== null && (typeof ms !== 'number' || !(ms >= 0))) throw new TypeError('ms must be a non-negative number or null');
    call(native.setStatementTimeout, this._handle(), ms === null ? -1 : Math.floor(ms));
  }

  /** Write a consistent copy into `dest` (absent or empty). Restore with `new Database(dest)`. */
  backup(dest) {
    if (typeof dest !== 'string') throw new TypeError('dest must be a string');
    call(native.backup, this._handle(), dest);
  }
}

// -- the Promise API ------------------------------------------------------------

/** Like call(), for a native function that returns a Promise. */
async function callAsync(fn, ...args) {
  try {
    return await fn(...args);
  } catch (e) {
    if (e instanceof TypeError) throw e;
    const code = Number(e.code);
    const Cls = CLASSES[code];
    if (Cls) throw new Cls(e.message, code);
    throw e;
  }
}

/**
 * The rows of one statement, read lazily in chunks on the thread pool — see
 * AsyncDatabase#query. Use `for await…of`; breaking out releases it.
 */
class AsyncRowIterator {
  constructor(cursor, chunkRows, columns, rows) {
    this._cursor = cursor;
    this._chunkRows = chunkRows;
    this.columns = columns;
    this._buffer = rows;
    this._at = 0;
    if (rows.length === 0) this._release();
  }

  get closed() {
    return this._cursor === null;
  }

  close() {
    this._buffer = [];
    this._at = 0;
    this._release();
  }

  _release() {
    const cursor = this._cursor;
    this._cursor = null;
    if (cursor) native.cursorFree(cursor);
  }

  async next() {
    for (;;) {
      if (this._at < this._buffer.length) return { value: this._buffer[this._at++], done: false };
      if (this._cursor === null) return { value: undefined, done: true };
      // One chunk at a time: a cursor is used by one thread at a time.
      if (this._pending) await this._pending;
      else {
        this._pending = callAsync(native.cursorNextAsync, this._cursor, this._chunkRows);
        try {
          const [, rows] = await this._pending;
          this._buffer = rows;
          this._at = 0;
          if (rows.length === 0) this._release();
        } finally {
          this._pending = null;
        }
      }
    }
  }

  async return() {
    this.close();
    return { value: undefined, done: true };
  }

  [Symbol.asyncIterator]() {
    return this;
  }
}

/**
 * The Promise API: every call runs on libuv's thread pool and returns a
 * Promise, so the event loop keeps turning while Flats works. Get one with
 * `await flats.open(path)`. `close()` waits for calls still in flight.
 */
class AsyncDatabase {
  constructor(handle) {
    this._db = handle;
    this._inFlight = new Set();
    this._closing = null;
  }

  get closed() {
    return this._db === null;
  }

  /** Track a call so close() can wait for it. */
  _run(start) {
    if (this._db === null || this._closing) {
      return Promise.reject(new FlatsError('database is closed', 1));
    }
    const p = (async () => start(this._db))();
    this._inFlight.add(p);
    const done = () => this._inFlight.delete(p);
    p.then(done, done);
    return p;
  }

  /** Wait for calls in flight, then checkpoint and close. Idempotent. */
  close() {
    if (!this._closing) {
      this._closing = (async () => {
        await Promise.allSettled([...this._inFlight]);
        const db = this._db;
        this._db = null;
        if (db) await callAsync(native.closeAsync, db);
      })();
    }
    return this._closing;
  }

  /** Run one V-SQL statement; resolves to a Result. Values go in `params`. */
  execute(sql, params = []) {
    if (typeof sql !== 'string') return Promise.reject(new TypeError('sql must be a string'));
    if (!Array.isArray(params)) return Promise.reject(new TypeError('params must be an array'));
    return this._run(async (db) => {
      const [columns, rows] = await callAsync(native.executeAsync, db, sql, params);
      return new Result(columns, rows);
    });
  }

  /** Stream a statement's rows: resolves to an async iterator. */
  query(sql, params = [], { chunkRows = 1000 } = {}) {
    if (typeof sql !== 'string') return Promise.reject(new TypeError('sql must be a string'));
    if (!Array.isArray(params)) return Promise.reject(new TypeError('params must be an array'));
    if (!Number.isInteger(chunkRows) || chunkRows < 1) {
      return Promise.reject(new RangeError('chunkRows must be a positive integer'));
    }
    return this._run(async (db) => {
      const cursor = await callAsync(native.queryAsync, db, sql, params);
      const [columns, rows] = await callAsync(native.cursorNextAsync, cursor, chunkRows);
      return new AsyncRowIterator(cursor, chunkRows, columns, rows);
    });
  }

  /** Top-k nearest `vector`; the same options as Database#search. */
  search(collection, vector, { k = 10, where, params = [], returning } = {}) {
    let sql;
    try {
      sql = searchSql(collection, vector, { k, where, returning });
    } catch (e) {
      return Promise.reject(e);
    }
    return this.execute(sql, [vector, ...params]);
  }

  /** Insert row objects; resolves to the count. Batches of BATCH_ROWS, each atomic. */
  insert(collection, rows) {
    try {
      ident(collection, 'collection');
    } catch (e) {
      return Promise.reject(e);
    }
    return this._run(async (db) => {
      const batch = call(native.batchNew, db, collection);
      let total = 0;
      try {
        for (const row of rows) {
          addRow(batch, row);
          if (call(native.batchLen, batch) >= BATCH_ROWS) {
            total += await callAsync(native.batchCommitAsync, batch);
          }
        }
        if (call(native.batchLen, batch) > 0) total += await callAsync(native.batchCommitAsync, batch);
      } finally {
        native.batchFree(batch);
      }
      return total;
    });
  }

  /** Stop every statement running now (synchronous: it never waits). */
  interrupt() {
    if (this._db === null) throw new FlatsError('database is closed', 1);
    call(native.interrupt, this._db);
  }

  /** Stop any statement running longer than `ms`; null disables. */
  setStatementTimeout(ms) {
    if (this._db === null) throw new FlatsError('database is closed', 1);
    if (ms !== null && (typeof ms !== 'number' || !(ms >= 0))) throw new TypeError('ms must be a non-negative number or null');
    call(native.setStatementTimeout, this._db, ms === null ? -1 : Math.floor(ms));
  }

  /** Write a consistent copy into `dest`. */
  backup(dest) {
    if (typeof dest !== 'string') return Promise.reject(new TypeError('dest must be a string'));
    return this._run((db) => callAsync(native.backupAsync, db, dest));
  }
}

/** Open (creating if missing) the database in `path`; resolves to an AsyncDatabase. */
async function open(dbPath, options) {
  if (typeof dbPath !== 'string') throw new TypeError('path must be a string');
  const salvage = salvageOption(options);
  return new AsyncDatabase(await callAsync(salvage ? native.openSalvageAsync : native.openAsync, dbPath));
}

/** The library version, e.g. "0.1.0". */
function version() {
  return native.version();
}

module.exports = {
  open,
  AsyncDatabase,
  AsyncRowIterator,
  Database,
  Result,
  RowIterator,
  FlatsError,
  InvalidArgumentError,
  QueryError,
  EngineError,
  InternalError,
  InterruptedError,
  BATCH_ROWS,
  version,
};
