// The Node.js SDK, end to end, against the real addon.
//
//   cargo build --release -p flats-node
//   node --test sdk/node/test/
//
// The addon is found through FLATS_NODE_ADDON, next to index.js as
// flats.node, or in the repository's target/ directory.

'use strict';

const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const flats = require('..');

const SCHEMA =
  'CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), year INT, score FLOAT) ' +
  'WITH (capacity = 100000);';

const f32 = (x) => Math.fround(x);

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'flats-node-'));
}

function rows(n, start = 0) {
  return Array.from({ length: n }, (_, j) => {
    const i = start + j;
    return { id: `d${i}`, vector: [i, 1, 0], year: i, score: i / 2 };
  });
}

describe('execute', () => {
  let dir, db;
  beforeEach(() => {
    dir = tmpdir();
    db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
  });
  afterEach(() => {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('a round trip', () => {
    db.execute("INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 2024, 0.5);");
    const res = db.execute('SELECT id, year, score, vector FROM docs;');
    assert.deepEqual(res.columns, ['id', 'year', 'score', 'vector']);
    assert.deepEqual(res.rows, [['a', 2024, 0.5, [1, 0, 0]]]);
    assert.equal(res.length, 1);
    assert.deepEqual(res.toObjects(), [{ id: 'a', year: 2024, score: 0.5, vector: [1, 0, 0] }]);
    assert.deepEqual([...res], [['a', 2024, 0.5, [1, 0, 0]]], 'iterable');
  });

  test('a write returns an empty result', () => {
    const res = db.execute("INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);");
    assert.deepEqual(res.columns, []);
    assert.deepEqual(res.rows, []);
  });

  test('integers beyond 2^53 come back as BigInt, exactly', () => {
    db.insert('docs', [{ id: 'big', vector: [0, 0, 1], year: 9007199254740993n, score: 0 }]);
    const [[year]] = db.execute("SELECT year FROM docs WHERE id = 'big';").rows;
    assert.equal(typeof year, 'bigint');
    assert.equal(year, 9007199254740993n);
  });

  test('unicode text round-trips', () => {
    db.insert('docs', [{ id: 'ñandú 🦤', vector: [0, 0, 1], year: 1, score: 0 }]);
    assert.deepEqual(db.execute('SELECT id FROM docs;').rows, [['ñandú 🦤']]);
  });

  test('a bad statement throws a QueryError', () => {
    assert.throws(() => db.execute('SELECT id FROM missing;'), (e) => {
      assert.ok(e instanceof flats.QueryError);
      assert.ok(e instanceof flats.FlatsError);
      assert.equal(e.code, 2);
      assert.match(e.message, /missing/);
      return true;
    });
  });

  test('an engine refusal throws an EngineError', () => {
    const sql = "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);";
    db.execute(sql);
    assert.throws(() => db.execute(sql), (e) => e instanceof flats.EngineError && /'a'/.test(e.message));
  });

  test('non-string sql is a TypeError', () => {
    assert.throws(() => db.execute(42), TypeError);
  });
});

describe('insert', () => {
  let dir, db;
  beforeEach(() => {
    dir = tmpdir();
    db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
  });
  afterEach(() => {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('lands every row', () => {
    assert.equal(db.insert('docs', rows(500)), 500);
    assert.deepEqual(db.execute('SELECT id, year, score, vector FROM docs WHERE year = 7;').rows, [
      ['d7', 7, 3.5, [7, 1, 0]],
    ]);
    assert.equal(db.execute('SELECT id FROM docs;').length, 500);
  });

  test('takes arrays and typed arrays for the vector', () => {
    db.insert('docs', [
      { id: 'array', vector: [1, 2, 3], year: 1, score: 0 },
      { id: 'f32', vector: new Float32Array([1, 2, 3]), year: 2, score: 0 },
      { id: 'f64', vector: new Float64Array([1, 2, 3]), year: 3, score: 0 },
    ]);
    for (const [id, vector] of db.execute('SELECT id, vector FROM docs;').rows) {
      assert.deepEqual(vector, [1, 2, 3], id);
    }
  });

  test('a whole number into a FLOAT column stays a float column value', () => {
    db.insert('docs', [{ id: 'a', vector: [1, 0, 0], year: 1, score: 3 }]);
    assert.deepEqual(db.execute('SELECT score FROM docs;').rows, [[3]]);
  });

  test('is all or nothing', () => {
    assert.throws(() => db.insert('docs', [...rows(10), rows(1)[0]]), flats.EngineError);
    assert.equal(db.execute('SELECT id FROM docs;').length, 0);
  });

  test('refuses bad rows before committing any', () => {
    const cases = [
      { id: 'x', vector: [1, 0], year: 1, score: 0 }, // dimension
      { id: 'x', vector: [1, 0, 0], year: '1', score: 0 }, // type
      { id: 'x', vector: [1, 0, 0], year: 1 }, // missing column
      { id: 'x', vector: [1, 0, 0], year: 1, score: 0, nope: 1 }, // unknown column
      { id: 'x', vector: [1, Infinity, 0], year: 1, score: 0 }, // not finite
      { id: 'x', vector: [1, 0, 0], year: 1.5, score: 0 }, // not an integer
      { id: 'x', vector: [1, 0, 0], year: true, score: 0 }, // not a V-SQL value
    ];
    for (const bad of cases) {
      assert.throws(() => db.insert('docs', [...rows(3), bad]), flats.FlatsError, JSON.stringify(bad));
    }
    assert.equal(db.execute('SELECT id FROM docs;').length, 0);
  });

  test('of nothing is zero', () => {
    assert.equal(db.insert('docs', []), 0);
  });

  test('into a missing collection throws a QueryError', () => {
    assert.throws(() => db.insert('missing', rows(1)), flats.QueryError);
  });
});

describe('search', () => {
  let dir, db;
  beforeEach(() => {
    dir = tmpdir();
    db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
    db.insert(
      'docs',
      Array.from({ length: 50 }, (_, i) => ({ id: `d${i}`, vector: [i, 1, 0], year: 2000 + i, score: 0 })),
    );
  });
  afterEach(() => {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('ranks nearest first', () => {
    const hits = db.search('docs', [49, 1, 0], { k: 3, returning: ['id', 'score'] });
    assert.deepEqual(hits.columns, ['id', 'score']);
    assert.deepEqual(hits.rows.map((r) => r[0]), ['d49', 'd48', 'd47']);
  });

  test('with a filter', () => {
    const hits = db.search('docs', [49, 1, 0], { k: 5, where: 'year < 2010', returning: ['id'] });
    assert.equal(hits.length, 5);
    assert.ok(hits.rows.every(([id]) => Number(id.slice(1)) < 10));
  });

  test('tiny components survive exactly', () => {
    // String(1e-7) is '1e-7': the engine must take the exponent.
    const v = [1e-7, -2.5e-30, 3.4e38];
    db.insert('docs', [{ id: 'tiny', vector: v, year: 0, score: 0 }]);
    const [[got]] = db.execute("SELECT vector FROM docs WHERE id = 'tiny';").rows;
    assert.deepEqual(got, v.map(f32));
    assert.equal(db.search('docs', v, { k: 1 }).length, 1);
  });

  test('a non-finite query vector throws a RangeError', () => {
    assert.throws(() => db.search('docs', [NaN, 0, 0]), RangeError);
  });
});

describe('lifecycle', () => {
  test('close, reopen, data survives; a closed database throws', () => {
    const dir = tmpdir();
    const p = path.join(dir, 'db');
    const db = new flats.Database(p);
    db.execute(SCHEMA);
    db.insert('docs', rows(3));
    db.close();
    assert.equal(db.closed, true);
    assert.throws(() => db.execute('SELECT id FROM docs;'), flats.FlatsError);
    db.close(); // idempotent
    const again = new flats.Database(p);
    assert.equal(again.execute('SELECT id FROM docs;').length, 3);
    again.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('backup and restore', () => {
    const dir = tmpdir();
    const db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
    db.insert('docs', rows(4));
    db.backup(path.join(dir, 'copy'));
    assert.throws(() => db.backup(path.join(dir, 'copy')), flats.EngineError, 'not empty any more');
    db.close();
    const copy = new flats.Database(path.join(dir, 'copy'));
    assert.equal(copy.execute('SELECT id FROM docs;').length, 4);
    copy.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('drop collection', () => {
    const dir = tmpdir();
    const db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
    db.execute('DROP COLLECTION docs;');
    assert.throws(() => db.execute('SELECT id FROM docs;'), flats.QueryError);
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('opening a file instead of a directory throws', () => {
    const dir = tmpdir();
    const file = path.join(dir, 'not-a-dir');
    fs.writeFileSync(file, 'x');
    assert.throws(() => new flats.Database(file), flats.FlatsError);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('an unclosed database is released by the garbage collector', () => {
    // No assertion beyond "does not crash": the finalizer must close the
    // handle exactly once even though close() was never called.
    const dir = tmpdir();
    (() => {
      const db = new flats.Database(path.join(dir, 'db'));
      db.execute(SCHEMA);
    })();
    if (global.gc) global.gc();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('version', () => {
    assert.equal(flats.version(), '0.1.0');
  });
});

describe('parameters', () => {
  let dir, db;
  beforeEach(() => {
    dir = tmpdir();
    db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
  });
  afterEach(() => {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const PAYLOADS = [
    "x'); DROP COLLECTION docs; --",
    "' OR '1'='1",
    "a' OR year > 0 OR id = 'b",
    '?',
    "'",
    'ñ 🦤\n;',
  ];

  test('execute fills placeholders', () => {
    db.execute('INSERT INTO docs (id, vector, year, score) VALUES (?, ?, ?, ?);', ['a', [1, 0, 0], 2024, 0.5]);
    assert.deepEqual(db.execute('SELECT id, year, score FROM docs WHERE year = ? AND id != ?;', [2024, 'z']).rows, [
      ['a', 2024, 0.5],
    ]);
  });

  test('injection payloads are plain values', () => {
    PAYLOADS.forEach((p, i) =>
      db.execute('INSERT INTO docs (id, vector, year, score) VALUES (?, ?, ?, ?);', [p, [1, 0, 0], i, 0]),
    );
    assert.equal(db.execute('SELECT id FROM docs;').length, PAYLOADS.length);
    for (const p of PAYLOADS) {
      assert.deepEqual(db.execute('SELECT id FROM docs WHERE id = ?;', [p]).rows, [[p]]);
      const hits = db.search('docs', [1, 0, 0], { k: 10, where: 'id = ?', params: [p], returning: ['id'] });
      assert.deepEqual(hits.rows, [[p]], p);
    }
  });

  test('a whole number is an INT, a fraction a FLOAT, a BigInt an exact INT', () => {
    db.execute('INSERT INTO docs (id, vector, year, score) VALUES (?, ?, ?, ?);', [
      'a',
      new Float32Array([1, 0, 0]),
      9007199254740993n,
      3, // a whole number into a FLOAT column: coerced, as a literal 3 would be
    ]);
    assert.deepEqual(db.execute('SELECT year, score FROM docs WHERE year = ?;', [9007199254740993n]).rows, [
      [9007199254740993n, 3],
    ]);
    assert.throws(() => db.execute('SELECT id FROM docs WHERE year = ?;', [1.5]), flats.QueryError);
  });

  test('search passes the query vector as a parameter', () => {
    const v = [1e-7, -2.5e-30, 3.4e38];
    db.insert('docs', [{ id: 'tiny', vector: v, year: 0, score: 0 }]);
    const hits = db.search('docs', new Float32Array(v), { k: 1, returning: ['id', 'vector'] });
    assert.deepEqual(hits.rows[0], ['tiny', v.map(f32)]);
  });

  test('search where with parameters', () => {
    db.insert('docs', Array.from({ length: 20 }, (_, i) => ({ id: `d${i}`, vector: [i, 1, 0], year: 2000 + i, score: 0 })));
    const hits = db.search('docs', [1, 0, 0], { k: 3, where: 'year < ? AND id != ?', params: [2010, 'd9'], returning: ['id'] });
    assert.deepEqual(hits.rows.map((r) => r[0]), ['d8', 'd7', 'd6']);
  });

  test('count mismatches and bad values', () => {
    assert.throws(() => db.execute('SELECT id FROM docs WHERE year = ?;'), flats.QueryError);
    assert.throws(() => db.execute('SELECT id FROM docs WHERE year = ?;', [1, 2]), flats.QueryError);
    assert.throws(() => db.execute('SELECT id FROM docs WHERE year = ?;', [NaN]), flats.QueryError);
    for (const bad of [true, null, undefined, {}, () => 1]) {
      assert.throws(() => db.execute('SELECT id FROM docs WHERE year = ?;', [bad]), flats.InvalidArgumentError, String(bad));
    }
    assert.throws(() => db.execute('SELECT id FROM docs;', 'not an array'), TypeError);
  });

  test('names cannot be injected through search or insert', () => {
    for (const bad of ['docs; DROP COLLECTION docs', 'docs WHERE 1', '', '1docs']) {
      assert.throws(() => db.search(bad, [1, 0, 0]), TypeError, bad);
    }
    assert.throws(() => db.search('docs', [1, 0, 0], { returning: ['id, vector FROM docs; --'] }), TypeError);
    assert.equal(db.execute('SELECT id FROM docs;').length, 0);
  });
});

describe('locking', () => {
  test('a second open of the same directory is refused', () => {
    const dir = tmpdir();
    const p = path.join(dir, 'db');
    const first = new flats.Database(p);
    assert.throws(() => new flats.Database(p), (e) => e instanceof flats.EngineError && /locked/.test(e.message));
    first.close();
    new flats.Database(p).close(); // free again once closed
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
