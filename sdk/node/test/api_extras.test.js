// upsert, get, delete by key, collections and describe.

'use strict';

const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const flats = require('..');

const SCHEMA =
  'CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(2), n INT, f FLOAT) WITH (capacity = 1000);';

describe('api extras', () => {
  let dir, db;
  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'flats-extras-'));
    db = new flats.Database(path.join(dir, 'db'));
    db.execute(SCHEMA);
    db.insert('docs', [{ id: 'a', vector: [1, 0], n: 1, f: 0.5 }]);
  });
  afterEach(() => {
    if (!db.closed) db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('upsert inserts and replaces', () => {
    const n = db.upsert('docs', [
      { id: 'a', vector: [0, 1], n: 10, f: 1 },
      { id: 'b', vector: [1, 1], n: 20, f: 2 },
    ]);
    assert.equal(n, 2);
    assert.deepEqual(db.get('docs', 'a'), { id: 'a', vector: [0, 1], n: 10, f: 1 });
    assert.equal(db.get('docs', 'b').n, 20);
    assert.equal(db.execute('SELECT id FROM docs;').length, 2);
  });

  test('a key repeated in one upsert is refused', () => {
    assert.throws(
      () =>
        db.upsert('docs', [
          { id: 'z', vector: [0, 1], n: 1, f: 1 },
          { id: 'z', vector: [1, 1], n: 2, f: 2 },
        ]),
      flats.EngineError,
    );
    assert.equal(db.get('docs', 'z'), null);
  });

  test('get and delete by key', () => {
    assert.equal(db.get('docs', 'nope'), null);
    db.delete('docs', 'nope'); // fine
    db.delete('docs', 'a');
    assert.equal(db.get('docs', 'a'), null);
    db.execute('CREATE COLLECTION nums (k INT PRIMARY KEY, vector VECTOR(1)) WITH (capacity = 10);');
    db.insert('nums', [{ k: 7, vector: [1] }]);
    assert.deepEqual(db.get('nums', 7), { k: 7, vector: [1] });
  });

  test('a key is a value, never statement text', () => {
    assert.equal(db.get('docs', "a' OR '1'='1"), null);
    db.delete('docs', "a' OR '1'='1");
    assert.notEqual(db.get('docs', 'a'), null);
  });

  test('collections and describe', () => {
    db.execute('CREATE COLLECTION other (k INT PRIMARY KEY, emb VECTOR(3)) WITH (capacity = 7);');
    assert.deepEqual(
      db.collections().sort((x, y) => x.name.localeCompare(y.name)),
      [
        { name: 'docs', dimension: 2, capacity: 1000, rows: 1, primaryKey: 'id' },
        { name: 'other', dimension: 3, capacity: 7, rows: 0, primaryKey: 'k' },
      ],
    );
    assert.deepEqual(db.describe('docs'), {
      name: 'docs',
      dimension: 2,
      capacity: 1000,
      rows: 1,
      primaryKey: 'id',
      columns: [
        { name: 'id', type: 'TEXT', primaryKey: true },
        { name: 'vector', type: 'VECTOR(2)', primaryKey: false },
        { name: 'n', type: 'INT', primaryKey: false },
        { name: 'f', type: 'FLOAT', primaryKey: false },
      ],
    });
    assert.throws(() => db.describe('nope'), flats.QueryError);
    assert.throws(() => db.get('docs; DROP', 'a'), TypeError);
  });

  test('the Promise API mirrors them', async () => {
    db.close();
    const adb = await flats.open(path.join(dir, 'db'));
    await adb.upsert('docs', [{ id: 'a', vector: [0, 1], n: 5, f: 0 }]);
    assert.equal((await adb.get('docs', 'a')).n, 5);
    await adb.delete('docs', 'a');
    assert.equal(await adb.get('docs', 'a'), null);
    assert.deepEqual((await adb.collections()).map((c) => c.name), ['docs']);
    assert.equal((await adb.describe('docs')).primaryKey, 'id');
    await adb.close();
  });
});
