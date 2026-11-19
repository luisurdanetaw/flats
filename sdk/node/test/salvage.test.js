// A database with a damaged store file refuses to open; `{ salvage: true }`
// gets past it.

'use strict';

const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const flats = require('..');

/** Ten checkpointed rows, then the tuple store damaged. */
function damagedDb() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'flats-salvage-'));
  const dbPath = path.join(root, 'db');
  const db = new flats.Database(dbPath);
  db.execute('CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);');
  db.insert('t', Array.from({ length: 10 }, (_, i) => ({ id: i, vector: [1, 0] })));
  db.close();
  const snap = path.join(dbPath, 'collection-0', 'tuples.snap');
  const bytes = fs.readFileSync(snap);
  bytes[bytes.length >> 1] ^= 0xff;
  fs.writeFileSync(snap, bytes);
  return { root, dbPath };
}

describe('salvage', () => {
  let root, dbPath;
  beforeEach(() => ({ root, dbPath } = damagedDb()));
  afterEach(() => fs.rmSync(root, { recursive: true, force: true }));

  test('open refuses with an EngineError naming the file', () => {
    assert.throws(
      () => new flats.Database(dbPath),
      (e) => e instanceof flats.EngineError && /tuples\.snap/.test(e.message) && /salvage/.test(e.message),
    );
  });

  test('{ salvage: true } opens it, and the database is healthy after', () => {
    const db = new flats.Database(dbPath, { salvage: true });
    assert.equal(db.execute('SELECT id FROM t;').length, 0);
    db.insert('t', [{ id: 1, vector: [1, 0] }]);
    db.close();
    assert.ok(fs.existsSync(path.join(dbPath, 'collection-0', 'tuples.snap.damaged')));
    const again = new flats.Database(dbPath);
    assert.deepEqual(again.execute('SELECT id FROM t;').rows, [[1]]);
    again.close();
  });

  test('{ salvage: false } and no options are a plain open', () => {
    assert.throws(() => new flats.Database(dbPath, { salvage: false }), flats.EngineError);
    assert.throws(() => new flats.Database(dbPath, {}), flats.EngineError);
  });

  test('options are checked', () => {
    assert.throws(() => new flats.Database(dbPath, { salvage: 'yes' }), TypeError);
    assert.throws(() => new flats.Database(dbPath, true), TypeError);
  });

  test('flats.open takes the same option', async () => {
    await assert.rejects(flats.open(dbPath), flats.EngineError);
    const db = await flats.open(dbPath, { salvage: true });
    assert.equal((await db.execute('SELECT id FROM t;')).length, 0);
    await db.close();
    await assert.rejects(flats.open(dbPath, { salvage: 1 }), TypeError);
  });
});
