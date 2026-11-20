// The Promise API: `await flats.open(path)` → AsyncDatabase. Work runs on
// libuv's thread pool, so the event loop keeps turning while Flats works.

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

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'flats-async-'));
}

const rows = (n) => Array.from({ length: n }, (_, i) => ({ id: `d${i}`, vector: [i, 1, 0], year: i, score: i / 2 }));

describe('async', () => {
  let dir, db;
  beforeEach(async () => {
    dir = tmpdir();
    db = await flats.open(path.join(dir, 'db'));
    await db.execute(SCHEMA);
  });
  afterEach(async () => {
    await db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('open resolves to an AsyncDatabase whose calls are promises', async () => {
    assert.ok(db instanceof flats.AsyncDatabase);
    const p = db.execute('SELECT id FROM docs;');
    assert.ok(p instanceof Promise);
    assert.deepEqual((await p).rows, []);
  });

  test('execute, insert, search and parameters', async () => {
    assert.equal(await db.insert('docs', rows(100)), 100);
    const res = await db.execute('SELECT id, year FROM docs WHERE year = ?;', [7]);
    assert.deepEqual(res.rows, [['d7', 7]]);
    const hits = await db.search('docs', [99, 1, 0], { k: 3, where: 'year < ?', params: [50], returning: ['id'] });
    assert.deepEqual(hits.rows.map((r) => r[0]), ['d49', 'd48', 'd47']);
  });

  test('query is an async iterator over chunks', async () => {
    await db.insert('docs', rows(2500));
    const all = (await db.execute('SELECT id FROM docs;')).rows;
    const it = await db.query('SELECT id FROM docs;', [], { chunkRows: 300 });
    assert.deepEqual(it.columns, ['id']);
    const streamed = [];
    for await (const row of it) streamed.push(row);
    assert.deepEqual(streamed, all);
    // Breaking out releases it.
    const it2 = await db.query('SELECT id FROM docs;', [], { chunkRows: 10 });
    for await (const _row of it2) break;
    assert.equal(it2.closed, true);
  });

  test('errors reject with the matching class', async () => {
    await assert.rejects(db.execute('SELECT id FROM missing;'), flats.QueryError);
    await db.insert('docs', rows(1));
    await assert.rejects(db.insert('docs', rows(1)), flats.EngineError);
    await assert.rejects(db.execute('SELECT id FROM docs WHERE year = ?;', [true]), flats.InvalidArgumentError);
    await assert.rejects(() => db.search('docs; DROP', [1, 0, 0]), TypeError);
  });

  test('the event loop keeps turning while searches run', async () => {
    await db.execute(
      'CREATE COLLECTION big (id INT PRIMARY KEY, vector VECTOR(256)) WITH (capacity = 200000);',
    );
    const v = new Float32Array(256).fill(0.01);
    const big = Array.from({ length: 60000 }, (_, i) => ({ id: i, vector: v }));
    await db.insert('big', big);
    let ticks = 0;
    const timer = setInterval(() => ticks++, 1);
    const t0 = process.hrtime.bigint();
    // A few at a time — fewer than there are cores — so what is measured is
    // the event loop, not the OS scheduler under oversubscription.
    for (let round = 0; round < 12; round++) {
      await Promise.all([db.search('big', v, { k: 5 }), db.search('big', v, { k: 5 })]);
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    clearInterval(timer);
    // With the work on the thread pool the 1 ms timer fires throughout; had
    // the searches run on this thread, it could not fire until they ended.
    assert.ok(ticks >= Math.min(5, Math.floor(ms / 4)), `ticks=${ticks} over ${ms.toFixed(1)} ms`);
  });

  test('backup', async () => {
    await db.insert('docs', rows(5));
    await db.backup(path.join(dir, 'copy'));
    const copy = await flats.open(path.join(dir, 'copy'));
    assert.equal((await copy.execute('SELECT id FROM docs;')).length, 5);
    await copy.close();
  });
});

describe('async lifecycle', () => {
  test('close waits for work in flight, then refuses new work', async () => {
    const dir = tmpdir();
    const db = await flats.open(path.join(dir, 'db'));
    await db.execute(SCHEMA);
    await db.insert('docs', rows(1000));
    const pending = Array.from({ length: 50 }, (_, i) => db.execute('SELECT id FROM docs WHERE year = ?;', [i]));
    const closing = db.close();
    const results = await Promise.all(pending);
    assert.ok(results.every((r) => r.length === 1), 'every in-flight call finished');
    await closing;
    assert.equal(db.closed, true);
    await assert.rejects(db.execute('SELECT id FROM docs;'), flats.FlatsError);
    await db.close(); // idempotent
    // The directory is free again.
    const again = await flats.open(path.join(dir, 'db'));
    assert.equal((await again.execute('SELECT id FROM docs;')).length, 1000);
    await again.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('open rejects for a bad path', async () => {
    const dir = tmpdir();
    const file = path.join(dir, 'f');
    fs.writeFileSync(file, 'x');
    await assert.rejects(flats.open(file), flats.FlatsError);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
