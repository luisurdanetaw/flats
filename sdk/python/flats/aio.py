"""Flats for asyncio.

The same database, with every call run on a worker thread so the event loop
keeps turning while Flats works (each call releases the GIL)::

    import flats.aio

    db = await flats.aio.open("./my-db")
    hits = await db.search("docs", query_vector, k=10, where="lang = ?", params=["en"])
    async with await db.query("SELECT id FROM docs;") as rows:
        async for (doc_id,) in rows:
            ...
    await db.close()

Methods mirror :class:`flats.Database` one for one and take the same
arguments; they return awaitables. ``close()`` waits for calls in flight.
"""

import asyncio
import functools

from . import Database

__all__ = ["open", "AsyncDatabase", "AsyncCursor"]


async def _in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


async def open(path, *, salvage=False):  # noqa: A001 — mirrors flats.Database(path)
    """Open (creating if missing) the database in ``path``. ``salvage`` is as
    for ``flats.Database``."""
    return AsyncDatabase(await _in_thread(Database, path, salvage=salvage))


class AsyncCursor:
    """A statement's rows, read in chunks on a worker thread: ``async for``.
    Leaving ``async with``, breaking out, or ``close()`` releases it."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.columns = cursor.columns
        self._buffer = []
        self._at = 0

    @property
    def closed(self):
        return self._cursor.closed

    def close(self):
        self._buffer, self._at = [], 0
        self._cursor.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._at >= len(self._buffer):
            if self._cursor.closed:
                raise StopAsyncIteration
            self._buffer, self._at = await _in_thread(self._cursor._take_chunk), 0
            if not self._buffer:
                raise StopAsyncIteration
        row = self._buffer[self._at]
        self._at += 1
        return row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()


class AsyncDatabase:
    """An open database whose calls are awaitables. Safe to use from many
    tasks at once."""

    def __init__(self, db):
        self._db = db

    @property
    def closed(self):
        return self._db.closed

    async def close(self):
        """Wait for calls in flight, then checkpoint and close. Idempotent."""
        await _in_thread(self._db.close)

    async def execute(self, sql, params=()):
        return await _in_thread(self._db.execute, sql, params)

    async def query(self, sql, params=(), *, chunk_rows=1000):
        return AsyncCursor(await _in_thread(self._db.query, sql, params, chunk_rows=chunk_rows))

    async def search(self, collection, vector, k=10, *, where=None, params=(), returning=None):
        return await _in_thread(
            self._db.search, collection, vector, k, where=where, params=params, returning=returning
        )

    async def insert(self, collection, rows):
        return await _in_thread(self._db.insert, collection, rows)

    async def backup(self, dest):
        return await _in_thread(self._db.backup, dest)

    def interrupt(self):
        """Stop every statement running now (not a coroutine: it never
        waits)."""
        self._db.interrupt()

    def set_timeout(self, seconds):
        """See :meth:`flats.Database.set_timeout`."""
        self._db.set_timeout(seconds)
