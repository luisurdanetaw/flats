"""flats.aio: the asyncio API. Every call runs on a worker thread, so the
event loop keeps turning while Flats works."""

import asyncio
import os
import tempfile
import time
import unittest

import flats
import flats.aio

SCHEMA = (
    "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), year INT, score FLOAT) "
    "WITH (capacity = 100000);"
)


def rows(n):
    return [{"id": f"d{i}", "vector": [float(i), 1.0, 0.0], "year": i, "score": i / 2} for i in range(n)]


class TestAio(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.db = await flats.aio.open(os.path.join(self.dir, "db"))
        await self.db.execute(SCHEMA)

    async def asyncTearDown(self):
        await self.db.close()
        self._tmp.cleanup()

    async def test_execute_insert_search(self):
        self.assertEqual(await self.db.insert("docs", rows(100)), 100)
        res = await self.db.execute("SELECT id, year FROM docs WHERE year = ?;", [7])
        self.assertEqual(list(res), [("d7", 7)])
        hits = await self.db.search("docs", [99, 1, 0], k=3, where="year < ?", params=[50], returning=["id"])
        self.assertEqual([h[0] for h in hits], ["d49", "d48", "d47"])

    async def test_query_is_an_async_iterator(self):
        await self.db.insert("docs", rows(2500))
        everything = list(await self.db.execute("SELECT id FROM docs;"))
        cur = await self.db.query("SELECT id FROM docs;", chunk_rows=300)
        self.assertEqual(cur.columns, ("id",))
        streamed = [row async for row in cur]
        self.assertEqual(streamed, everything)
        async with await self.db.query("SELECT id FROM docs;", chunk_rows=10) as cur2:
            async for _ in cur2:
                break
        self.assertTrue(cur2.closed)

    async def test_errors_raise_the_matching_class(self):
        with self.assertRaises(flats.QueryError):
            await self.db.execute("SELECT id FROM missing;")
        await self.db.insert("docs", rows(1))
        with self.assertRaises(flats.EngineError):
            await self.db.insert("docs", rows(1))
        with self.assertRaises(ValueError):
            await self.db.search("docs; DROP", [1, 0, 0])

    async def test_the_event_loop_keeps_turning(self):
        await self.db.execute("CREATE COLLECTION big (id INT PRIMARY KEY, vector VECTOR(256)) WITH (capacity = 200000);")
        import array
        v = array.array("f", [0.01] * 256)
        await self.db.insert("big", ({"id": i, "vector": v} for i in range(150000)))
        ticks = 0
        stop = False

        async def ticker():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.001)

        t = asyncio.ensure_future(ticker())
        t0 = time.monotonic()
        # A few at a time — fewer than there are cores — so what is measured
        # is the GIL and the loop, not the OS scheduler under oversubscription.
        for _ in range(10):
            await asyncio.gather(*(self.db.search("big", v, k=5) for _ in range(2)))
        elapsed = time.monotonic() - t0
        stop = True
        await t
        # Run on the loop's own thread, the searches would leave the ticker
        # one or two turns in total; on worker threads it keeps ticking.
        self.assertGreater(elapsed, 0.03, "the workload must outlast the ticker's resolution")
        self.assertGreaterEqual(ticks, int(elapsed / 0.005), f"ticks={ticks} over {elapsed:.3f}s")

    async def test_backup(self):
        await self.db.insert("docs", rows(5))
        await self.db.backup(os.path.join(self.dir, "copy"))
        copy = await flats.aio.open(os.path.join(self.dir, "copy"))
        self.assertEqual(len(await copy.execute("SELECT id FROM docs;")), 5)
        await copy.close()


class TestAioLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_close_waits_for_work_in_flight(self):
        with tempfile.TemporaryDirectory() as d:
            db = await flats.aio.open(os.path.join(d, "db"))
            await db.execute(SCHEMA)
            await db.insert("docs", rows(1000))
            pending = [
                asyncio.ensure_future(db.execute("SELECT id FROM docs WHERE year = ?;", [i])) for i in range(50)
            ]
            await asyncio.sleep(0)
            await db.close()
            results = await asyncio.gather(*pending, return_exceptions=True)
            for r in results:
                # Each call either finished before close took the handle or
                # was refused cleanly — never a crash.
                self.assertTrue(isinstance(r, flats.FlatsError) or len(r) == 1, r)
            self.assertTrue(db.closed)
            with self.assertRaises(flats.FlatsError):
                await db.execute("SELECT id FROM docs;")
            await db.close()

    async def test_open_failure_raises(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(flats.FlatsError):
                await flats.aio.open(f.name)


if __name__ == "__main__":
    unittest.main()
