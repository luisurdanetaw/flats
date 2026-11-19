"""A database with a damaged store file refuses to open; salvage=True gets
past it. Run like test_flats.py."""

import asyncio
import os
import tempfile
import unittest

import flats
import flats.aio


def damaged_db(root):
    """Ten checkpointed rows, then the tuple store damaged."""
    path = os.path.join(root, "db")
    with flats.Database(path) as db:
        db.execute(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) "
            "WITH (capacity = 100);"
        )
        db.insert("t", ({"id": i, "vector": [1, 0]} for i in range(10)))
    snap = os.path.join(path, "collection-0", "tuples.snap")
    with open(snap, "r+b") as f:
        data = bytearray(f.read())
        data[len(data) // 2] ^= 0xFF
        f.seek(0)
        f.write(data)
    return path


class Salvage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = damaged_db(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_open_refuses_with_an_engine_error_naming_the_file(self):
        with self.assertRaises(flats.EngineError) as caught:
            flats.Database(self.path)
        self.assertIn("tuples.snap", str(caught.exception))
        self.assertIn("salvage", str(caught.exception))

    def test_salvage_opens_and_the_database_is_healthy_after(self):
        with flats.Database(self.path, salvage=True) as db:
            self.assertEqual(len(db.execute("SELECT id FROM t;")), 0)
            db.insert("t", [{"id": 1, "vector": [1, 0]}])
        self.assertTrue(
            os.path.exists(os.path.join(self.path, "collection-0", "tuples.snap.damaged"))
        )
        with flats.Database(self.path) as db:
            self.assertEqual(list(db.execute("SELECT id FROM t;")), [(1,)])

    def test_salvage_is_keyword_only(self):
        with self.assertRaises(TypeError):
            flats.Database(self.path, True)

    def test_aio_open_takes_salvage_too(self):
        async def go():
            with self.assertRaises(flats.EngineError):
                await flats.aio.open(self.path)
            db = await flats.aio.open(self.path, salvage=True)
            res = await db.execute("SELECT id FROM t;")
            await db.close()
            return len(res)

        self.assertEqual(asyncio.run(go()), 0)


if __name__ == "__main__":
    unittest.main()
