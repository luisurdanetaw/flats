"""upsert, get, delete by key, collections and describe. Run like test_flats.py."""

import asyncio
import os
import tempfile
import unittest

import flats
import flats.aio

SCHEMA = (
    "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(2), n INT, "
    "f FLOAT) WITH (capacity = 1000);"
)


class Extras(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = flats.Database(os.path.join(self._tmp.name, "db"))
        self.db.execute(SCHEMA)
        self.db.insert("docs", [{"id": "a", "vector": [1, 0], "n": 1, "f": 0.5}])

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_upsert_inserts_and_replaces(self):
        n = self.db.upsert("docs", [
            {"id": "a", "vector": [0, 1], "n": 10, "f": 1.0},
            {"id": "b", "vector": [1, 1], "n": 20, "f": 2},
        ])
        self.assertEqual(n, 2)
        self.assertEqual(self.db.get("docs", "a"),
                         {"id": "a", "vector": [0.0, 1.0], "n": 10, "f": 1.0})
        self.assertEqual(self.db.get("docs", "b")["n"], 20)
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), 2)

    def test_upsert_refuses_a_key_repeated_in_one_batch(self):
        with self.assertRaises(flats.EngineError):
            self.db.upsert("docs", [
                {"id": "z", "vector": [0, 1], "n": 1, "f": 1.0},
                {"id": "z", "vector": [1, 1], "n": 2, "f": 2.0},
            ])
        self.assertIsNone(self.db.get("docs", "z"))

    def test_get_returns_none_for_a_missing_key(self):
        self.assertIsNone(self.db.get("docs", "nope"))
        self.assertEqual(self.db.get("docs", "a")["vector"], [1.0, 0.0])

    def test_get_and_delete_by_an_int_key(self):
        self.db.execute("CREATE COLLECTION nums (k INT PRIMARY KEY, vector VECTOR(1)) WITH (capacity = 10);")
        self.db.insert("nums", [{"k": 7, "vector": [1]}, {"k": 8, "vector": [2]}])
        self.assertEqual(self.db.get("nums", 7), {"k": 7, "vector": [1.0]})
        self.db.delete("nums", 7)
        self.assertIsNone(self.db.get("nums", 7))
        self.assertIsNotNone(self.db.get("nums", 8))

    def test_delete_of_a_missing_key_is_fine(self):
        self.db.delete("docs", "nope")
        self.assertIsNotNone(self.db.get("docs", "a"))

    def test_a_key_is_a_value_never_text(self):
        # The key goes in as a parameter: a quote can't change the statement.
        self.assertIsNone(self.db.get("docs", "a' OR '1'='1"))
        self.db.delete("docs", "a' OR '1'='1")
        self.assertIsNotNone(self.db.get("docs", "a"))

    def test_collections_and_describe(self):
        self.db.execute("CREATE COLLECTION other (k INT PRIMARY KEY, emb VECTOR(3)) WITH (capacity = 7);")
        self.assertEqual(
            sorted(self.db.collections(), key=lambda c: c["name"]),
            [
                {"name": "docs", "dimension": 2, "capacity": 1000, "rows": 1, "primary_key": "id"},
                {"name": "other", "dimension": 3, "capacity": 7, "rows": 0, "primary_key": "k"},
            ],
        )
        self.assertEqual(self.db.describe("docs"), {
            "name": "docs", "dimension": 2, "capacity": 1000, "rows": 1, "primary_key": "id",
            "columns": [
                {"name": "id", "type": "TEXT", "primary_key": True},
                {"name": "vector", "type": "VECTOR(2)", "primary_key": False},
                {"name": "n", "type": "INT", "primary_key": False},
                {"name": "f", "type": "FLOAT", "primary_key": False},
            ],
        })
        with self.assertRaises(flats.QueryError):
            self.db.describe("nope")

    def test_names_are_checked(self):
        with self.assertRaises(ValueError):
            self.db.get("docs; DROP", "a")
        with self.assertRaises(ValueError):
            self.db.upsert("bad name", [])

    def test_aio_mirrors_them(self):
        path = os.path.join(self._tmp.name, "async")

        async def go():
            db = await flats.aio.open(path)
            await db.execute(SCHEMA)
            await db.upsert("docs", [{"id": "a", "vector": [1, 0], "n": 1, "f": 0.5}])
            await db.upsert("docs", [{"id": "a", "vector": [0, 1], "n": 2, "f": 0.5}])
            got = await db.get("docs", "a")
            await db.delete("docs", "a")
            gone = await db.get("docs", "a")
            names = [c["name"] for c in await db.collections()]
            desc = await db.describe("docs")
            await db.close()
            return got, gone, names, desc["primary_key"]

        got, gone, names, pk = asyncio.run(go())
        self.assertEqual(got["n"], 2)
        self.assertIsNone(gone)
        self.assertEqual(names, ["docs"])
        self.assertEqual(pk, "id")


if __name__ == "__main__":
    unittest.main()
