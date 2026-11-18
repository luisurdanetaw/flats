"""The Python SDK, end to end, against the real shared library.

Run from the repository root after building the C ABI:

    cargo build --release -p flats-c
    python3 -m unittest discover -s sdk/python/tests -t sdk/python

The library is found through FLATS_LIBRARY, or in the repository's target/
directory (release first, then debug).
"""

import array
import os
import struct
import tempfile
import threading
import unittest

import flats

SCHEMA = (
    "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), "
    "year INT, score FLOAT) WITH (capacity = 100000);"
)


def f32(x):
    """x rounded to the nearest float32, as Flats stores it."""
    return struct.unpack("f", struct.pack("f", x))[0]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.db = flats.Database(os.path.join(self.dir, "db"))
        self.db.execute(SCHEMA)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()


class TestExecute(Base):
    def test_a_round_trip(self):
        self.db.execute(
            "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 2024, 0.5);"
        )
        res = self.db.execute("SELECT id, year, score, vector FROM docs;")
        self.assertEqual(res.columns, ("id", "year", "score", "vector"))
        self.assertEqual(list(res), [("a", 2024, 0.5, [1.0, 0.0, 0.0])])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "a")
        self.assertEqual(res.dicts(), [{"id": "a", "year": 2024, "score": 0.5, "vector": [1.0, 0.0, 0.0]}])

    def test_a_write_returns_an_empty_result(self):
        res = self.db.execute(
            "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);"
        )
        self.assertEqual(res.columns, ())
        self.assertEqual(list(res), [])

    def test_text_round_trips_unicode(self):
        self.db.insert("docs", [{"id": "ñandú 🦤", "vector": [0, 0, 1], "year": 1, "score": 0.0}])
        self.assertEqual(self.db.execute("SELECT id FROM docs;")[0], ("ñandú 🦤",))

    def test_a_bad_statement_raises_a_query_error(self):
        with self.assertRaises(flats.QueryError) as cm:
            self.db.execute("SELECT id FROM missing;")
        self.assertIn("missing", str(cm.exception))
        self.assertEqual(cm.exception.code, 2)
        self.assertIsInstance(cm.exception, flats.FlatsError)

    def test_an_engine_refusal_raises_an_engine_error(self):
        sql = "INSERT INTO docs (id, vector, year, score) VALUES ('a', [1, 0, 0], 1, 1.0);"
        self.db.execute(sql)
        with self.assertRaises(flats.EngineError) as cm:
            self.db.execute(sql)
        self.assertIn("'a'", str(cm.exception))

    def test_non_str_sql_is_a_type_error(self):
        with self.assertRaises(TypeError):
            self.db.execute(b"SELECT id FROM docs;")


class TestInsert(Base):
    def rows(self, n, start=0):
        return [
            {"id": f"d{i}", "vector": [float(i), 1.0, 0.0], "year": i, "score": i / 2}
            for i in range(start, start + n)
        ]

    def test_insert_lands_every_row(self):
        self.assertEqual(self.db.insert("docs", self.rows(500)), 500)
        res = self.db.execute("SELECT id, year, score, vector FROM docs WHERE year = 7;")
        self.assertEqual(list(res), [("d7", 7, 3.5, [7.0, 1.0, 0.0])])
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), 500)

    def test_insert_takes_any_float_sequence_for_the_vector(self):
        rows = [
            {"id": "list", "vector": [1, 2, 3], "year": 1, "score": 0.0},
            {"id": "tuple", "vector": (1.0, 2.0, 3.0), "year": 2, "score": 0.0},
            {"id": "array", "vector": array.array("f", [1, 2, 3]), "year": 3, "score": 0.0},
            {"id": "double", "vector": array.array("d", [1, 2, 3]), "year": 4, "score": 0.0},
        ]
        self.assertEqual(self.db.insert("docs", rows), 4)
        vectors = {r[0]: r[1] for r in self.db.execute("SELECT id, vector FROM docs;")}
        for name in ("list", "tuple", "array", "double"):
            self.assertEqual(vectors[name], [1.0, 2.0, 3.0], name)

    def test_insert_is_all_or_nothing(self):
        rows = self.rows(10) + [self.rows(1)[0]]  # d0 twice
        with self.assertRaises(flats.EngineError):
            self.db.insert("docs", rows)
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), 0)

    def test_insert_refuses_bad_rows_before_committing_any(self):
        cases = [
            {"id": "x", "vector": [1, 0], "year": 1, "score": 0.0},  # dimension
            {"id": "x", "vector": [1, 0, 0], "year": "1", "score": 0.0},  # type
            {"id": "x", "vector": [1, 0, 0], "year": 1},  # missing column
            {"id": "x", "vector": [1, 0, 0], "year": 1, "score": 0.0, "nope": 1},  # unknown
            {"id": "x", "vector": [1, float("inf"), 0], "year": 1, "score": 0.0},  # inf
            {"id": "x", "vector": [1, 0, 0], "year": True, "score": 0.0},  # bool is not INT
        ]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(flats.FlatsError):
                    self.db.insert("docs", self.rows(3) + [bad])
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), 0)

    def test_insert_of_nothing_is_zero(self):
        self.assertEqual(self.db.insert("docs", []), 0)

    def test_insert_into_a_missing_collection(self):
        with self.assertRaises(flats.QueryError):
            self.db.insert("missing", self.rows(1))

    def test_a_large_load_is_batched_and_fast_enough(self):
        n = 20_000
        self.assertEqual(self.db.insert("docs", self.rows(n)), n)
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), n)


class TestSearch(Base):
    def setUp(self):
        super().setUp()
        self.db.insert(
            "docs",
            [
                {"id": f"d{i}", "vector": [float(i), 1.0, 0.0], "year": 2000 + i, "score": 0.0}
                for i in range(50)
            ],
        )

    def test_search_ranks_nearest_first(self):
        hits = self.db.search("docs", [49.0, 1.0, 0.0], k=3, returning=["id", "score"])
        self.assertEqual(hits.columns, ("id", "score"))
        self.assertEqual([h[0] for h in hits], ["d49", "d48", "d47"])

    def test_search_with_a_filter(self):
        hits = self.db.search("docs", [49.0, 1.0, 0.0], k=5, where="year < 2010", returning=["id"])
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(int(h[0][1:]) < 10 for h in hits))

    def test_tiny_components_survive_exactly(self):
        # repr() of a small float uses an exponent ('1e-07'); the engine must
        # take it, and the stored float32 must be the one we meant.
        v = [1e-7, -2.5e-30, 3.4e38]
        self.db.insert("docs", [{"id": "tiny", "vector": v, "year": 0, "score": 0.0}])
        got = self.db.execute("SELECT vector FROM docs WHERE id = 'tiny';")[0][0]
        self.assertEqual(got, [f32(x) for x in v])
        hits = self.db.search("docs", v, k=1, returning=["id"])
        self.assertEqual(len(hits), 1)

    def test_a_non_finite_query_vector_is_refused(self):
        with self.assertRaises(ValueError):
            self.db.search("docs", [float("nan"), 0, 0], k=1)


class TestLifecycle(unittest.TestCase):
    def test_context_manager_closes_and_data_survives(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "db")
            with flats.Database(path) as db:
                db.execute(SCHEMA)
                db.insert("docs", [{"id": "a", "vector": [1, 0, 0], "year": 1, "score": 0.0}])
            self.assertTrue(db.closed)
            with self.assertRaises(flats.FlatsError):
                db.execute("SELECT id FROM docs;")
            with flats.Database(path) as db:
                self.assertEqual(len(db.execute("SELECT id FROM docs;")), 1)

    def test_close_twice_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            db = flats.Database(os.path.join(d, "db"))
            db.close()
            db.close()

    def test_backup_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            with flats.Database(os.path.join(d, "db")) as db:
                db.execute(SCHEMA)
                db.insert("docs", [{"id": "a", "vector": [1, 0, 0], "year": 1, "score": 0.0}])
                db.backup(os.path.join(d, "copy"))
                with self.assertRaises(flats.EngineError):
                    db.backup(os.path.join(d, "copy"))  # not empty any more
            with flats.Database(os.path.join(d, "copy")) as copy:
                self.assertEqual(list(copy.execute("SELECT id FROM docs;")), [("a",)])

    def test_drop_collection(self):
        with tempfile.TemporaryDirectory() as d:
            with flats.Database(os.path.join(d, "db")) as db:
                db.execute(SCHEMA)
                db.execute("DROP COLLECTION docs;")
                with self.assertRaises(flats.QueryError):
                    db.execute("SELECT id FROM docs;")

    def test_open_failure_raises(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(flats.FlatsError):
                flats.Database(f.name)  # a file, not a directory

    def test_threads_share_one_database(self):
        with tempfile.TemporaryDirectory() as d:
            with flats.Database(os.path.join(d, "db")) as db:
                db.execute(SCHEMA)

                def work(t):
                    db.insert(
                        "docs",
                        [
                            {"id": f"t{t}-{i}", "vector": [1, 0, 0], "year": i, "score": 0.0}
                            for i in range(200)
                        ],
                    )
                    db.search("docs", [1, 0, 0], k=5)

                threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(len(db.execute("SELECT id FROM docs;")), 800)

    def test_version(self):
        self.assertEqual(flats.version(), "0.1.0")


class TestParams(Base):
    """Parameters are values, never syntax."""

    PAYLOADS = [
        "x'); DROP COLLECTION docs; --",
        "' OR '1'='1",
        "a' OR year > 0 OR id = 'b",
        "?",
        "'",
        "ñ 🦤\n;",
    ]

    def test_execute_fills_placeholders(self):
        self.db.execute(
            "INSERT INTO docs (id, vector, year, score) VALUES (?, ?, ?, ?);",
            ["a", [1, 0, 0], 2024, 0.5],
        )
        res = self.db.execute("SELECT id, year, score FROM docs WHERE year = ? AND id != ?;", [2024, "z"])
        self.assertEqual(list(res), [("a", 2024, 0.5)])

    def test_injection_payloads_are_plain_values(self):
        for i, p in enumerate(self.PAYLOADS):
            self.db.execute(
                "INSERT INTO docs (id, vector, year, score) VALUES (?, ?, ?, ?);",
                [p, [1, 0, 0], i, 0.0],
            )
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), len(self.PAYLOADS))
        for p in self.PAYLOADS:
            self.assertEqual(list(self.db.execute("SELECT id FROM docs WHERE id = ?;", [p])), [(p,)])
            hits = self.db.search("docs", [1, 0, 0], k=10, where="id = ?", params=[p], returning=["id"])
            self.assertEqual(list(hits), [(p,)], p)

    def test_search_passes_the_query_vector_as_a_parameter(self):
        v = [1e-7, -2.5e-30, 3.4e38]
        self.db.insert("docs", [{"id": "tiny", "vector": v, "year": 0, "score": 0.0}])
        hits = self.db.search("docs", array.array("f", v), k=1, returning=["id", "vector"])
        self.assertEqual(hits[0], ("tiny", [f32(x) for x in v]))

    def test_search_where_with_parameters(self):
        self.db.insert(
            "docs",
            [{"id": f"d{i}", "vector": [float(i), 1.0, 0.0], "year": 2000 + i, "score": 0.0} for i in range(20)],
        )
        hits = self.db.search("docs", [1, 0, 0], k=3, where="year < ? AND id != ?", params=[2010, "d9"],
                              returning=["id"])
        self.assertEqual([h[0] for h in hits], ["d8", "d7", "d6"])

    def test_a_count_mismatch_raises_a_query_error(self):
        with self.assertRaises(flats.QueryError):
            self.db.execute("SELECT id FROM docs WHERE year = ?;")
        with self.assertRaises(flats.QueryError):
            self.db.execute("SELECT id FROM docs WHERE year = ?;", [1, 2])

    def test_parameter_types(self):
        with self.assertRaises(flats.InvalidArgumentError):
            self.db.execute("SELECT id FROM docs WHERE year = ?;", [True])
        with self.assertRaises(flats.InvalidArgumentError):
            self.db.execute("SELECT id FROM docs WHERE year = ?;", [None])
        with self.assertRaises(flats.QueryError) as cm:  # NaN is no literal
            self.db.execute("SELECT id FROM docs WHERE year = ?;", [float("nan")])
        self.assertIn("finite", str(cm.exception))
        with self.assertRaises(flats.QueryError):  # TEXT against INT
            self.db.execute("SELECT id FROM docs WHERE year = ?;", ["2024"])

    def test_names_cannot_be_injected_through_search(self):
        for bad in ["docs; DROP COLLECTION docs", "docs WHERE 1", "", "1docs"]:
            with self.assertRaises(ValueError, msg=bad):
                self.db.search(bad, [1, 0, 0])
        with self.assertRaises(ValueError):
            self.db.search("docs", [1, 0, 0], returning=["id, vector FROM docs; --"])
        self.assertEqual(len(self.db.execute("SELECT id FROM docs;")), 0)


if __name__ == "__main__":
    unittest.main()
