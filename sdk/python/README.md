# Flats — Python

ctypes bindings over the C ABI. Python 3.8+, no third-party packages.

```sh
cargo build --release -p flats-c
export FLATS_LIBRARY=$PWD/target/release/libflats_c.so   # or copy it into sdk/python/flats/
PYTHONPATH=sdk/python python3 -c "import flats; print(flats.version())"
```

```python
import array
import flats

with flats.Database("./my-db") as db:
    db.execute("CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), year INT) "
               "WITH (capacity = 100000);")

    # Batched through the binary insert path, 10k rows per atomic statement.
    # A float32 buffer (array('f'), numpy float32) is copied in one move.
    db.insert("docs", [
        {"id": "a", "vector": array.array("f", [0.1, 0.9, 0.0]), "year": 2023},
        {"id": "b", "vector": [0.8, 0.1, 1e-5], "year": 2024},
    ])

    hits = db.search("docs", [1, 0, 0], k=5, where="year >= 2024", returning=["id", "score"])
    print(hits.columns, list(hits))       # ('id', 'score') [('b', 0.8...)]

    db.execute("UPDATE docs SET year = 2025 WHERE id = 'b';")
    db.backup("./my-db-backup")
```

Errors: `QueryError` (bad statement, unknown name), `EngineError` (duplicate
key, capacity, I/O), `InvalidArgumentError`, `InternalError` — all
`FlatsError`, with the C status in `.code`. A `Database` is safe to share
between threads; calls release the GIL.

Tests: `python3 -m unittest discover -s sdk/python/tests -t sdk/python`.
