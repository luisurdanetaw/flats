"""Flats — a tiny embeddable vector database. Python bindings.

A thin ctypes layer over the C ABI (``sdk/c/include/flats.h``). No third-party
packages; the only requirement is the shared library ``libflats_c.so``.

    import flats

    with flats.Database("./my-db") as db:
        db.execute(
            "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), title TEXT) "
            "WITH (capacity = 10000);"
        )
        db.insert("docs", [{"id": "d1", "vector": [0.1, 0.2, 0.3], "title": "hello"}])
        for row in db.search("docs", [0.1, 0.2, 0.3], k=5, returning=["id", "score"]):
            print(row)

Finding the library: ``FLATS_LIBRARY`` (a path to the .so) wins; otherwise the
package directory, then a Flats checkout's ``target/release`` and
``target/debug``.
"""

import ctypes
import math
import os
import re
from collections.abc import Mapping, Sequence

__all__ = [
    "Database",
    "Result",
    "Cursor",
    "FlatsError",
    "InvalidArgumentError",
    "QueryError",
    "EngineError",
    "InternalError",
    "version",
]

# ---------------------------------------------------------------------------
# Errors — one class per status code in flats.h
# ---------------------------------------------------------------------------


class FlatsError(Exception):
    """Base class for every error Flats raises. ``code`` is the C status."""

    code = None

    def __init__(self, message, code=None):
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidArgumentError(FlatsError):
    """A value the schema refuses, or a malformed argument (FLATS_ERR_INVALID_ARGUMENT)."""

    code = 1


class QueryError(FlatsError):
    """The statement did not parse or bind, or named something missing (FLATS_ERR_QUERY)."""

    code = 2


class EngineError(FlatsError):
    """The engine refused or failed: duplicate key, capacity, I/O (FLATS_ERR_ENGINE)."""

    code = 3


class InternalError(FlatsError):
    """A bug in Flats (FLATS_ERR_INTERNAL)."""

    code = 4


_ERRORS = {1: InvalidArgumentError, 2: QueryError, 3: EngineError, 4: InternalError}

_INT, _FLOAT, _TEXT, _VECTOR = 1, 2, 3, 4

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------


def _candidates():
    env = os.environ.get("FLATS_LIBRARY")
    if env:
        yield env
        return
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "libflats_c.so")
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for profile in ("release", "debug"):
        yield os.path.join(root, "target", profile, "libflats_c.so")


def _load():
    tried = []
    for path in _candidates():
        if os.path.exists(path):
            return _declare(ctypes.CDLL(path))
        tried.append(path)
    raise ImportError(
        "flats: libflats_c.so not found (build it with `cargo build --release -p flats-c`, "
        "or set FLATS_LIBRARY). Looked in: " + ", ".join(tried)
    )


class _Db(ctypes.Structure):
    pass


class _Result(ctypes.Structure):
    pass


class _Batch(ctypes.Structure):
    pass


class _Cursor(ctypes.Structure):
    pass


class _Param(ctypes.Structure):
    """flats_param, field for field."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("int_value", ctypes.c_int64),
        ("float_value", ctypes.c_double),
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("vector_len", ctypes.c_size_t),
    ]


def _declare(lib):
    c_int, c_size, c_char_p = ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p
    db_p, res_p, batch_p = (ctypes.POINTER(t) for t in (_Db, _Result, _Batch))
    size_p = ctypes.POINTER(c_size)
    sigs = {
        "flats_version": (c_char_p, []),
        "flats_last_error": (c_char_p, []),
        "flats_open": (c_int, [c_char_p, ctypes.POINTER(db_p)]),
        "flats_close": (c_int, [db_p]),
        "flats_execute": (c_int, [db_p, c_char_p, ctypes.POINTER(res_p)]),
        "flats_execute_params": (
            c_int,
            [db_p, c_char_p, ctypes.POINTER(_Param), c_size, ctypes.POINTER(res_p)],
        ),
        "flats_backup": (c_int, [db_p, c_char_p]),
        "flats_query": (
            c_int,
            [db_p, c_char_p, ctypes.POINTER(_Param), c_size, ctypes.POINTER(ctypes.POINTER(_Cursor))],
        ),
        "flats_cursor_next": (c_int, [ctypes.POINTER(_Cursor), c_size, ctypes.POINTER(res_p)]),
        "flats_cursor_free": (None, [ctypes.POINTER(_Cursor)]),
        "flats_result_column_count": (c_size, [res_p]),
        "flats_result_column_name": (c_char_p, [res_p, c_size]),
        "flats_result_row_count": (c_size, [res_p]),
        "flats_result_type": (c_int, [res_p, c_size, c_size]),
        "flats_result_int": (ctypes.c_int64, [res_p, c_size, c_size]),
        "flats_result_float": (ctypes.c_double, [res_p, c_size, c_size]),
        "flats_result_text": (ctypes.c_void_p, [res_p, c_size, c_size, size_p]),
        "flats_result_vector": (ctypes.POINTER(ctypes.c_float), [res_p, c_size, c_size, size_p]),
        "flats_result_free": (None, [res_p]),
        "flats_batch_new": (c_int, [db_p, c_char_p, ctypes.POINTER(batch_p)]),
        "flats_batch_vector": (c_int, [batch_p, ctypes.POINTER(ctypes.c_float), c_size]),
        "flats_batch_int": (c_int, [batch_p, c_char_p, ctypes.c_int64]),
        "flats_batch_float": (c_int, [batch_p, c_char_p, ctypes.c_double]),
        "flats_batch_text": (c_int, [batch_p, c_char_p, ctypes.c_char_p, c_size]),
        "flats_batch_end_row": (c_int, [batch_p]),
        "flats_batch_len": (c_size, [batch_p]),
        "flats_batch_commit": (c_int, [batch_p, size_p]),
        "flats_batch_free": (None, [batch_p]),
    }
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


_lib = _load()


def _check(rc):
    if rc != 0:
        msg = _lib.flats_last_error()
        message = msg.decode("utf-8", "replace") if msg else f"flats error {rc}"
        raise _ERRORS.get(rc, FlatsError)(message, rc)


def _utf8(s, what):
    if not isinstance(s, str):
        raise TypeError(f"{what} must be str, not {type(s).__name__}")
    return s.encode("utf-8")


def version():
    """The library version, e.g. ``"0.1.0"``."""
    return _lib.flats_version().decode()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Result(Sequence):
    """The rows of one statement: a sequence of tuples, with ``columns``.

    Cells are ``int``, ``float``, ``str``, or ``list[float]`` (an embedding).
    A write statement returns an empty result with no columns.
    """

    __slots__ = ("columns", "_rows")

    def __init__(self, columns, rows):
        self.columns = tuple(columns)
        self._rows = rows

    def __getitem__(self, i):
        return self._rows[i]

    def __len__(self):
        return len(self._rows)

    def dicts(self):
        """Every row as a ``{column: value}`` dict."""
        return [dict(zip(self.columns, row)) for row in self._rows]

    def __repr__(self):
        return f"Result(columns={self.columns!r}, rows={len(self._rows)})"


def _collect(res):
    columns, rows = _columns_and_rows(res)
    return Result(columns, rows)


def _columns_and_rows(res):
    lib = _lib
    ncols = lib.flats_result_column_count(res)
    columns = [lib.flats_result_column_name(res, c).decode("utf-8") for c in range(ncols)]
    n = ctypes.c_size_t()
    rows = []
    for r in range(lib.flats_result_row_count(res)):
        row = []
        for c in range(ncols):
            t = lib.flats_result_type(res, r, c)
            if t == _INT:
                row.append(lib.flats_result_int(res, r, c))
            elif t == _FLOAT:
                row.append(lib.flats_result_float(res, r, c))
            elif t == _TEXT:
                p = lib.flats_result_text(res, r, c, ctypes.byref(n))
                row.append(ctypes.string_at(p, n.value).decode("utf-8"))
            elif t == _VECTOR:
                p = lib.flats_result_vector(res, r, c, ctypes.byref(n))
                row.append(p[: n.value])
            else:
                raise InternalError(f"unknown cell type {t} at row {r}, column {c}")
        rows.append(tuple(row))
    return columns, rows


class Cursor:
    """The rows of one statement, read lazily in chunks — see
    :meth:`Database.query`. Iterate it once; close it (or use ``with``) to
    release it early. ``columns`` is known before the first row.
    """

    def __init__(self, handle, chunk_rows):
        self._cur = handle
        self._chunk_rows = chunk_rows
        self._buffer = []
        self._at = 0
        self._done = False
        # The column names come with the first chunk.
        self._fill()
        self.columns = self._columns

    def _fill(self):
        res = ctypes.POINTER(_Result)()
        _check(_lib.flats_cursor_next(self._cur, self._chunk_rows, ctypes.byref(res)))
        try:
            columns, rows = _columns_and_rows(res)
        finally:
            _lib.flats_result_free(res)
        self._columns = tuple(columns)
        self._buffer, self._at = rows, 0
        if not rows:
            self._release()

    @property
    def closed(self):
        return self._cur is None

    def close(self):
        """Release the cursor. Idempotent; iteration then simply ends."""
        self._buffer, self._at = [], 0
        self._release()

    def _release(self):
        cur, self._cur = self._cur, None
        self._done = True
        if cur:
            _lib.flats_cursor_free(cur)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._at < len(self._buffer):
                row = self._buffer[self._at]
                self._at += 1
                return row
            if self._done:
                raise StopIteration
            self._fill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parameters and names
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _ident(name, what):
    """A collection or column name, checked: names are statement text, so
    they must be plain identifiers — values go in parameters."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"{what} must be a plain identifier, got {name!r}")
    return name


def _params(values):
    """Python values as a ctypes flats_param array, plus the buffers it points
    into (which must stay alive for the call)."""
    values = list(values)
    arr = (_Param * len(values))()
    keep = []
    for i, v in enumerate(values):
        p = arr[i]
        if isinstance(v, bool) or v is None:
            raise InvalidArgumentError(f"parameter {i}: {v!r} is not a V-SQL value")
        if isinstance(v, int):
            p.type, p.int_value = _INT, v
        elif isinstance(v, float):
            p.type, p.float_value = _FLOAT, v
        elif isinstance(v, str):
            data = v.encode("utf-8")
            keep.append(data)
            p.type, p.text, p.text_len = _TEXT, data, len(data)
        elif _is_vector(v):
            buf = _float32_buffer(v)
            keep.append(buf)
            p.type, p.vector, p.vector_len = _VECTOR, buf, len(buf)
        else:
            raise InvalidArgumentError(f"parameter {i}: {type(v).__name__} is not a V-SQL value")
    return arr, keep


def _float32_buffer(value):
    """``value`` as a ctypes float array. A contiguous float32 buffer
    (``array.array('f')``, a NumPy ``float32`` array) is copied in one move;
    anything else is converted element by element."""
    try:
        view = memoryview(value)
    except TypeError:
        view = None
    if view is not None and view.format == "f" and view.ndim == 1 and view.c_contiguous:
        return (ctypes.c_float * len(view)).from_buffer_copy(view)
    return (ctypes.c_float * len(value))(*value)


def _is_vector(value):
    return isinstance(value, (Sequence, memoryview)) and not isinstance(
        value, (str, bytes, bytearray)
    ) or hasattr(value, "typecode")


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


class Database:
    """An open Flats database. Safe to share between threads.

    Use as a context manager, or call ``close()``; closing checkpoints.
    """

    #: Rows sent per insert statement. Each batch is atomic; a large
    #: ``insert`` is atomic per batch, not as a whole.
    BATCH_ROWS = 10_000

    def __init__(self, path):
        handle = ctypes.POINTER(_Db)()
        _check(_lib.flats_open(_utf8(os.fspath(path), "path"), ctypes.byref(handle)))
        self._db = handle

    # -- lifecycle --------------------------------------------------------

    @property
    def closed(self):
        return self._db is None

    def close(self):
        """Checkpoint and close. Idempotent."""
        db, self._db = getattr(self, "_db", None), None
        if db:
            _check(_lib.flats_close(db))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _handle(self):
        if self._db is None:
            raise FlatsError("database is closed")
        return self._db

    # -- statements -------------------------------------------------------

    def execute(self, sql, params=()):
        """Run one V-SQL statement and return its :class:`Result`.

        Write ``?`` wherever a value goes and pass the values in ``params``, in
        order. **Always pass outside values this way**: a parameter is a value,
        never statement text, so nothing in it can change what the statement
        does. ``int``, ``float``, ``str`` and float sequences (vectors) are
        accepted.
        """
        arr, keep = _params(params)
        res = ctypes.POINTER(_Result)()
        _check(
            _lib.flats_execute_params(
                self._handle(), _utf8(sql, "sql"), arr, len(arr), ctypes.byref(res)
            )
        )
        del keep
        try:
            return _collect(res)
        finally:
            _lib.flats_result_free(res)

    def query(self, sql, params=(), *, chunk_rows=1000):
        """Run one V-SQL statement and return a :class:`Cursor` over its rows.

        Like :meth:`execute`, but rows are read from storage as you iterate,
        ``chunk_rows`` at a time, so memory stays flat however many there are.
        The cursor reads one snapshot: rows written after this call are not in
        it. Use it for large reads; :meth:`execute` collects everything.
        """
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive int")
        arr, keep = _params(params)
        cur = ctypes.POINTER(_Cursor)()
        _check(
            _lib.flats_query(self._handle(), _utf8(sql, "sql"), arr, len(arr), ctypes.byref(cur))
        )
        del keep
        return Cursor(cur, chunk_rows)

    def search(self, collection, vector, k=10, *, where=None, params=(), returning=None):
        """Top-``k`` rows of ``collection`` nearest ``vector`` (dot product).

        ``where`` is a V-SQL predicate whose values are ``?`` placeholders
        filled from ``params``: ``where="lang = ? AND year > ?",
        params=["en", 2020]``. ``returning`` is a list of column names,
        ``score`` and ``vector`` included. The query vector is passed as a
        binary parameter, exactly.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("vector components must be finite")
        sql = f"SEARCH TOP {k} NEAREST TO ? FROM {_ident(collection, 'collection')}"
        if where:
            sql += f" WHERE {where}"
        if returning:
            sql += " RETURNING " + ", ".join(_ident(c, "column") for c in returning)
        return self.execute(sql + ";", [vector, *params])

    def insert(self, collection, rows):
        """Insert ``rows`` — mappings of column name to value — and return the
        count. The embedding is the one sequence-valued entry (a list, tuple,
        or ``array.array``); every other column takes ``int``, ``float`` or
        ``str``. Rows go in batches of :attr:`BATCH_ROWS`, each all-or-nothing.
        """
        _ident(collection, "collection")
        batch = ctypes.POINTER(_Batch)()
        _check(_lib.flats_batch_new(self._handle(), _utf8(collection, "collection"), ctypes.byref(batch)))
        total = 0
        try:
            for row in rows:
                self._add_row(batch, row)
                if _lib.flats_batch_len(batch) >= self.BATCH_ROWS:
                    total += self._commit(batch)
            if _lib.flats_batch_len(batch):
                total += self._commit(batch)
        finally:
            _lib.flats_batch_free(batch)
        return total

    def backup(self, dest):
        """Write a consistent copy into ``dest`` (absent or an empty
        directory). Restore with ``Database(dest)``."""
        _check(_lib.flats_backup(self._handle(), _utf8(os.fspath(dest), "dest")))

    # -- batch plumbing ---------------------------------------------------

    @staticmethod
    def _add_row(batch, row):
        if not isinstance(row, Mapping):
            raise TypeError(f"each row must be a mapping, not {type(row).__name__}")
        lib = _lib
        for column, value in row.items():
            if _is_vector(value):
                buf = _float32_buffer(value)
                _check(lib.flats_batch_vector(batch, buf, len(buf)))
                continue
            name = _utf8(column, "column name")
            if isinstance(value, bool):
                raise InvalidArgumentError(f"column {column!r}: bool is not a V-SQL value")
            if isinstance(value, int):
                _check(lib.flats_batch_int(batch, name, value))
            elif isinstance(value, float):
                _check(lib.flats_batch_float(batch, name, value))
            elif isinstance(value, str):
                data = value.encode("utf-8")
                _check(lib.flats_batch_text(batch, name, data, len(data)))
            else:
                raise InvalidArgumentError(
                    f"column {column!r}: {type(value).__name__} is not a V-SQL value"
                )
        _check(lib.flats_batch_end_row(batch))

    @staticmethod
    def _commit(batch):
        n = ctypes.c_size_t()
        _check(_lib.flats_batch_commit(batch, ctypes.byref(n)))
        return n.value
