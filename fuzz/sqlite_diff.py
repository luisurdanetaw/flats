"""Differential testing of Flats against SQLite, through the Python SDK.

Random collections get random rows, random UPDATEs and DELETEs, and random
reads, mirrored into an SQLite table. V-SQL predicates are valid SQLite, so
the SAME `WHERE` text and parameters go to both engines, and their answers
must agree:

* SELECT ... WHERE p   — the same set of keys.
* SEARCH ... WHERE p   — checked against a brute-force top-k in Python
  floats over the rows SQLite says pass the filter.
* after every write    — every row, every value, identical.

SQLite is the independent oracle for predicate semantics (AND/OR
precedence, INT vs FLOAT comparison, string equality, 64-bit edges, -0.0),
and the Python side exercises the SDK's parameter and batch-insert paths,
which the Rust model test (tests/differential.rs) doesn't touch.

    cargo build --release -p flats-c
    python3 fuzz/sqlite_diff.py --seeds 50 --ops 300 [--seed-base N]

Exit status 1 on the first disagreement, with the seed, the statement and
both answers, also written to $FLATS_SIM_OUT (default target/sim-failures).
"""

import argparse
import math
import os
import random
import sqlite3
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sdk", "python"))
import flats  # noqa: E402

COLS = ["a", "b", "n", "m", "x", "y", "title", "year"]
TEXTS = ["", "a", "b", "A", "a ", "it's", "ünï", "日本語", "SELECT", "a;b", "\\", "x\ny", "%", "_"]
INT_EDGES = [0, 1, -1, 2**63 - 1, -(2**63), -(2**63) + 1, 2**53, 2**53 + 1]
FLOAT_EDGES = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-7, 5e-324, 2.2250738585072014e-308,
               1e300, -1e300, 1.7976931348623157e308, 9007199254740993.0]
OPS = ["=", "!=", "<", "<=", ">", ">="]


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def int_(self):
        p = self.r.random()
        if p < 0.1:
            return self.r.choice(INT_EDGES)
        if p < 0.2:
            return self.r.randint(-(2**63), 2**63 - 1)
        return self.r.randint(-12, 12)

    def float_(self):
        p = self.r.random()
        if p < 0.1:
            return self.r.choice(FLOAT_EDGES)
        if p < 0.2:
            f = struct.unpack("d", struct.pack("Q", self.r.getrandbits(64)))[0]
            return f if math.isfinite(f) else 0.75
        if p < 0.3:
            return self.r.randint(-1000, 1000) / 7.0
        return self.r.randint(-12, 12) / 2.0

    def text(self):
        if self.r.random() < 0.7:
            return self.r.choice(TEXTS)
        return "".join(self.r.choice("abz 'é") for _ in range(self.r.randint(0, 5)))

    def value(self, ty):
        if ty == "INT":
            return self.int_()
        if ty == "FLOAT":
            # An INT into a FLOAT column is legal; Flats coerces it.
            return self.int_() if self.r.random() < 0.15 else self.float_()
        return self.text()

    def vector(self, dim):
        style = self.r.randrange(3)
        if style == 0:
            return [float(self.r.randint(-2, 2)) for _ in range(dim)]
        return [f32(self.r.randint(-1000, 1000) / 100.0) for _ in range(dim)]


class Case:
    """One seeded run: a Flats database and an SQLite mirror."""

    def __init__(self, seed, root):
        self.seed = seed
        self.g = Gen(seed)
        self.db = flats.Database(os.path.join(root, "db"))
        self.sq = sqlite3.connect(":memory:")
        n = self.g.r.randint(1, 4)
        names = self.g.r.sample(COLS, n)
        self.cols = [(c, self.g.r.choice(["INT", "FLOAT", "TEXT"])) for c in names]
        self.pk_ty = self.g.r.choice(["INT", "TEXT"])
        self.dim = self.g.r.randint(1, 6)
        decl = ["id %s PRIMARY KEY" % self.pk_ty, "vector VECTOR(%d)" % self.dim]
        decl += ["%s %s" % c for c in self.cols]
        self.db.execute("CREATE COLLECTION t (%s) WITH (capacity = 1000000);" % ", ".join(decl))
        lite = {"INT": "INTEGER", "FLOAT": "REAL", "TEXT": "TEXT"}
        sdecl = ["id %s PRIMARY KEY" % lite[self.pk_ty], "vector TEXT"]
        sdecl += ["%s %s" % (c, lite[t]) for c, t in self.cols]
        self.sq.execute("CREATE TABLE t (%s) STRICT" % ", ".join(sdecl))
        self.keys = set()
        self.vectors = {}

    def close(self):
        self.db.close()
        self.sq.close()

    # -- predicates: one text, two engines --------------------------------

    def leaf(self):
        c, ty = self.g.r.choice(self.cols + [("id", self.pk_ty)])
        op = self.g.r.choice(["=", "!="]) if ty == "TEXT" else self.g.r.choice(OPS)
        v = self.existing(c) if self.g.r.random() < 0.5 else None
        if v is None:
            v = self.g.value(ty)
        if self.g.r.random() < 0.2:
            flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)
            return "? %s %s" % (flipped, c), [(v, ty)]
        return "%s %s ?" % (c, op), [(v, ty)]

    def pred(self, depth=3):
        if self.g.r.random() < 0.04:
            parts = [self.leaf() for _ in range(self.g.r.randint(2, 45))]
            return " OR ".join(p for p, _ in parts), [v for _, vs in parts for v in vs]
        if depth == 0 or self.g.r.random() < 0.45:
            return self.leaf()
        a, av = self.pred(depth - 1)
        b, bv = self.pred(depth - 1)
        if self.g.r.random() < 0.5:
            # AND binds tighter, so an OR under an AND needs parentheses.
            return "(%s) AND (%s)" % (a, b), av + bv
        return "%s OR %s" % (a, b), av + bv

    def existing(self, col):
        row = self.sq.execute("SELECT %s FROM t ORDER BY random() LIMIT 1" % col).fetchone()
        return row[0] if row else None

    @staticmethod
    def flats_params(vals):
        return [v for v, _ in vals]

    @staticmethod
    def sqlite_params(vals):
        # Flats coerces an INT compared with a FLOAT column to a double;
        # SQLite would compare the integer exactly. Mirror the coercion.
        return [float(v) if ty == "FLOAT" else v for v, ty in vals]

    # -- ops ---------------------------------------------------------------

    def fresh_key(self):
        while True:
            k = self.g.int_() if self.pk_ty == "INT" else self.g.text() + str(self.g.r.randint(0, 10**6))
            if k not in self.keys:
                return k

    def insert(self):
        rows = []
        for _ in range(self.g.r.choice([1, 1, 5, 40])):
            k = self.fresh_key()
            self.keys.add(k)
            row = {"id": k, "vector": self.g.vector(self.dim)}
            for c, ty in self.cols:
                row[c] = self.g.value(ty)
            rows.append(row)
        self.db.insert("t", rows)  # the SDK's batch path
        for row in rows:
            vals = [row["id"], repr(row["vector"])]
            vals += [float(row[c]) if ty == "FLOAT" else row[c] for c, ty in self.cols]
            self.sq.execute("INSERT INTO t VALUES (%s)" % ", ".join("?" * len(vals)), vals)
            self.vectors[row["id"]] = row["vector"]
        return "insert %d rows" % len(rows)

    def update(self):
        if not self.cols:
            return None
        c, ty = self.g.r.choice(self.cols)
        v = self.g.value(ty)
        where, wv = self.pred(2)
        sql = "UPDATE t SET %s = ? WHERE %s;" % (c, where)
        self.db.execute(sql, [v] + self.flats_params(wv))
        self.sq.execute(sql, [float(v) if ty == "FLOAT" else v] + self.sqlite_params(wv))
        return sql

    def delete(self):
        where, wv = self.pred(2)
        sql = "DELETE FROM t WHERE %s;" % where
        self.db.execute(sql, self.flats_params(wv))
        gone = [r[0] for r in self.sq.execute("SELECT id FROM t WHERE %s" % where, self.sqlite_params(wv))]
        self.sq.execute(sql, self.sqlite_params(wv))
        for k in gone:
            self.keys.discard(k)
            self.vectors.pop(k, None)
        return sql

    def select(self):
        where, wv = self.pred()
        sql = "SELECT id FROM t WHERE %s;" % where
        got = sorted(map(repr, (r[0] for r in self.db.execute(sql, self.flats_params(wv)))))
        want = sorted(map(repr, (r[0] for r in self.sq.execute(sql, self.sqlite_params(wv)))))
        if got != want:
            self.fail(sql, wv, "flats %d rows, sqlite %d\n  only flats: %s\n  only sqlite: %s" % (
                len(got), len(want), sorted(set(got) - set(want))[:5], sorted(set(want) - set(got))[:5]))
        return sql

    def search(self):
        q = self.g.vector(self.dim)
        n = len(self.keys)
        k = self.g.r.randint(1, max(1, n // 3)) if self.g.r.random() < 0.8 else self.g.r.randint(1, 12)
        use_where = self.g.r.random() < 0.6
        where, wv = self.pred() if use_where else ("", [])
        clause = " WHERE %s" % where if use_where else ""
        sql = "SEARCH TOP ? NEAREST TO ? FROM t%s RETURNING id, score;" % clause
        got = list(self.db.execute(sql, [k, q] + self.flats_params(wv)))
        passing = [r[0] for r in self.sq.execute("SELECT id FROM t%s" % clause, self.sqlite_params(wv))]
        scored = {}
        for key in passing:
            v = self.vectors[key]
            s = sum(a * b for a, b in zip(q, v))
            tol = 1e-4 * sum(abs(a * b) for a, b in zip(q, v)) + 1e-30
            scored[key] = (s, tol)
        if len(got) != min(k, len(passing)):
            self.fail(sql, wv, "flats returned %d rows, want %d" % (len(got), min(k, len(passing))))
        chosen = set()
        for key, score in got:
            if key not in scored or key in chosen:
                self.fail(sql, wv, "flats returned %r, which does not pass the filter (or twice)" % (key,))
            s, tol = scored[key]
            if abs(score - s) > tol:
                self.fail(sql, wv, "%r scored %r, want %r" % (key, score, s))
            chosen.add(key)
        if got:
            lo_key = min(chosen, key=lambda c: scored[c][0])
            lo, lo_tol = scored[lo_key]
            for key, (s, tol) in scored.items():
                if key not in chosen and s > lo + max(tol, lo_tol):
                    self.fail(sql, wv, "%r (score %r) left out of the top %d; %r made it" % (key, s, k, lo))
        return sql

    def verify(self):
        names = ["id"] + [c for c, _ in self.cols]
        sql = "SELECT %s FROM t;" % ", ".join(names)
        canon = lambda r: tuple(repr(x) for x in r)  # noqa: E731
        got = sorted(canon(r) for r in self.db.execute(sql))
        # SQLite folds -0.0 into 0.0 on the way in; compare it as 0.0 there.
        want = sorted(canon(r) for r in self.sq.execute(sql[:-1]))
        fix = lambda rows: [tuple("0.0" if x == "-0.0" else x for x in r) for r in rows]  # noqa: E731
        if fix(got) != fix(want):
            diff = set(fix(got)) ^ set(fix(want))
            self.fail(sql, [], "tables differ: %d vs %d rows; e.g. %s" % (len(got), len(want), sorted(diff)[:4]))

    def fail(self, sql, vals, why):
        msg = "seed %d: %s\n  statement: %s\n  params: %r\n" % (self.seed, why, sql, vals)
        out = os.environ.get("FLATS_SIM_OUT", os.path.join(HERE, "..", "target", "sim-failures"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "sqlite-diff-seed-%d.txt" % self.seed), "w") as f:
            f.write(msg)
        raise AssertionError(msg)

    def run(self, n_ops):
        for i in range(n_ops):
            p = self.g.r.random()
            if not self.keys or p < 0.25:
                self.insert()
            elif p < 0.35:
                self.update()
            elif p < 0.42:
                self.delete()
            elif p < 0.70:
                self.select()
            else:
                self.search()
            if i % 25 == 24:
                self.verify()
        self.verify()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=int(os.environ.get("FLATS_SIM_SEEDS", 20)))
    ap.add_argument("--seed-base", type=int, default=int(os.environ.get("FLATS_SIM_SEED_BASE", 0)))
    ap.add_argument("--ops", type=int, default=int(os.environ.get("FLATS_SIM_OPS", 250)))
    args = ap.parse_args()
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        with tempfile.TemporaryDirectory() as root:
            case = Case(seed, root)
            try:
                case.run(args.ops)
            finally:
                case.close()
    print("sqlite_diff: %d seeds x %d ops agree" % (args.seeds, args.ops))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
