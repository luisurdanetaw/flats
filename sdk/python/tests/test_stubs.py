"""The type stubs (flats/*.pyi) describe exactly the public API that exists:
every public class, method and function is in both, with the same
parameter names. A stub that drifts from the code is worse than none."""

import ast
import inspect
import os
import unittest

import flats
import flats.aio

HERE = os.path.dirname(os.path.abspath(flats.__file__))


def stub_api(path):
    """{name: params} for functions, {Class.method: params} for methods."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}

    def params(fn):
        a = fn.args
        return [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs if p.arg != "self"]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = params(node)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = None
            for item in node.body:
                is_property = any(
                    isinstance(d, ast.Name) and d.id == "property" for d in getattr(item, "decorator_list", [])
                )
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not is_property
                    and (not item.name.startswith("_") or item.name in ("__init__",))
                ):
                    out[f"{node.name}.{item.name}"] = params(item)
    return out


# Classes a user constructs; the rest come from calls (a Result from
# execute, an AsyncDatabase from flats.aio.open), so their __init__ is
# internal and stays out of the stubs.
CONSTRUCTED = {"Database"}


def runtime_api(module, names):
    out = {}
    for name in names:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            out[name] = None
            for attr, member in vars(obj).items():
                if attr.startswith("_") and not (attr == "__init__" and name in CONSTRUCTED):
                    continue
                if isinstance(member, property) or not callable(member):
                    continue
                if isinstance(member, staticmethod):
                    continue
                sig = inspect.signature(member)
                out[f"{name}.{attr}"] = [p for p in sig.parameters if p != "self"]
        elif callable(obj):
            out[name] = list(inspect.signature(obj).parameters)
    return out


class Stubs(unittest.TestCase):
    def check(self, module, stub, names):
        want = runtime_api(module, names)
        have = stub_api(os.path.join(HERE, stub))
        for key, params in want.items():
            self.assertIn(key, have, f"{stub} is missing {key}")
            if params is not None and have[key] is not None:
                self.assertEqual(have[key], params, f"{stub}: {key} parameters")
        for key in have:
            base = key.split(".")[0]
            if base in names:
                self.assertIn(key, want, f"{stub} declares {key}, which doesn't exist")

    def test_the_package_stub_matches(self):
        self.check(flats, "__init__.pyi", ["Database", "Result", "Cursor", "version"])

    def test_the_asyncio_stub_matches(self):
        self.check(flats.aio, "aio.pyi", ["AsyncDatabase", "AsyncCursor", "open"])

    def test_the_wheel_ships_the_stubs(self):
        setup = open(os.path.join(HERE, "..", "setup.py"), encoding="utf-8").read()
        self.assertIn("*.pyi", setup)
        self.assertTrue(os.path.exists(os.path.join(HERE, "py.typed")))


if __name__ == "__main__":
    unittest.main()
