"""Packaging for the Flats Python SDK.

The package is pure Python over a bundled shared library, so the wheel must
be tagged for the platform the library was built for (it is not a "pure"
wheel). `build_wheel.sh` builds the library and then this.

The license is not set yet: that is the code owner's decision, and an index
will refuse an upload without one.
"""

from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Carries a compiled library, so the wheel gets a platform tag."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """`py3-none-<platform>`: the library is loaded with ctypes, so the wheel
    works on any Python 3 — only the platform matters, not the CPython ABI."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(
    name="flats",
    version="0.1.0",
    description="Exact vector search with metadata filters, embedded in your process.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=["flats"],
    package_data={"flats": ["libflats_c.so", "py.typed", "*.pyi"]},
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
    zip_safe=False,
)
