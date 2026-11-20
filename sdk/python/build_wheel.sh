#!/usr/bin/env bash
# Build a wheel that bundles libflats_c.so: sdk/python/dist/flats-*.whl.
#
#   sdk/python/build_wheel.sh
#
# The wheel is tagged for this platform (linux_x86_64, say). For PyPI, run it
# through `auditwheel repair` on a manylinux image so it installs on other
# Linux systems; that tool is not needed to build or test locally.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cargo build --release -p flats-c --manifest-path "$root/Cargo.toml"
cp "$root/target/release/libflats_c.so" "$here/flats/libflats_c.so"
trap 'rm -f "$here/flats/libflats_c.so"' EXIT
rm -rf "$here/build" "$here/dist" "$here"/*.egg-info
(cd "$here" && python3 setup.py -q bdist_wheel)
rm -rf "$here/build" "$here"/*.egg-info
ls "$here"/dist/*.whl
