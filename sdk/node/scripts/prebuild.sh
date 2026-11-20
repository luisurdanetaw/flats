#!/usr/bin/env bash
# Build the addon and place it where a published package looks for it:
# prebuilds/<platform>-<arch>/flats.node. Run once per platform (CI does it
# on each runner); `npm pack` then ships every prebuild that is present.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
cargo build --release -p flats-node --manifest-path "$root/Cargo.toml"
target=$(node -p 'process.platform + "-" + process.arch')
mkdir -p "$here/prebuilds/$target"
cp "$root/target/release/libflats_node.so" "$here/prebuilds/$target/flats.node"
echo "prebuilds/$target/flats.node"
