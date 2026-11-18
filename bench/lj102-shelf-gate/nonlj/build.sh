#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the gate drivers in one source tree.
#
#   build.sh <source-tree> <target-dir>
#
# The binary-LJ driver lives beside this script and is copied into the
# tree's examples directory for the build, then removed. Nothing else in the
# tree is touched. The build uses the repository's own release profile and
# lock file, with no extra compiler flags.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: build.sh <source-tree> <target-dir>" >&2
    exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="$(cd "$1" && pwd)"
target="$2"

cp "$here/h2h_blj.rs" "$tree/examples/h2h_blj.rs"
trap 'rm -f "$tree/examples/h2h_blj.rs"' EXIT

cd "$tree"
cargo build --release --locked --target-dir "$target" \
    --example h2h_morse --example h2h_metal --example h2h_seed --example h2h_blj
git -C "$tree" rev-parse HEAD > "$target/release/SOURCE_HEAD"
git -C "$tree" rev-parse 'HEAD^{tree}' > "$target/release/SOURCE_TREE"
rustc -V > "$target/release/RUSTC_VERSION"
