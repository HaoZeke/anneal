#!/bin/sh
# Copyright (c) anneal developers
# Release build of the gate driver from one clean source worktree.
#
#   build.sh <source-tree> <target-dir>
#
# The driver source beside this script is copied into the tree for the
# build and removed afterwards. The tree must be clean before and after.

set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: build.sh <source-tree> <target-dir>" >&2
    exit 2
fi

here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
tree=$(CDPATH= cd -- "$1" && pwd)
target=$2

porcelain=$(git -C "$tree" status --porcelain)
if [ -n "$porcelain" ]; then
    echo "source tree is not clean before build" >&2
    exit 1
fi

if [ -e "$tree/examples/lj_gate.rs" ]; then
    echo "temporary driver already exists" >&2
    exit 1
fi

cleanup() {
    rm -f "$tree/examples/lj_gate.rs"
}
trap cleanup EXIT

cp "$here/lj_gate.rs" "$tree/examples/lj_gate.rs"

(
    cd "$tree"
    cargo +1.88.0 build --release --locked --target-dir "$target" --example lj_gate
)

cleanup
trap - EXIT

porcelain=$(git -C "$tree" status --porcelain)
if [ -n "$porcelain" ]; then
    echo "source tree is not clean after build" >&2
    exit 1
fi

mkdir -p "$target/release"
git -C "$tree" rev-parse HEAD > "$target/release/SOURCE_HEAD"
git -C "$tree" rev-parse 'HEAD^{tree}' > "$target/release/SOURCE_TREE"
