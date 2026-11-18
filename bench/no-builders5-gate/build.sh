#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the gate drivers from one clean source worktree.
#
#   build.sh <source-tree> <target-dir> [toolchain]
#
# The temporary gate examples live beside this script and are copied into
# <source-tree>/examples for the build, then removed. <source-tree> must
# be clean before and after the build.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: build.sh <source-tree> <target-dir> [toolchain]" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="$(cd "$1" && pwd)"
target="$2"
toolchain="${3:-1.88.0}"

porcelain_before="$(git -C "$tree" status --porcelain)"
if [ -n "$porcelain_before" ]; then
    echo "source tree $tree is not clean before build" >&2
    exit 1
fi

if [ -e "$tree/examples/lj_shelf_gate.rs" ] || [ -e "$tree/examples/h2h_blj.rs" ]; then
    echo "temporary runner already exists in $tree/examples" >&2
    exit 1
fi

cleanup() {
    rm -f "$tree/examples/lj_shelf_gate.rs" "$tree/examples/h2h_blj.rs"
}
trap cleanup EXIT

cp "$here/lj_runner.rs" "$tree/examples/lj_shelf_gate.rs"
cp "$here/h2h_blj.rs" "$tree/examples/h2h_blj.rs"

cargo_args=()
if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q "^${toolchain}"; then
    cargo_args+=("+${toolchain}")
    rustc_cmd=(rustc "+${toolchain}")
else
    rustc_cmd=(rustc)
fi

(
    cd "$tree"
    cargo "${cargo_args[@]}" build --release --locked --target-dir "$target" \
        --example lj_shelf_gate \
        --example h2h_morse \
        --example h2h_metal \
        --example h2h_seed \
        --example h2h_blj
)

cleanup
trap - EXIT

porcelain_after="$(git -C "$tree" status --porcelain)"
if [ -n "$porcelain_after" ]; then
    echo "source tree $tree is not clean after build" >&2
    exit 1
fi

mkdir -p "$target/release"
git -C "$tree" rev-parse HEAD > "$target/release/SOURCE_HEAD"
git -C "$tree" rev-parse 'HEAD^{tree}' > "$target/release/SOURCE_TREE"
printf '%s\n' "$porcelain_after" > "$target/release/PORCELAIN_STATUS"
"${rustc_cmd[@]}" -V > "$target/release/RUSTC_VERSION"
