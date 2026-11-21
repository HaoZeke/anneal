#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of the panel drivers from one clean arm worktree.
#
#   build.sh <arm-tree> <expected-tree> <target-dir> [toolchain]
#
# The drivers from gate_examples.py are written into <arm-tree>/examples for
# the build and moved into the target directory after it. The worktree is
# clean, at
# <expected-tree>, before and after.
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: build.sh <arm-tree> <expected-tree> <target-dir> [toolchain]" >&2
  exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tree=$(cd "$1" && pwd)
expected=$2
target=$3
toolchain=${4:-1.88.0}
names=(gate_h2h_gmin_proof gate_h2h_morse gate_h2h_sc gate_h2h_blj)

check_tree() {
  local actual
  actual=$(git -C "$tree" rev-parse 'HEAD^{tree}')
  if [ "$actual" != "$expected" ]; then
    echo "tree $actual is not $expected" >&2
    exit 1
  fi
  if [ -n "$(git -C "$tree" status --porcelain --untracked-files=all)" ]; then
    echo "$tree is not clean" >&2
    exit 1
  fi
}

check_tree
cleanup() {
  local trash="$target/.trash/build-$$"
  mkdir -p "$trash"
  for name in "${names[@]}"; do
    if [ -e "$tree/examples/$name.rs" ]; then
      mv "$tree/examples/$name.rs" "$trash/"
    fi
  done
}
trap cleanup EXIT

python3 "$here/gate_examples.py" "$tree" "$tree/examples"
args=()
for name in "${names[@]}"; do
  args+=(--example "$name")
done

(cd "$tree" && cargo "+$toolchain" build --release --locked --features rgpot-ex \
  --target-dir "$target" "${args[@]}")

mkdir -p "$target/drivers"
for name in "${names[@]}"; do
  mv "$tree/examples/$name.rs" "$target/drivers/$name.rs"
done
trap - EXIT
check_tree

git -C "$tree" rev-parse 'HEAD^{tree}' >"$target/SOURCE_TREE"
rustc "+$toolchain" -V >"$target/RUSTC_VERSION"
