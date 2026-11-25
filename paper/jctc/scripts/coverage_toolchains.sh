#!/bin/sh
# Copyright (c) anneal developers
# Line and region coverage uses PINNED_TOOLCHAIN from coverage_receipts.py.
# Branch coverage uses NIGHTLY_TOOLCHAIN with BRANCH_FLAG from coverage_receipts.py.
# This script does not write a toolchain file.
set -eu

here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
toolchains=$(
  python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
import coverage_receipts as cr
print(cr.PINNED_TOOLCHAIN, cr.NIGHTLY_TOOLCHAIN, cr.BRANCH_FLAG)
' "$here"
)
IFS=' ' read -r pinned nightly branch_flag <<EOF
$toolchains
EOF
: "${pinned:?missing PINNED_TOOLCHAIN}"
: "${nightly:?missing NIGHTLY_TOOLCHAIN}"
: "${branch_flag:?missing BRANCH_FLAG}"

cd "$(git rev-parse --show-toplevel)"

echo "rustc_188 $(rustup run "$pinned" rustc -V)"
echo "cargo_188 $(rustup run "$pinned" cargo -V)"
echo "rustc_nightly $(rustup run "$nightly" rustc -V)"
echo "cargo_nightly $(rustup run "$nightly" cargo -V)"
rustup run "$nightly" rustc "$branch_flag" -V

mode=${1:?mode is lines or branches}
out=${2:?output json}
shift 2

case "$mode" in
  lines) toolchain=$pinned ;;
  branches) toolchain=$nightly ;;
  *)
    echo "mode is lines or branches" >&2
    exit 2
    ;;
esac

if [ "$mode" = branches ]; then
  RUSTUP_TOOLCHAIN=$toolchain cargo llvm-cov --locked --json \
    --output-path "$out" --branch "$@"
else
  RUSTUP_TOOLCHAIN=$toolchain cargo llvm-cov --locked --json \
    --output-path "$out" "$@"
fi
