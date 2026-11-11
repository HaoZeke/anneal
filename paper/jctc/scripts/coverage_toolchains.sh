#!/bin/sh
# Copyright (c) anneal developers
# Line and region coverage uses rustc 1.88.0.
# Branch coverage uses nightly-2026-11-01.
# The nightly rustc flag is -Zcoverage-options=branch.
# This script does not write a toolchain file.
set -eu

pinned=1.88.0
nightly=nightly-2026-11-01
branch_flag=-Zcoverage-options=branch

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
