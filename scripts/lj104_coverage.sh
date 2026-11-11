#!/bin/sh
# Copyright (c) anneal developers
# Line cover and timings use the 1.88 compiler.
# Branch cover uses the date-pinned nightly, with no toolchain file.
# The scope is the example and the library code that example calls.

set -eu

STABLE=1.88.0
NIGHTLY=nightly-2025-06-26

echo "stable_rustc=$(rustup run "$STABLE" rustc -V)"
echo "stable_cargo=$(rustup run "$STABLE" cargo -V)"
echo "nightly_rustc=$(rustup run "$NIGHTLY" rustc -V)"
echo "nightly_cargo=$(rustup run "$NIGHTLY" cargo -V)"

echo "The stable run uses instrument-coverage."
echo "The nightly run adds -Zcoverage-options=branch."
echo "No toolchain file is read or written."

stable_flags="-C instrument-coverage"
nightly_flags="-C instrument-coverage -Zcoverage-options=branch"

run_one() {
  label=$1
  tc=$2
  flags=$3
  out=$4
  mkdir -p "$out"
  rm -f "$out"/*.profraw
  LLVM_PROFILE_FILE="$out/p-%p-%m.profraw" \
    RUSTFLAGS="$flags" \
    rustup run "$tc" cargo test --offline --lib --example h2h_gmin_proof -- \
      --test-threads=1
}

if [ "${1:-}" = "--stable" ]; then
  run_one stable "$STABLE" "$stable_flags" logs/coverage/stable-raw
fi
if [ "${1:-}" = "--nightly" ]; then
  run_one nightly "$NIGHTLY" "$nightly_flags" logs/coverage/nightly-raw
fi
