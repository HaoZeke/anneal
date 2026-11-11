#!/bin/sh
# Copyright (c) anneal developers
# Run cargo-mutants on one Rust file.
# The mutants.out log keeps the caught, missed, and timeout counts.
# Usage: mutants_one_file.sh FILE OUTDIR
set -eu

file=${1:?rust file}
out=${2:?output directory}
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$out"
RUSTUP_TOOLCHAIN=1.88.0 cargo mutants \
  --file "$file" \
  --timeout 180 \
  --output "$out" \
  -- \
  --lib
