#!/usr/bin/env bash
# Copyright (c) anneal developers
# Release build of examples/ase_nonlj. One core, one thread.
# Usage: build_arm.sh WORKTREE TARGET_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: build_arm.sh WORKTREE TARGET_DIR" >&2
  exit 2
fi

ROOT=$(cd "$1" && pwd)
mkdir -p "$2"
TARGET=$(cd "$2" && pwd)
CARGO_HOME_DIR=${CARGO_HOME:-$HOME/.cargo}

export LJOS_SEAT="${LJOS_SEAT:-benchbot}"
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export TBB_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export JULIA_NUM_THREADS=1
export CARGO_BUILD_JOBS=1
export CARGO_TARGET_DIR="$TARGET"
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${CARGO_HOME_DIR}=/cargo"

cd "$ROOT"
taskset -c 0 cargo +1.88.0 build --release --locked --example ase_nonlj -j 1
BIN="$TARGET/release/examples/ase_nonlj"
if [[ ! -x $BIN ]]; then
  echo "missing binary" >&2
  exit 1
fi
sha256sum "$BIN"
