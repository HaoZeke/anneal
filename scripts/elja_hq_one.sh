#!/usr/bin/env bash
# One HyperQueue task: one seed, one arm, paper budget.
# HQ sets HQ_TASK_ID. Arm is `rec` (scientific recommended) or `base` (for_cluster).
set -euo pipefail
N=${1:?n}
BUDGET=${2:?budget}
ARM=${3:?rec|base}
export SEED_OFFSET=${SEED_OFFSET:-${HQ_TASK_ID:-0}}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
BIN=${LJ_BIN:-$HOME/anneal-build/target/release/examples/lj_cluster_search}
exec "$BIN" "$N" "$BUDGET" 1 "$ARM"
