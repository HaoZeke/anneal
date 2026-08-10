#!/usr/bin/env bash
# One HyperQueue task: one seed, one arm, paper budget.
# HQ sets HQ_TASK_ID. Arm is `rec` (scientific recommended) or `base` (for_cluster).
set -euo pipefail
N=${1:?n}
BUDGET=${2:?budget}
ARM=${3:?rec|base}
export SEED_OFFSET=${HQ_TASK_ID:-0}
BIN=${LJ_BIN:-$HOME/anneal-build/target/release/examples/lj_cluster_search}
exec "$BIN" "$N" "$BUDGET" 1 "$ARM"
