#!/usr/bin/env bash
# One HyperQueue task: one seed of an ensemble campaign, or one chain when
# HISTORY_REPLICAS is unset. HQ sets HQ_TASK_ID; the arm is the mechanism
# list; every channel is read from the environment by the driver.
set -euo pipefail
N=${1:?n}
BUDGET=${2:?budget}
ARM=${3:?mechanisms}
export SEED_OFFSET=${SEED_OFFSET:-${HQ_TASK_ID:-0}}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
BIN=${LJ_BIN:-$HOME/anneal-psym/target/release/examples/lj_cluster_search}
sha256sum "$BIN"
exec "$BIN" "$N" "$BUDGET" 1 "$ARM"
