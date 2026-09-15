#!/usr/bin/env bash
#SBATCH --job-name=jcc-freeze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%A_%a.out
#
# One array task = one seed of one paper system, run against the frozen
# `recommended` arm. Budgets and seed counts are the published ones:
#
#   LJ38  4e5 x 72
#   LJ55  1e6 x 48
#   LJ75  4e6 x 48
#   LJ98  4e6 x 48
#
# Submit one system at a time so each harvest lands in its own directory:
#   sbatch --array=0-71 --export=ALL,SYS=38,BUDGET=400000 scripts/terra_jcc_freeze.sbatch
set -euo pipefail
SYS=${SYS:?SYS=38|55|75|98}
BUDGET=${BUDGET:?charged evaluation budget}
ARM=${ARM:-rec}
ROOT=${JCC_ROOT:-$HOME/Git/Github/Rust/anneal}
OUT=${JCC_OUT:-$HOME/ljwork/jcc-freeze}/lj${SYS}
BIN=${LJ_BIN:-${JCC_TARGET:-$ROOT/target}/release/examples/lj_cluster_search}

export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export SEED_OFFSET=${SLURM_ARRAY_TASK_ID:-0}
# One replica, through the ensemble driver. The deposited orbit dumps that
# the freeze reproduces were run this way ("history ensembles: 1 replicas"),
# and the plain single-seed path is a different trajectory at the same flags,
# so both arms take this one or neither does.
export HISTORY_REPLICAS=${HISTORY_REPLICAS:-1}
# No minimum history. The ensemble driver defaults to a private history, which
# on LJ75 spends about two minutes of a four-minute run observing minima the
# single-chain arm never consults; the deposited orbit dumps ran `no history`.
export HISTORY=${HISTORY:-none}

mkdir -p "$OUT"
exec "$BIN" "$SYS" "$BUDGET" 1 "$ARM" >"$OUT/lj${SYS}_${ARM}_${SEED_OFFSET}.out" 2>&1
