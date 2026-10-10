#!/usr/bin/env bash
# Frozen hist75 communicating dump: 4 replicas, 4e6 aggregate, comm preset.
# Same split as Elja HQ 5617-5621 (private 10/48, shared 12-14/48, single 16/48).
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_hist75_comm.sh: run under sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
export PATH="${HOME}/.cargo/bin:${PATH}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
BIN=${LJ_BIN:-$CARGO_TARGET_DIR/release/examples/lj_cluster_search}
if [[ ! -x $BIN ]]; then
  echo "missing $BIN" >&2
  exit 2
fi
export SEED_OFFSET=${SEED_OFFSET:-${SLURM_ARRAY_TASK_ID:-0}}
export HISTORY_REPLICAS=${HISTORY_REPLICAS:-4}
export HISTORY=${HISTORY:-shared}
export HISTORY_POLICY=${HISTORY_POLICY:-accepted}
OUT=${COMM_OUT:-$ROOT/hist75-comm}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID seed=$SEED_OFFSET replicas=$HISTORY_REPLICAS history=$HISTORY policy=$HISTORY_POLICY"
echo "RUN $BIN 75 4000000 1 comm"
"$BIN" 75 4000000 1 comm | tee "$OUT/seed${SEED_OFFSET}.out"
if grep -q SOLVED "$OUT/seed${SEED_OFFSET}.out"; then
  echo HIST75_COMM_OK
else
  echo HIST75_COMM_MISS
fi
