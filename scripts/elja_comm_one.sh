#!/usr/bin/env bash
# One LJ seed at a frozen paper budget on Config::communicating.
# N and budget are arguments so 38/55/75/98 share one driver.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_comm_one.sh: run under sbatch, not on $(hostname)" >&2
  exit 1
fi
N=${1:?LJ site count}
BUDGET=${2:?charged evaluations}
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
export PATH="${HOME}/.cargo/bin:${PATH}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
BIN=${LJ_BIN:-$CARGO_TARGET_DIR/release/examples/lj_cluster_search}
if [[ ! -x $BIN ]]; then
  echo "missing $BIN; run scripts/elja_comm_build.sh first" >&2
  exit 2
fi
if ldd "$BIN" | grep -F "not found" >/dev/null; then
  echo "unresolved libraries in $BIN" >&2
  ldd "$BIN" >&2
  exit 2
fi
export SEED_OFFSET=${SEED_OFFSET:-${SLURM_ARRAY_TASK_ID:-0}}
OUT=${COMM_OUT:-$ROOT/comm$N}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID n=$N budget=$BUDGET seed=$SEED_OFFSET"
echo "RUN $BIN $N $BUDGET 1 comm"
"$BIN" "$N" "$BUDGET" 1 comm | tee "$OUT/seed${SEED_OFFSET}.out"
if grep -q SOLVED "$OUT/seed${SEED_OFFSET}.out"; then
  echo COMM_OK
else
  echo COMM_MISS
fi
