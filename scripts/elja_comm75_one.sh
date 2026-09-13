#!/usr/bin/env bash
# One LJ75 seed at the frozen paper budget on Config::communicating.
# Same 4e6 one-chain budget as tbl:comb recommended 4/48 and o75-orbit 31/48.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_comm75_one.sh: run under sbatch, not on $(hostname)" >&2
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
  echo "missing $BIN; run scripts/elja_comm_build.sh first" >&2
  exit 2
fi
if ldd "$BIN" | grep -F "not found" >/dev/null; then
  echo "unresolved libraries in $BIN" >&2
  ldd "$BIN" >&2
  exit 2
fi
export SEED_OFFSET=${SEED_OFFSET:-${SLURM_ARRAY_TASK_ID:-0}}
OUT=${COMM_OUT:-$ROOT/comm75}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID seed=$SEED_OFFSET"
echo "RUN $BIN 75 4000000 1 comm"
"$BIN" 75 4000000 1 comm | tee "$OUT/seed${SEED_OFFSET}.out"
if grep -q 'verified -397.492331' "$OUT/seed${SEED_OFFSET}.out" \
  && grep -q SOLVED "$OUT/seed${SEED_OFFSET}.out"; then
  echo COMM75_OK
else
  echo COMM75_MISS
fi
