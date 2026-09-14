#!/usr/bin/env bash
# One water-hexamer seed at the frozen 4000-eval paper budget, comm kernel.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_hexamer_one.sh: run under sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
BIN=${MOL_BIN:-$CARGO_TARGET_DIR/release/examples/molecular_cluster}
export RGPOT_XTB_ENGINE=${RGPOT_XTB_ENGINE:-$ROOT/engines/libxtb_engine.so}
export LD_LIBRARY_PATH="${HOME}/rgpot/.pixi/envs/xtbbld/lib:${HOME}/ira/lib:${GCCLIB}:${LD_LIBRARY_PATH:-}"
export SEED_OFFSET=${SEED_OFFSET:-${SLURM_ARRAY_TASK_ID:-0}}
OUT=${COMM_OUT:-$ROOT/hexamer-comm}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID seed=$SEED_OFFSET"
"$BIN" 6 4000 1 comm | tee "$OUT/seed${SEED_OFFSET}.out"
if grep -q SOLVED "$OUT/seed${SEED_OFFSET}.out"; then
  echo HEX_OK
else
  echo HEX_MISS
fi
