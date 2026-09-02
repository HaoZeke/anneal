#!/usr/bin/env bash
# CuH2 fcc slab baseline: four seeds, plain surface, paper budget 2500.
# The EAM evaluation is cheap against the hexamer GFN2 8 ms floor; 2500
# charged evaluations on four parallel seeds finish well inside 30 minutes.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_measure_cuh2.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTB_CACHE_LIB=${XTB_CACHE_LIB:-$HOME/.cache/rattler/cache/pkgs/xtb-6.7.1-h8876d29_4/lib}
OUT=${MOLSLAB_OUT:-$ROOT/results/cuh2-fcc-baseline}
SLAB_BIN=${SLAB_BIN:-$ROOT/target/release/examples/slab_adsorption}
CON=${SLAB_CON:-$ROOT/examples/fixtures/cuh2_fcc_slab.con}
SLAB_BUDGET=${SLAB_BUDGET:-2500}
SEEDS=${SEEDS:-4}

if [[ ! -x $SLAB_BIN ]]; then
  echo "missing $SLAB_BIN" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/librgpot_cuh2.so ]]; then
  echo "missing $ROOT/engines/librgpot_cuh2.so" >&2
  exit 1
fi
if [[ ! -f $CON ]]; then
  echo "missing slab geometry: $CON" >&2
  exit 1
fi

xtb_lib=
if [[ -d $RGPOT/.pixi/envs/xtbbld/lib ]]; then
  xtb_lib=$RGPOT/.pixi/envs/xtbbld/lib
elif [[ -d $XTB_CACHE_LIB ]]; then
  xtb_lib=$XTB_CACHE_LIB
fi
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
export IRA_LIB_DIR
export LD_LIBRARY_PATH="${xtb_lib:+$xtb_lib:}$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}"
export RGPOT_CUH2_LIBRARY=$ROOT/engines/librgpot_cuh2.so
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true

mkdir -p "$OUT"
{
  printf 'host=%s\n' "$(hostname)"
  printf 'job=%s\n' "$SLURM_JOB_ID"
  printf 'con=%s\n' "$CON"
  printf 'budget=%s\n' "$SLAB_BUDGET"
  printf 'seeds=%s\n' "$SEEDS"
  printf 'bin=%s\n' "$SLAB_BIN"
  printf 'engine=%s\n' "$RGPOT_CUH2_LIBRARY"
} >"$OUT/MEASURE_PROVENANCE"

dir=$OUT/cuh2_plain
mkdir -p "$dir"
echo "RUN cuh2_plain $SLAB_BIN $CON $SLAB_BUDGET 1" | tee "$dir/command.txt"
(
  cd "$dir"
  for seed in $(seq 0 $((SEEDS - 1))); do
    (
      export SEED_OFFSET=$seed
      "$SLAB_BIN" "$CON" "$SLAB_BUDGET" 1 \
        >"$dir/seed_${seed}.out" 2>"$dir/seed_${seed}.err"
    ) &
  done
  wait
)

python3 "$ROOT/scripts/terra_summarize_molslab.py" "$OUT" || true
printf 'TERRA_CUH2_BASELINE_OK out=%s\n' "$OUT"
