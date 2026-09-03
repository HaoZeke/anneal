#!/usr/bin/env bash
# Multi-site Cu(111)+H6 comparison: plain hop, recommended hop, and the
# random-relaxation baseline at the same total charged evaluations.
#
# 8 seeds x 5000 charged evaluations for each hop arm. The baseline
# spends 8*5000 on independent random placements and quenches.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_measure_slab_h6.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-slab}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTB_CACHE_LIB=${XTB_CACHE_LIB:-$HOME/.cache/rattler/cache/pkgs/xtb-6.7.1-h8876d29_4/lib}
OUT=${MOLSLAB_OUT:-$ROOT/results/slab-h6}
TARGET=${CARGO_TARGET_DIR:-$ROOT/target}
SLAB_BIN=${SLAB_BIN:-$TARGET/release/examples/slab_adsorption}
RELAX_BIN=${RELAX_BIN:-$TARGET/release/examples/slab_random_relax}
CON=${SLAB_CON:-$ROOT/examples/fixtures/cuh2_fcc_slab_h6.con}
SLAB_BUDGET=${SLAB_BUDGET:-5000}
SEEDS=${SEEDS:-8}
RELAX_BUDGET=${RELAX_BUDGET:-$((SLAB_BUDGET * SEEDS))}
ENGINE=${RGPOT_CUH2_LIBRARY:-$ROOT/engines/librgpot_cuh2.so}

if [[ ! -x $SLAB_BIN ]]; then
  echo "missing $SLAB_BIN" >&2
  exit 1
fi
if [[ ! -x $RELAX_BIN ]]; then
  echo "missing $RELAX_BIN" >&2
  exit 1
fi
if [[ ! -e $ENGINE ]]; then
  echo "missing CuH2 engine: $ENGINE" >&2
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
export RGPOT_CUH2_LIBRARY=$ENGINE
export POTENTIAL_LIBRARY=${POTENTIAL_LIBRARY:-$ENGINE}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true

run_seeds() {
  local label=$1
  local arm=$2
  local dir=$OUT/$label
  mkdir -p "$dir"
  echo "RUN $label $SLAB_BIN $CON $SLAB_BUDGET 1 $arm" | tee "$dir/command.txt"
  (
    cd "$dir"
    for seed in $(seq 0 $((SEEDS - 1))); do
      (
        export SEED_OFFSET=$seed
        timeout --signal=TERM --kill-after=30s "${SEED_TIMEOUT:-45m}" \
          "$SLAB_BIN" "$CON" "$SLAB_BUDGET" 1 "$arm" \
          >"$dir/seed_${seed}.out" 2>"$dir/seed_${seed}.err"
      ) &
    done
    wait
  )
}

mkdir -p "$OUT"
{
  source_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
  printf 'source=%s\n' "${source_commit:-unknown}"
  printf 'host=%s\n' "$(hostname)"
  printf 'job=%s\n' "$SLURM_JOB_ID"
  printf 'con=%s\n' "$CON"
  printf 'budget=%s\n' "$SLAB_BUDGET"
  printf 'seeds=%s\n' "$SEEDS"
  printf 'relax_budget=%s\n' "$RELAX_BUDGET"
  printf 'slab_bin=%s\n' "$SLAB_BIN"
  printf 'relax_bin=%s\n' "$RELAX_BIN"
  printf 'engine=%s\n' "$ENGINE"
} >"$OUT/MEASURE_PROVENANCE"

run_seeds "cuh2_h6_plain" plain
run_seeds "cuh2_h6_recommended" recommended

dir=$OUT/cuh2_h6_random_relax
mkdir -p "$dir"
echo "RUN cuh2_h6_random_relax $RELAX_BIN $CON $RELAX_BUDGET 0" | tee "$dir/command.txt"
(
  cd "$dir"
  timeout --signal=TERM --kill-after=30s "${RELAX_TIMEOUT:-90m}" \
    "$RELAX_BIN" "$CON" "$RELAX_BUDGET" 0 \
    >"$dir/seed_0.out" 2>"$dir/seed_0.err"
)

python3 "$ROOT/scripts/terra_summarize_slab_h6.py" "$OUT"
printf 'TERRA_SLAB_H6_MEASURE_OK out=%s\n' "$OUT"
