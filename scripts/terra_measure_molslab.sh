#!/usr/bin/env bash
# Baseline and two-phase water / CuH2 measurements on Terra.
#
# 8 seeds in parallel. Water hexamer through decamer at the paper hexamer
# charged budget (4000). CuH2 fcc slab at the paper slab budget (2500).
# Two-phase arms: relative kappa in {0.7, 0.8} and mu in {2.5, 5}, scaled
# by the example through energy_scale / length_scale^2.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_measure_molslab.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTB_CACHE_LIB=${XTB_CACHE_LIB:-$HOME/.cache/rattler/cache/pkgs/xtb-6.7.1-h8876d29_4/lib}
OUT=${MOLSLAB_OUT:-$ROOT/results/molslab-two-phase}
MOL_BIN=${MOL_BIN:-$ROOT/target/release/examples/molecular_cluster}
SLAB_BIN=${SLAB_BIN:-$ROOT/target/release/examples/slab_adsorption}
CON=${SLAB_CON:-$ROOT/examples/fixtures/cuh2_fcc_slab.con}
MOL_BUDGET=${MOL_BUDGET:-4000}
SLAB_BUDGET=${SLAB_BUDGET:-2500}
SEEDS=${SEEDS:-8}

if [[ ! -x $MOL_BIN || ! -x $SLAB_BIN ]]; then
  echo "missing example binaries; run scripts/terra_build_molslab_dev.sh" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/libxtb_engine.so || ! -e $ROOT/engines/librgpot_cuh2.so ]]; then
  echo "missing engines under $ROOT/engines" >&2
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
export RGPOT_XTB_ENGINE=$ROOT/engines/libxtb_engine.so
export RGPOT_CUH2_LIBRARY=$ROOT/engines/librgpot_cuh2.so
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_seeds() {
  local label=$1
  shift
  local dir=$OUT/$label
  mkdir -p "$dir"
  echo "RUN $label $*" | tee "$dir/command.txt"
  (
    cd "$dir"
    for seed in $(seq 0 $((SEEDS - 1))); do
      (
        export SEED_OFFSET=$seed
        timeout --signal=TERM --kill-after=30s "${SEED_TIMEOUT:-25m}" \
          "$@" >"$dir/seed_${seed}.out" 2>"$dir/seed_${seed}.err"
      ) &
    done
    wait
  )
}

mkdir -p "$OUT"
{
  source_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
  if [[ -z $source_commit && -s $ROOT/SOURCE_COMMIT ]]; then
    source_commit=$(cat "$ROOT/SOURCE_COMMIT")
  fi
  printf 'source=%s\n' "${source_commit:-unknown}"
  printf 'host=%s\n' "$(hostname)"
  printf 'job=%s\n' "$SLURM_JOB_ID"
  printf 'mol_budget=%s\n' "$MOL_BUDGET"
  printf 'slab_budget=%s\n' "$SLAB_BUDGET"
  printf 'seeds=%s\n' "$SEEDS"
} >"$OUT/MEASURE_PROVENANCE"

for m in 6 7 8 9 10; do
  unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true
  run_seeds "h2o${m}_plain" "$MOL_BIN" "$m" "$MOL_BUDGET" 1
  for kappa in 0.7 0.8; do
    for mu in 2.5 5; do
      export TWO_PHASE_KAPPA=$kappa
      export TWO_PHASE_MU=$mu
      export TWO_PHASE_BETA=1
      run_seeds "h2o${m}_k${kappa}_m${mu}" "$MOL_BIN" "$m" "$MOL_BUDGET" 1
    done
  done
  unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true
done

unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true
run_seeds "cuh2_plain" "$SLAB_BIN" "$CON" "$SLAB_BUDGET" 1

python3 "$ROOT/scripts/terra_summarize_molslab.py" "$OUT"

printf 'TERRA_MOLSLAB_MEASURE_OK out=%s\n' "$OUT"
