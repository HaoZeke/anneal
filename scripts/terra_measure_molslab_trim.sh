#!/usr/bin/env bash
# Trimmed Terra measure: (H2O)7 and (H2O)8, plain vs kappa 0.7, four seeds.
#
# GFN2 wall on the hexamer sweep: median 7848 us, max 20336 us per
# value_and_gradient. Budget 2000 at 25 us-ms worst case is 50 s per seed.
# Four sequential arms on 4 parallel seeds finish in a few minutes if the
# engine returns; SEED_TIMEOUT bounds a hung SCF.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_measure_molslab_trim.sh requires a Slurm allocation" >&2
  exit 1
fi

ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
RGPOT=${RGPOT_ROOT:-$HOME/Git/Github/TheochemUI/gpr_optim/subprojects/rgpot}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTB_CACHE_LIB=${XTB_CACHE_LIB:-$HOME/.cache/rattler/cache/pkgs/xtb-6.7.1-h8876d29_4/lib}
OUT=${MOLSLAB_OUT:-$ROOT/results/molslab-trim-78}
MOL_BIN=${MOL_BIN:-$ROOT/target/release/examples/molecular_cluster}
MOL_BUDGET=${MOL_BUDGET:-2000}
SEEDS=${SEEDS:-4}
SEED_TIMEOUT=${SEED_TIMEOUT:-8m}

if [[ ! -x $MOL_BIN ]]; then
  echo "missing $MOL_BIN" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/libxtb_engine.so ]]; then
  echo "missing $ROOT/engines/libxtb_engine.so" >&2
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEED_TIMEOUT

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
        timeout --signal=TERM --kill-after=30s "$SEED_TIMEOUT" \
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
  printf 'seeds=%s\n' "$SEEDS"
  printf 'seed_timeout=%s\n' "$SEED_TIMEOUT"
} >"$OUT/MEASURE_PROVENANCE"

for m in 7 8; do
  unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true
  run_seeds "h2o${m}_plain" "$MOL_BIN" "$m" "$MOL_BUDGET" 1
  export TWO_PHASE_KAPPA=0.7
  export TWO_PHASE_BETA=1
  unset TWO_PHASE_MU || true
  run_seeds "h2o${m}_k0.7" "$MOL_BIN" "$m" "$MOL_BUDGET" 1
  unset TWO_PHASE_KAPPA TWO_PHASE_MU TWO_PHASE_BETA || true
done

python3 "$ROOT/scripts/terra_summarize_molslab.py" "$OUT"

printf 'TERRA_MOLSLAB_TRIM_OK out=%s\n' "$OUT"
