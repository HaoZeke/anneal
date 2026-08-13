#!/usr/bin/env bash
# Submit matched shared/control LJ ensemble arrays on Elja.
set -euo pipefail

STAGE=${1:?development, qualification, or production}
ROOT=${LJ_ROOT:-$HOME/anneal-build}
REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
RUNNER=$ROOT/scripts/elja_jcc_lj_ensemble.sh
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/jcc}
PYTHON=${JCC_PYTHON:-$HOME/rgpot/.pixi/envs/xtbbld/bin/python}
RADIUS_READER=$REPRO_ROOT/workflow/jcc/read_census_radius.py
CAMPAIGN=${JCC_CAMPAIGN:-jcc-2026-${STAGE}}

case "$STAGE" in
  development)
    ENSEMBLES=${JCC_ENSEMBLES:-4}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-0}
    ;;
  qualification)
    ENSEMBLES=${JCC_ENSEMBLES:-12}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-100000}
    ;;
  production)
    ENSEMBLES=${JCC_ENSEMBLES:-24}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-200000}
    ;;
  *)
    echo "stage must be development, qualification, or production" >&2
    exit 2
    ;;
esac

if [[ ! -x $RUNNER ]]; then
  echo "missing ensemble runner: $RUNNER" >&2
  exit 1
fi
if [[ ! -x $PYTHON || ! -f $RADIUS_READER ]]; then
  echo "missing census-radius validator or its Python interpreter" >&2
  exit 1
fi
(cd "$REPRO_ROOT" && sha256sum --check --strict results_jcc/calibration/SHA256SUMS)

mkdir -p "$OUT_ROOT/submissions"
last=$((ENSEMBLES - 1))
for n in 75 98 102 104; do
  radius_variable="LJ${n}_CENSUS_RADIUS"
  radius=${!radius_variable:-}
  if [[ -z $radius ]]; then
    radius=$("$PYTHON" "$RADIUS_READER" \
      "$REPRO_ROOT/results_jcc/calibration/lj${n}.json")
  fi
  for arm in shared control; do
    log="$OUT_ROOT/submissions/${STAGE}-lj${n}-${arm}-%A_%a.out"
    sbatch \
      --parsable \
      --job-name="jcc-lj${n}-${arm}" \
      --array="0-${last}" \
      --partition="${ELJA_PARTITION:-s-normal}" \
      --account="${ELJA_ACCOUNT:-chem-ui}" \
      --time="${ELJA_TIME:-2-00:00:00}" \
      --cpus-per-task=5 \
      --mem="${ELJA_MEM:-16G}" \
      --output="$log" \
      --export="ALL,CATALOG_CAMPAIGN=${CAMPAIGN},SEED_OFFSET_BASE=${SEED_OFFSET_BASE}" \
      "$RUNNER" "$n" 4000000 "$arm" slurm-array "$radius"
  done
done
