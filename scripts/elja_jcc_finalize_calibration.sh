#!/usr/bin/env bash
# Validate all hard-LJ development pools and materialize fixed census radii.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_finalize_calibration.sh must run under Slurm" >&2
  exit 1
fi

REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
PYTHON=${JCC_PYTHON:-$HOME/rgpot/.pixi/envs/xtbbld/bin/python}
[[ -x $PYTHON ]] || { echo "missing calibration Python: $PYTHON" >&2; exit 1; }
cd "$REPRO_ROOT"
"$PYTHON" workflow/jcc/calibrate_census.py \
  --config config/jcc/census_calibration.json

for n in 75 98 102 104; do
  "$PYTHON" -m json.tool "results_jcc/calibration/lj${n}.json" >/dev/null
done
sha256sum \
  results_jcc/calibration/lj75.json \
  results_jcc/calibration/lj98.json \
  results_jcc/calibration/lj102.json \
  results_jcc/calibration/lj104.json \
  > results_jcc/calibration/SHA256SUMS
touch results_jcc/calibration/TERMINAL_OK
echo "CALIBRATION_FINALIZE_OK job=${SLURM_JOB_ID}"
