#!/usr/bin/env bash
# Validate all hard-LJ development pools and materialize fixed census radii.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_finalize_calibration.sh must run under Slurm" >&2
  exit 1
fi

REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
PYTHON=${JCC_PYTHON:-python}
cd "$REPRO_ROOT"
(cd development/jcc/reference && sha256sum --check --strict SHA256SUMS)
"$PYTHON" workflow/jcc/calibrate_census.py \
  --config config/jcc/census_calibration.yaml

for n in 75 98 102 104; do
  "$PYTHON" -m json.tool "results/jcc/calibration/lj${n}.json" >/dev/null
done
sha256sum \
  results/jcc/calibration/lj75.json \
  results/jcc/calibration/lj98.json \
  results/jcc/calibration/lj102.json \
  results/jcc/calibration/lj104.json \
  > results/jcc/calibration/SHA256SUMS
touch results/jcc/calibration/TERMINAL_OK
echo "CALIBRATION_FINALIZE_OK job=${SLURM_JOB_ID}"
