#!/bin/bash
# Preset rerun of the GFN2 conformer campaign at its CREST budgets, then the table.
# Usage: ORGANIC_BENCH=BINARY experiments/robustness_rerun.sh CAMPAIGN OUT
set -euo pipefail
CAMPAIGN=${1:?campaign directory} OUT=${2:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
CHEM=${CHEM_PREFIX:-${TMPDIR:-/tmp}/chem}
export PATH=$CHEM/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XTB_EXE=${XTB_EXE:-$CHEM/bin/xtb}
export ORGANIC_BENCH=${ORGANIC_BENCH:?organic_bench binary}
export ASE_ENGINE=xtb-cli XTB_ACC=0.01
PY=${PYTHON:-$CHEM/bin/python}
mkdir -p "$OUT"
"$PY" "$HERE/robustness_rerun.py" --campaign "$CAMPAIGN" jobs > "$OUT/jobs.txt"
xargs -P "${SLOTS:-7}" -L 1 "$PY" "$HERE/robustness_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" run \
  < "$OUT/jobs.txt" > "$OUT/driver.log" 2>&1
"$PY" "$HERE/robustness_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" table > "$OUT/table.md"
