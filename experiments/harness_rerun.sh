#!/bin/bash
# Preset rerun of the GFN2 conformer campaign after the harness change, then the table.
# Usage: ORGANIC_BENCH=BINARY experiments/harness_rerun.sh CAMPAIGN BEFORE OUT
set -euo pipefail
CAMPAIGN=${1:?campaign directory} BEFORE=${2:?robustness rerun directory} OUT=${3:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/tmp/chem/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XTB_EXE=${XTB_EXE:-/tmp/chem/bin/xtb}
export ORGANIC_BENCH=${ORGANIC_BENCH:?organic_bench binary}
export ASE_ENGINE=xtb-cli XTB_ACC=0.01
PY=${PYTHON:-/tmp/chem/bin/python}
mkdir -p "$OUT"
"$PY" "$HERE/harness_rerun.py" --campaign "$CAMPAIGN" jobs > "$OUT/jobs.txt"
xargs -P "${SLOTS:-7}" -L 1 "$PY" "$HERE/harness_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" run \
  < "$OUT/jobs.txt" > "$OUT/driver.log" 2>&1
"$PY" "$HERE/harness_rerun.py" --campaign "$CAMPAIGN" --before "$BEFORE" --out "$OUT" table \
  > "$OUT/table.md"
