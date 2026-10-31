#!/bin/bash
# Run the preset, which polishes its records, on the GFN2 conformer campaign's jobs.
# Usage: ORGANIC_BENCH=BINARY experiments/hessopt_rerun.sh CAMPAIGN OUT
set -euo pipefail
CAMPAIGN=${1:?campaign directory} OUT=${2:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/tmp/chem/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XTB_EXE=${XTB_EXE:-/tmp/chem/bin/xtb}
export ORGANIC_BENCH=${ORGANIC_BENCH:?organic_bench binary}
export ASE_ENGINE=xtb-cli XTB_ACC=0.01
PY=${PYTHON:-/tmp/chem/bin/python}
mkdir -p "$OUT"
"$PY" "$HERE/gap_rerun.py" --campaign "$CAMPAIGN" jobs > "$OUT/jobs.txt"
echo "preset start $(date -u +%FT%TZ)" >> "$OUT/sets.log"
xargs -P "${SLOTS:-7}" -L 1 "$PY" "$HERE/hessopt_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" run \
  < "$OUT/jobs.txt" >> "$OUT/driver.log" 2>&1
echo "preset end $(date -u +%FT%TZ)" >> "$OUT/sets.log"
