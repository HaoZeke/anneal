#!/bin/bash
# Run the opt-in search arms on the GFN2 conformer campaign, one set after another.
# Usage: ORGANIC_BENCH=BINARY experiments/gap_rerun.sh CAMPAIGN OUT ARM...
set -euo pipefail
CAMPAIGN=${1:?campaign directory} OUT=${2:?output directory}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/tmp/chem/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XTB_EXE=${XTB_EXE:-/tmp/chem/bin/xtb}
export ORGANIC_BENCH=${ORGANIC_BENCH:?organic_bench binary}
export ASE_ENGINE=xtb-cli XTB_ACC=0.01
PY=${PYTHON:-/tmp/chem/bin/python}
mkdir -p "$OUT"
"$PY" "$HERE/gap_rerun.py" --campaign "$CAMPAIGN" jobs > "$OUT/jobs.txt"
for arm in "$@"; do
  echo "$arm start $(date -u +%FT%TZ)" >> "$OUT/sets.log"
  sed "s/^/$arm /" "$OUT/jobs.txt" |
    xargs -P "${SLOTS:-7}" -L 1 "$PY" "$HERE/gap_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" run \
      >> "$OUT/driver-$arm.log" 2>&1
  echo "$arm end $(date -u +%FT%TZ)" >> "$OUT/sets.log"
done
