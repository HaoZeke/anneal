#!/bin/bash
# Run the preset, which keeps the covalent graph, on the GFN2 conformer campaign's jobs.
# The graph-rule references must already be frozen by `graph_rerun.py rescore` in OUT.
# Usage: ORGANIC_BENCH=BINARY experiments/graph_rerun.sh CAMPAIGN OUT
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
if [ ! -f "$OUT/graph_refs.json" ]; then
  echo "no frozen references: run graph_rerun.py rescore into $OUT first" >&2
  exit 2
fi
"$PY" "$HERE/gap_rerun.py" --campaign "$CAMPAIGN" jobs > "$OUT/jobs.txt"
echo "preset start $(date -u +%FT%TZ)" >> "$OUT/sets.log"
xargs -P "${SLOTS:-7}" -L 1 "$PY" "$HERE/graph_rerun.py" --campaign "$CAMPAIGN" --out "$OUT" run \
  < "$OUT/jobs.txt" >> "$OUT/driver.log" 2>&1
echo "preset end $(date -u +%FT%TZ)" >> "$OUT/sets.log"
