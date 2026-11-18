#!/usr/bin/env bash
# Copyright (c) anneal developers
# Append one same-session current-vs-new recommended row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/morse-weak2
SYSTEM=""
ARM=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM=$2; shift 2 ;;
    --arm) ARM=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SYSTEM" && -z "$ARM" && -z "$SEED" ]]; then
  exec "${PYTHON:-python3}" "$HERE/summarize.py"
fi

mkdir -p "$WORK/runs" "$WORK/home"
ROWS=$ROOT/bench/morse-weak2/rows.csv
if [[ -f $ROWS ]] && grep -q "^${SYSTEM},${ARM},${SEED}," "$ROWS"; then
  echo "keep existing $SYSTEM $ARM seed $SEED"
  exit 0
fi

: "${MORSE_WEAK2_BASE_REVISION:?}"
: "${MORSE_WEAK2_FIX_REVISION:?}"
export MORSE_WEAK2_BASELINE=${MORSE_WEAK2_BASELINE:-$WORK/morse_weak_baseline}
export MORSE_WEAK2_LJ_BASELINE=${MORSE_WEAK2_LJ_BASELINE:-$WORK/h2h_gmin_baseline}
export MORSE_WEAK2_AU_BASELINE=${MORSE_WEAK2_AU_BASELINE:-$WORK/ase_panel_baseline}
export MORSE_WEAK2_CANDIDATE=${MORSE_WEAK2_CANDIDATE:-/work/target-panel/release/examples/morse_weak}
PY=${PYTHON:-python3}
OUT=$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stdout
ERR=$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stderr

sudo -n unshare --uts sh -c "hostname bench; \
  export HOME=$WORK/home; \
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1; \
  export OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1; \
  export MORSE_WEAK2_BASE_REVISION=$MORSE_WEAK2_BASE_REVISION; \
  export MORSE_WEAK2_FIX_REVISION=$MORSE_WEAK2_FIX_REVISION; \
  export MORSE_WEAK2_BASELINE=$MORSE_WEAK2_BASELINE; \
  export MORSE_WEAK2_LJ_BASELINE=$MORSE_WEAK2_LJ_BASELINE; \
  export MORSE_WEAK2_AU_BASELINE=$MORSE_WEAK2_AU_BASELINE; \
  export MORSE_WEAK2_CANDIDATE=$MORSE_WEAK2_CANDIDATE; \
  exec taskset -c 0 $PY $HERE/run_one.py $SYSTEM $ARM $SEED" >"$OUT" 2>"$ERR"

python3 - "$OUT" "$ROWS" <<'PY'
import sys
from pathlib import Path

line = Path(sys.argv[1]).read_text().strip()
fields = line.split(",")
if len(fields) != 17 or fields[12] != "bench":
    raise SystemExit("invalid morse-weak2 row")
path = Path(sys.argv[2])
header = (
    "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,"
    "code_revision,host,stop,restarts,orbits,hops"
)
key = ",".join(fields[:3]) + ","
text = path.read_text() if path.exists() else ""
if key in text:
    raise SystemExit(0)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a") as handle:
    if not text:
        handle.write(header + "\n")
    handle.write(line + "\n")
PY
