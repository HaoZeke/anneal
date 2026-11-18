#!/usr/bin/env bash
# Copyright (c) anneal developers
# Same-session baseline (Config::recommended) vs adaptive population
# (Config::recommended_adaptive) on the 12 Morse and 5 LJ clusters.
#   enter_bench.sh --system m80r6 --arm recommended_adaptive --seed 2000
# No arguments regenerates the summary and README from committed rows.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/morse-weak
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

case "$SYSTEM" in
  lj38|lj55|lj75|lj98|lj104|m38r3|m38r6|m38r10|m38r14|m55r3|m55r6|m55r10|m55r14|m75r6|m75r14|m80r6|m80r14) ;;
  *) echo "unknown system $SYSTEM" >&2; exit 2 ;;
esac
case "$ARM" in
  recommended|recommended_adaptive) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SEED" in
  2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011) ;;
  *) echo "seed $SEED is outside 2000-2011" >&2; exit 2 ;;
esac

mkdir -p "$WORK/runs" "$WORK/home"
ROWS=$ROOT/bench/morse-weak/rows.csv
if [[ -f $ROWS ]] && grep -q "^${SYSTEM},${ARM},${SEED}," "$ROWS"; then
  echo "keep existing $SYSTEM $ARM seed $SEED"
  exit 0
fi

BIN=${MORSE_WEAK_BIN:-/work/target-panel/release/examples/morse_weak}
REV=${MORSE_WEAK_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}
PY=${PYTHON:-python3}
sudo -n unshare --uts sh -c "hostname bench; \
  export HOME=$WORK/home; \
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1; \
  export OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1; \
  export MORSE_WEAK_BIN=$BIN MORSE_WEAK_REVISION=$REV; \
  exec taskset -c 0 $PY $HERE/run_one.py $SYSTEM $ARM $SEED" \
  > "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stdout" 2> "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stderr"

python3 - "$SYSTEM" "$ARM" "$SEED" "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stdout" "$ROWS" << 'PY'
import sys
from pathlib import Path

system, arm, seed, stdout_path, rows_path = sys.argv[1:]
lines = Path(stdout_path).read_text().splitlines()
row = next(line for line in lines if line.startswith(system + ","))
fields = row.split(",")
if fields[0] != system or fields[1] != arm or fields[2] != seed or fields[12] != "bench":
    raise SystemExit("row does not match the requested seed")
if len(fields) != 17:
    raise SystemExit(f"expected 17 fields, got {len(fields)}")
path = Path(rows_path)
header = (
    "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,"
    "code_revision,host,stop,restarts,orbits,hops"
)
text = path.read_text() if path.exists() else ""
if f"{system},{arm},{seed}," in text:
    print(f"keep existing {system} {arm} seed {seed}")
    raise SystemExit(0)
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists() or path.stat().st_size == 0:
    path.write_text(header + "\n" + row + "\n")
else:
    with path.open("a") as handle:
        handle.write(row + "\n")
PY
