#!/usr/bin/env bash
# Copyright (c) anneal developers
# ASE basin hopping and minima hopping against Config::for_cluster.
#   enter_bench.sh --system lj38 --arm anneal --seed 2000
# No --arm rewrites that system's summary from the committed rows.
# An existing seed is left untouched. An ASE seed stops at the paper
# call budget or at three times the slowest anneal wall of that system.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/ase-panel
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

case "$SYSTEM" in
  lj38|lj55|lj98|lj104|lj150|au55) ;;
  *) echo "unknown system $SYSTEM" >&2; exit 2 ;;
esac

if [[ -z "$ARM" && -z "$SEED" ]]; then
  exec python3 "$HERE/summarize.py" "$SYSTEM"
fi
case "$ARM" in
  anneal|ase_bh|ase_mh) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SEED" in
  2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011) ;;
  *) echo "seed $SEED is outside 2000-2011" >&2; exit 2 ;;
esac

mkdir -p "$WORK/runs" "$WORK/home" "$WORK/numba"
ROWS=$ROOT/bench/ase-$SYSTEM/rows.csv
if [[ -f $ROWS ]] && grep -q ",${ARM},${SEED}," "$ROWS"; then
  echo "keep existing $SYSTEM $ARM seed $SEED"
  exit 0
fi

CAP=""
if [[ $ARM != anneal ]]; then
  CAP=$(python3 - "$ROWS" "$SYSTEM" << 'PY'
import csv
import sys
from pathlib import Path

path, system = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(path.open())) if path.exists() else []
anneal = [row for row in rows if row["arm"] == "anneal" and row["system"] == system and row["host"] == "bench"]
seeds = tuple(sorted(int(row["seed"]) for row in anneal))
if seeds != tuple(range(2000, 2012)):
    raise SystemExit(f"wall cap needs anneal seeds 2000-2011, found {seeds}")
print(f"{3.0 * max(float(row['wall_s']) for row in anneal):.6f}")
PY
)
fi

PANEL=${ASE_PANEL:-/work/target-panel/release/examples/ase_panel}
ASG=${ASG_EVAL:-/work/target-panel/release/examples/asg_eval}
PY=${PYTHON:-python3}
sudo -n unshare --uts sh -c "hostname bench; \
  export HOME=$WORK/home NUMBA_CACHE_DIR=$WORK/numba; \
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1; \
  export OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1; \
  export ASE_PANEL_WORK=$WORK/runs ASE_PANEL=$PANEL ASG_EVAL=$ASG ASE_PANEL_WALL_CAP=$CAP; \
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
if len(fields) != 15:
    raise SystemExit("row is missing stop and cap_s")
path = Path(rows_path)
header = "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,code_revision,host,stop,cap_s"
text = path.read_text() if path.exists() else ""
if f",{arm},{seed}," in text:
    print(f"keep existing {system} {arm} seed {seed}")
    raise SystemExit(0)
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists() or path.stat().st_size == 0:
    path.write_text(header + "\n" + row + "\n")
else:
    with path.open("a") as handle:
        handle.write(row + "\n")
PY
