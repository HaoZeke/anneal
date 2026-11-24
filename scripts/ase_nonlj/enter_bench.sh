#!/usr/bin/env bash
# Copyright (c) anneal developers
# Non-LJ (Morse) ASE basin hopping and minima hopping against
# Config::for_cluster and Config::recommended.
#   enter_bench.sh --system m38r6 --arm for_cluster --seed 2000
# No --arm rewrites the summary and README from committed rows.
# An existing seed is left untouched. Recommended, ASE basin hopping and
# minima hopping stop at three times the slowest for_cluster wall.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/ase-nonlj
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
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
  exec "$PIXI" run --locked --frozen -e chem "${PYTHON:-python3}" "$HERE/summarize.py"
fi

case "$SYSTEM" in
  m38r6|m38r14|m55r6|m55r14) ;;
  *) echo "unknown system $SYSTEM" >&2; exit 2 ;;
esac
case "$ARM" in
  for_cluster|recommended|ase_bh|ase_mh) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SEED" in
  2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011) ;;
  *) echo "seed $SEED is outside 2000-2011" >&2; exit 2 ;;
esac

BIN=${ASE_NONLJ:-/work/target-panel/release/examples/ase_nonlj}
REV=${ASE_NONLJ_REVISION:-9a30f3190b3bb7040c675a904f8388eb68224151}
mkdir -p "$WORK/runs" "$WORK/home" "$WORK/numba"
ROWS=${ASE_NONLJ_ROWS:-$ROOT/bench/ase-nonlj/rows.csv}
if [[ -f $ROWS ]] && grep -q "^${SYSTEM},${ARM},${SEED},.*,$REV,bench," "$ROWS"; then
  echo "keep existing $SYSTEM $ARM seed $SEED"
  exit 0
fi
PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" python3 - "$ROWS" << 'PY'
import sys
from pathlib import Path

from machine_provenance import require_csv_schema

require_csv_schema(
    Path(sys.argv[1]),
    "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,"
    "outside_s,code_revision,host,stop,cap_s,machine_id,cpu_model".split(","),
)
PY

CAP=""
if [[ $ARM != for_cluster ]]; then
  CAP=$("$PIXI" run --locked --frozen -e chem "${PYTHON:-python3}" - "$ROWS" "$SYSTEM" "$REV" << 'PY'
import csv
import sys
from pathlib import Path

path, system, revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(path.open())) if path.exists() else []
base = [
    row for row in rows
    if row["arm"] == "for_cluster"
    and row["system"] == system
    and row["host"] == "bench"
    and row["code_revision"] == revision
]
seeds = tuple(sorted(int(row["seed"]) for row in base))
if seeds != tuple(range(2000, 2012)):
    raise SystemExit(f"wall cap needs for_cluster seeds 2000-2011 for {system}, found {seeds}")
print(f"{3.0 * max(float(row['wall_s']) for row in base):.6f}")
PY
)
fi

if [[ ! $REV =~ ^[0-9a-f]{40}$ ]]; then
  echo "ASE_NONLJ_REVISION must be a full lowercase commit SHA" >&2
  exit 2
fi
if [[ ! -f $ROOT/pixi.lock ]]; then
  echo "pixi.lock is required" >&2
  exit 2
fi
if ! git -C "$ROOT" cat-file -e "${REV}^{commit}" 2>/dev/null; then
  echo "revision $REV is not in this checkout; fetch it or set ASE_NONLJ_REVISION to a reachable commit SHA" >&2
  exit 2
fi
PYRUN="$PIXI run --locked --frozen -e chem ${PYTHON:-python3}"
sudo -n unshare --uts sh -c "hostname bench; \
  export HOME=$WORK/home NUMBA_CACHE_DIR=$WORK/numba; \
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1; \
  export OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1; \
  export ASE_NONLJ_WORK=$WORK/runs ASE_NONLJ=$BIN ASE_NONLJ_WALL_CAP=$CAP ASE_NONLJ_REVISION=$REV; \
  exec taskset -c 0 $PYRUN $HERE/run_one.py $SYSTEM $ARM $SEED" \
  > "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stdout" 2> "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stderr"

$PYRUN - "$SYSTEM" "$ARM" "$SEED" "$WORK/runs/${SYSTEM}-${ARM}-${SEED}.stdout" "$ROWS" << 'PY'
import csv
import sys
from pathlib import Path

system, arm, seed, stdout_path, rows_path = sys.argv[1:]
lines = Path(stdout_path).read_text().splitlines()
row = next(line for line in lines if line.startswith(system + ","))
fields = next(csv.reader([row]))
if fields[0] != system or fields[1] != arm or fields[2] != seed or fields[12] != "bench":
    raise SystemExit("row does not match the requested seed")
if len(fields) != 17:
    raise SystemExit("row is missing provenance fields")
path = Path(rows_path)
header = "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,code_revision,host,stop,cap_s,machine_id,cpu_model"
text = path.read_text() if path.exists() else ""
revision = fields[11]
if path.exists() and any(
    old["system"] == system
    and old["arm"] == arm
    and old["seed"] == seed
    and old["code_revision"] == revision
    for old in csv.DictReader(text.splitlines())
):
    print(f"keep existing {system} {arm} seed {seed}")
    raise SystemExit(0)
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists() or path.stat().st_size == 0:
    path.write_text(header + "\n" + row + "\n")
else:
    if text.splitlines()[0] != header:
        raise SystemExit(f"{path} has a legacy schema; use a new campaign table")
    with path.open("a") as handle:
        handle.write(row + "\n")
PY
