#!/usr/bin/env bash
# Copyright (c) anneal developers
# Non-LJ (Morse) ASE basin hopping and minima hopping against
# Config::for_cluster and Config::recommended.
#   enter_bench.sh --system m38r6 --arm for_cluster --seed 2000
# No --arm rewrites the summary and README from committed rows.
# An existing for_cluster or recommended seed is left untouched. An
# existing basin-hopping or minima-hopping seed is left untouched when
# its row has no wall cap, and rerun when the stored row has one.
# Those two arms stop at the call budget, the same budget as for_cluster.
# A recommended seed stops at the cap already stored on its rows.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/ase-nonlj
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

mkdir -p "$WORK/runs" "$WORK/home" "$WORK/numba"
ROWS=$ROOT/bench/ase-nonlj/rows.csv
if [[ -f $ROWS ]]; then
  KIND=$(python3 - "$ROWS" "$SYSTEM" "$ARM" "$SEED" << 'PY'
import csv, sys
path, system, arm, seed = sys.argv[1:]
matches = [
    row for row in csv.DictReader(open(path, newline=""))
    if (row["system"], row["arm"], row["seed"]) == (system, arm, seed)
]
if not matches:
    print("missing")
elif any(not row["cap_s"] for row in matches):
    print("keep")
elif arm in ("ase_bh", "ase_mh"):
    print("capped")
else:
    print("keep")
PY
)
  if [[ $KIND == keep ]]; then
    echo "keep existing $SYSTEM $ARM seed $SEED"
    exit 0
  fi
  if [[ $KIND == capped ]]; then
    echo "rerun $SYSTEM $ARM seed $SEED; the stored row has a wall cap"
  fi
fi

CAP=""
if [[ $ARM == recommended ]]; then
  CAP=$(python3 - "$ROWS" "$SYSTEM" << 'PY'
import csv, sys
from pathlib import Path
path, system = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(path.open()))
caps = sorted({
    row["cap_s"] for row in rows
    if row["system"] == system and row["arm"] == "recommended" and row["cap_s"]
})
if len(caps) != 1:
    raise SystemExit(f"the stored recommended wall cap is not unique: {caps}")
print(caps[0])
PY
)
fi

BIN=${ASE_NONLJ:-/work/target-panel/release/examples/ase_nonlj}
REV=${ASE_NONLJ_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}
PY=${PYTHON:-python3}
user_site=$(python3 -c 'import site; print(site.getusersitepackages())')
sudo -n unshare --uts sh -c "hostname bench; \
  exec sudo -u $(id -un) env \
  HOME=$WORK/home NUMBA_CACHE_DIR=$WORK/numba PYTHONPATH=$user_site \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1 \
  ASE_NONLJ_WORK=$WORK/runs ASE_NONLJ=$BIN ASE_NONLJ_WALL_CAP=$CAP ASE_NONLJ_REVISION=$REV \
  taskset -c 0 $PY $HERE/run_one.py $SYSTEM $ARM $SEED" \
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
if f"{system},{arm},{seed}," in text and fields[14] != "":
    print(f"keep existing {system} {arm} seed {seed}")
    raise SystemExit(0)
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists() or path.stat().st_size == 0:
    path.write_text(header + "\n" + row + "\n")
else:
    with path.open("a") as handle:
        handle.write(row + "\n")
PY
