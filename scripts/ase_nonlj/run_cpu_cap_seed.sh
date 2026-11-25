#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SYSTEM=${1:-}
SEED=${2:-}
PY=${PYTHON:-python3}
PLAN=$($PY "$HERE/cpu_cap_spec.py" "$SYSTEM" "$SEED")
eval "$($PY - "$PLAN" <<'PY'
import json
import shlex
import sys

plan = json.loads(sys.argv[1])
for key in (
    "source", "runner", "rows", "shape_rows", "work_env", "binary_env",
    "default_binary", "revision_env", "revision", "budget",
):
    print(f"{key.upper()}={shlex.quote(str(plan[key]))}")
PY
)"

WORK=${!WORK_ENV:-$ROOT/.work/ase-mh-cpu-cap}
ROWS=${ASE_CPU_CAP_ROWS:-$ROWS}
SHAPE_ROWS=${ASE_CPU_CAP_SHAPE_ROWS:-$SHAPE_ROWS}
BIN=${!BINARY_ENV:-$DEFAULT_BINARY}
REV=${!REVISION_ENV:-$REVISION}
CPU_CAP=${ASE_CPU_CAP_SECONDS:-43200}
BUDGET=${ASE_CPU_CAP_BUDGET:-$BUDGET}
RUN=$WORK/runs/$SYSTEM-ase_mh-$SEED
STDOUT=$WORK/runs/$SYSTEM-ase_mh-$SEED.stdout
STDERR=$WORK/runs/$SYSTEM-ase_mh-$SEED.stderr
USER_SITE=$($PY -c 'import site; print(site.getusersitepackages())')

if [[ ${ASE_CPU_CAP_DRY_RUN:-0} == 1 ]]; then
  printf '%s\n' "$PLAN"
  exit 0
fi

mkdir -p "$(dirname "$ROWS")"
if [[ ! -e "$ROWS" ]]; then
  printf '%s\n' \
    "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,code_revision,host,stop,cap_s" \
    >"$ROWS"
fi

if $PY - "$ROWS" "$SHAPE_ROWS" "$SYSTEM" "$SEED" "$REV" "$CPU_CAP" <<'PY'
import csv
import sys
from pathlib import Path

path, shape_path, system, seed, revision, cpu_cap = sys.argv[1:]
with open(path, newline="") as handle:
    performance_found = any(
        row["system"] == system
        and row["arm"] == "ase_mh"
        and row["seed"] == seed
        and row["code_revision"] == revision
        and row["cap_s"]
        and float(row["cap_s"]) == float(cpu_cap)
        for row in csv.DictReader(handle)
    )
shape_found = False
if Path(shape_path).exists():
    with open(shape_path, newline="") as handle:
        shape_found = any(
            row["system"] == system
            and row["arm"] == "ase_mh"
            and row["seed"] == seed
            and row["code_revision"] == revision
            for row in csv.DictReader(handle)
        )
raise SystemExit(0 if performance_found and shape_found else 1)
PY
then
  echo "keep existing CPU-capped $SYSTEM ase_mh seed $SEED"
  exit 0
fi

mkdir -p "$RUN" "$WORK/home" "$WORK/numba"
if ! grep -q "^$SYSTEM,ase_mh,$SEED," "$STDOUT" 2>/dev/null \
  || ! grep -q '^shape_json=' "$STDOUT" 2>/dev/null; then
  COMMON="LJOS_SEAT=${LJOS_SEAT:-benchbot} \
HOME=$WORK/home NUMBA_CACHE_DIR=$WORK/numba PYTHONPATH=$USER_SITE \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
OMP_PROC_BIND=TRUE OMP_PLACES=cores PYTHONUNBUFFERED=1"
  case "$SOURCE" in
    morse)
      RUN_ENV="ASE_NONLJ_WORK=$WORK/runs ASE_NONLJ=$BIN \
ASE_NONLJ_WALL_CAP= ASE_NONLJ_CPU_CAP=$CPU_CAP \
ASE_NONLJ_REVISION=$REV"
      RUN_ARGS="$SYSTEM ase_mh $SEED"
      ;;
    panel)
      RUN_ENV="ASE_PANEL_WORK=$WORK/runs ASE_PANEL=$BIN \
ASE_PANEL_WALL_CAP= ASE_PANEL_CPU_CAP=$CPU_CAP \
ASE_PANEL_BUDGET=$BUDGET ASE_PANEL_REVISION=$REV"
      RUN_ARGS="$SYSTEM ase_mh $SEED"
      ;;
    lj75)
      RUN_ENV="ASE_LJ75_WORK=$WORK/runs ANNEAL_LJ75=$BIN \
ASE_LJ75_WALL_CAP= ASE_LJ75_CPU_CAP=$CPU_CAP \
ASE_LJ75_BUDGET=$BUDGET ASE_LJ75_REVISION=$REV"
      RUN_ARGS="ase_mh $SEED"
      ;;
  esac
  sudo -n unshare --uts sh -c "hostname bench; \
    exec sudo -u $(id -un) env $COMMON $RUN_ENV \
    taskset -c 0 $PY $RUNNER $RUN_ARGS" >"$STDOUT" 2>"$STDERR"
fi

$PY - "$SYSTEM" "$SEED" "$REV" "$CPU_CAP" "$STDOUT" "$ROWS" "$SHAPE_ROWS" <<'PY'
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

system, seed, revision, cpu_cap, stdout_path, rows_path, shape_path = sys.argv[1:]
lines = Path(stdout_path).read_text().splitlines()
row = next(line for line in lines if line.startswith(f"{system},ase_mh,{seed},"))
fields = row.split(",")
if len(fields) != 15:
    raise SystemExit("performance row is missing fields")
if fields[11] != revision or fields[12] != "bench":
    raise SystemExit("performance row has unexpected provenance")
if fields[13] not in {"hit", "budget", "cap reached (12 h CPU)"}:
    raise SystemExit("performance row has an unexpected stop reason")
if float(fields[14]) != float(cpu_cap):
    raise SystemExit("performance row has an unexpected CPU cap")

shape = json.loads(next(line.split("=", 1)[1] for line in lines if line.startswith("shape_json=")))
if (
    shape["system"] != system
    or shape["arm"] != "ase_mh"
    or str(shape["seed"]) != seed
    or shape["code_revision"] != revision
):
    raise SystemExit("shape row does not match the performance row")


def append_row(path: Path, header: str, line: str) -> None:
    content = path.read_bytes() if path.exists() else b""
    if content and not content.endswith(b"\n"):
        content = content[:content.rfind(b"\n") + 1]
    if not content:
        content = (header + "\n").encode()
    content += (line + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


columns = [
    "system", "arm", "seed", "code_revision", "nproc", "model_name",
    "memory_bytes", "core_affinity", "cpu_s", "wall_s",
    "cpu_wall_ratio", "vm_stayed_live",
]
shape_rows = Path(shape_path)
shape_rows.parent.mkdir(parents=True, exist_ok=True)
shape_found = False
if shape_rows.exists():
    with shape_rows.open(newline="") as handle:
        shape_found = any(
            item["system"] == system
            and item["arm"] == "ase_mh"
            and item["seed"] == seed
            and item["code_revision"] == revision
            for item in csv.DictReader(handle)
        )
if not shape_found:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    record = {key: shape[key] for key in columns}
    record["vm_stayed_live"] = str(record["vm_stayed_live"]).lower()
    writer.writerow(record)
    append_row(shape_rows, ",".join(columns), output.getvalue().rstrip("\n"))

rows = Path(rows_path)
rows.parent.mkdir(parents=True, exist_ok=True)
with rows.open(newline="") as handle:
    performance_found = any(
        item["system"] == system
        and item["arm"] == "ase_mh"
        and item["seed"] == seed
        and item["code_revision"] == revision
        and item["cap_s"]
        and float(item["cap_s"]) == float(cpu_cap)
        for item in csv.DictReader(handle)
    )
if not performance_found:
    append_row(
        rows,
        "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,"
        "inside_s,outside_s,code_revision,host,stop,cap_s",
        row,
    )
PY
