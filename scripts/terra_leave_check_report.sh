#!/usr/bin/env bash
# Read the Leave contract off a finished occupancy run.
#
#     terra_leave_check_report.sh ENSEMBLE_DIR
set -euo pipefail
OUT=${1:?ensemble directory}
COORD="$OUT/coordinator.jsonl"
test -s "$COORD"
python3 - "$COORD" "$OUT" <<'PY'
import json
import pathlib
import sys

coordinator = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
gt = []
for line in coordinator.read_text().splitlines():
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("kind") == "occupancy_gt":
        gt.append(record)
if not gt:
    print("no occupancy_gt records")
    raise SystemExit(1)
last = gt[-1]
print("occupancy_gt records", len(gt))
for key in (
    "packing_n",
    "packing_n1",
    "packing_p0",
    "cell_n",
    "cell_n1",
    "cell_p0",
    "sparsified_n",
    "sparsified_n1",
    "packing_sat",
    "landfold_communities",
    "landfold_holes",
    "families",
    "min_families",
    "stop",
):
    print(f"  {key} = {last.get(key)}")
lying = [
    record
    for record in gt
    if record.get("packing_sat") and (record.get("cell_n1") or 0) > 0
]
print("records claiming saturation with singleton cells:", len(lying))
communities = sorted({record.get("landfold_communities") for record in gt})
print("landfold_communities seen:", communities)

# Each replica reports its own floor on the last line of its worker log.
bests = []
for worker in sorted(root.glob("workers/replica-*.out")):
    for line in worker.read_text().splitlines():
        if "personal best" not in line:
            continue
        for field in line.split():
            try:
                bests.append(float(field))
            except ValueError:
                continue
            break
if bests:
    bests.sort()
    print("replicas reporting", len(bests))
    print("best energy", f"{bests[0]:.6f}")
    print("distinct replica floors", len({round(value, 6) for value in bests}))
    print("below the LJ75 ico floor -396.282249:", sum(1 for value in bests if value < -396.282249))
PY
