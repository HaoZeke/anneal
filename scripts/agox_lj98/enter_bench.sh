#!/usr/bin/env bash
# Copyright (c) anneal developers
# LJ98 head-to-head. One UTS namespace, hostname bench, one thread.
# No arguments rewrite the summary from the committed rows.
# One seed appends a row:
#   enter_bench.sh --code agox --seed 6400
#   enter_bench.sh --code agox-documented --seed 6400
#   enter_bench.sh --code anneal --seed 6400
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/agox-lj98
CODE=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --code) CODE=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CODE" && -z "$SEED" ]]; then
  exec python3 "$HERE/summarize.py"
fi
if [[ -z "$CODE" || -z "$SEED" ]]; then
  echo "pass both --code and --seed" >&2
  exit 2
fi
case "$CODE" in
  agox|agox-documented|anneal) ;;
  *) echo "unknown code $CODE" >&2; exit 2 ;;
esac
case "$SEED" in
  6400|6401|6402|6403|6404|6405) ;;
  *) echo "seed $SEED is outside 6400-6405" >&2; exit 2 ;;
esac

mkdir -p "$WORK/bin" "$WORK/runs" "$WORK/home" "$WORK/scratch"
if [[ "$CODE" == "anneal" ]]; then
  cargo_home=${CARGO_HOME:-$HOME/.cargo}
  export CARGO_TARGET_DIR=$WORK/target
  export TMPDIR=$WORK/scratch
  export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
  (cd "$ROOT" && cargo +1.88.0 build --release --locked --example h2h_arm3 -j 4)
fi

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/scratch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LJ_BOARD_ROOT=results/lj-board
cd $ROOT
host=\$(cat /proc/sys/kernel/hostname)
if [ "\$host" != bench ]; then
  echo "uts hostname is not bench" >&2
  exit 2
fi
echo hostname=\$host
if [ "$CODE" = anneal ]; then
  rev=\$(git rev-parse HEAD)
  echo code_revision=\$rev
  exec $WORK/target/release/examples/h2h_arm3 lj98 default 4000000 $SEED
fi
exec python3 scripts/lj_board/run_arm.py $CODE 98 $SEED
EOF
chmod +x "$WORK/bin/inside.sh"

sudo -n unshare --uts sh -c 'hostname bench; exec /work/agox-lj98/bin/inside.sh' \
  > "$WORK/runs/${CODE}-${SEED}.stdout" 2> "$WORK/runs/${CODE}-${SEED}.stderr"

python3 - "$CODE" "$SEED" "$WORK/runs/${CODE}-${SEED}.stdout" "$ROOT" << 'PY'
import csv
import sys
from pathlib import Path

code, seed, stdout_path, root = sys.argv[1:]
text = Path(stdout_path).read_text()
fields = {}
for line in text.splitlines():
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
if fields.get("hostname") != "bench":
    raise SystemExit("row host is not bench")
if code == "anneal":
    revision = fields["code_revision"]
    hit = int(fields["hit"])
    charged = int(fields["charged"])
    evals = "" if fields.get("first_calls", "-") == "-" else fields["first_calls"]
    inside = fields.get("inside_s", "")
    outside = fields.get("outside_s", "")
    best = f"{float(fields['best']):.8f}"
    wall = f"{float(fields['wall_s']):.6f}"
else:
    log = Path(root) / "results" / "lj-board" / code / f"lj98-seed{seed}.log"
    payload = {}
    for line in log.read_text().splitlines():
        if "=" in line and " " not in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            payload[key] = value
    note = next(line for line in log.read_text().splitlines() if line.startswith("agox="))
    note_fields = {}
    for part in note.split():
        if "=" in part:
            key, value = part.split("=", 1)
            note_fields[key] = value
    revision = note_fields["commit"]
    hit = int(payload["hit"])
    charged = int(payload["charged"])
    evals = payload["calls"] if hit else ""
    inside = f"{float(payload['inside_s']):.6f}"
    outside = f"{float(payload['outside_s']):.6f}"
    best = f"{float(payload['best']):.6f}"
    wall = f"{float(payload['wall_s']):.6f}"
header = [
    "system", "code", "seed", "budget", "hit", "evals_to_hit", "charged",
    "wall_s", "inside_s", "outside_s", "best", "code_revision", "host",
]
row = {
    "system": "lj98",
    "code": code,
    "seed": seed,
    "budget": "4000000",
    "hit": str(hit),
    "evals_to_hit": evals,
    "charged": str(charged),
    "wall_s": wall,
    "inside_s": inside,
    "outside_s": outside,
    "best": best,
    "code_revision": revision,
    "host": "bench",
}
path = Path(root) / "bench" / "agox-lj98" / "rows.csv"
rows = []
if path.exists():
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
kept = [item for item in rows if not (item["code"] == code and item["seed"] == seed)]
kept.append(row)
kept.sort(key=lambda item: (item["code"], int(item["seed"])))
with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(kept)
PY
