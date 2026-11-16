#!/usr/bin/env bash
# Copyright (c) anneal developers
# LJ75 trail leave against Config::for_cluster.
# No arguments rewrite the summary from the committed rows.
# One seed appends a row and leaves an existing seed untouched:
#   enter_bench.sh --arm base --seed 2000
#   enter_bench.sh --arm leave --seed 2000
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/leave-lj75
ARM=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ARM" && -z "$SEED" ]]; then
  exec python3 "$HERE/summarize.py"
fi
if [[ -z "$ARM" || -z "$SEED" ]]; then
  echo "pass both --arm and --seed" >&2
  exit 2
fi
case "$ARM" in
  base|leave) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SEED" in
  2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011) ;;
  *) echo "seed $SEED is outside 2000-2011" >&2; exit 2 ;;
esac

mkdir -p "$WORK/bin" "$WORK/runs" "$WORK/scratch"
cargo_home=${CARGO_HOME:-$HOME/.cargo}
export CARGO_TARGET_DIR=$WORK/target
export TMPDIR=$WORK/scratch
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
(cd "$ROOT" && cargo +1.88.0 build --release --locked --example leave_gate -j 4)

sudo -n unshare --uts sh -c "hostname bench; exec taskset -c 0 $WORK/target/release/examples/leave_gate lj75 $ARM 4000000 $SEED" \
  > "$WORK/runs/${ARM}-${SEED}.stdout" 2> "$WORK/runs/${ARM}-${SEED}.stderr"

python3 - "$ARM" "$SEED" "$WORK/runs/${ARM}-${SEED}.stdout" "$ROOT" << 'PY'
import sys
from pathlib import Path

arm, seed, stdout_path, root = sys.argv[1:]
text = Path(stdout_path).read_text().splitlines()
boot = next(line.split("=", 1)[1] for line in text if line.startswith("boot_id="))
host = next(line.split("=", 1)[1] for line in text if line.startswith("hostname="))
if host != "bench":
    raise SystemExit("row host is not bench")
fields = dict(part.split("=", 1) for part in next(line for line in text if line.startswith("system=")).split())
if fields["arm"] != arm or fields["seed"] != seed:
    raise SystemExit("stdout does not match the requested seed")
evals = "" if fields["first_calls"] == "-" else fields["first_calls"]
header = "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,boot_id,hostname"
row = ",".join([
    fields["system"], fields["arm"], fields["seed"], fields["budget"], fields["hit"],
    evals, fields["charged"], fields["best"], fields["wall_s"], fields["inside_s"],
    fields["outside_s"], boot, host,
])
path = Path(root) / "bench" / "leave-lj75" / "rows.csv"
if path.exists():
    existing = path.read_text().splitlines()
else:
    existing = [header]
key = f",{arm},{seed},"
if any(key in line for line in existing):
    print(f"keep existing {arm} seed {seed}")
    raise SystemExit(0)
if existing == [header] or not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + row + "\n")
else:
    with path.open("a") as handle:
        handle.write(row + "\n")
PY
