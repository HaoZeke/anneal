#!/usr/bin/env bash
# Copyright (c) anneal developers
# The A/B gate on the hit panels. Builds both arms, then runs the panel in one
# UTS namespace with hostname bench, one process on core 0, and every other
# process moved to the remaining cores.
#
#   PANEL_B_REV=<commit> enter_bench.sh [run_panel.py options]
#   enter_bench.sh --summarize
#
#   PANEL_A_REV       arm A, default the develop commit the gate was written for
#   PANEL_B_REV       arm B, that commit with the three binary Lennard-Jones
#                     patches applied; required to run
#   PANEL_B_TREE      optional tree hash arm B must have
#   PANEL_WORK        scratch directory, default .bench/blj-panel in the repo
#   PANEL_OUT         where rows, summary, pairs and builds go, default this directory
#   RGPOT_SC_LIB      the Sutton-Chen library; built when unset
#   PANEL_SKIP_BUILD  1 reuses the binaries already built
#   PANEL_EVACUATE    0 leaves other processes where they are
#
# Finished runs already in rows.csv are kept. Nothing is run twice.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${PANEL_WORK:-$ROOT/.bench/blj-panel}
OUT=${PANEL_OUT:-$HERE}
A_REV=${PANEL_A_REV:-bc16d35cf13e71c2edaf343b80fd8c41368631d4}

if [ "${1:-}" = "--summarize" ]; then
  shift
  exec python3 -B "$HERE/run_panel.py" --summarize \
    --rows "$OUT/rows.csv" --summary "$OUT/summary.csv" --pairs "$OUT/pairs.csv" \
    --readme "$OUT/README.md" "$@"
fi

: "${PANEL_B_REV:?set PANEL_B_REV to the commit with the binary Lennard-Jones patches}"
export PANEL_WORK=$WORK PANEL_A_REV=$A_REV PANEL_B_REV
if [ "${PANEL_SKIP_BUILD:-0}" != 1 ]; then
  bash "$HERE/build_arms.sh"
fi
if [ -n "${PANEL_B_TREE:-}" ]; then
  found=$(git -C "$WORK/arm-b" rev-parse 'HEAD^{tree}')
  if [ "$found" != "$PANEL_B_TREE" ]; then
    echo "arm B tree is $found, expected $PANEL_B_TREE" >&2
    exit 1
  fi
fi

LIB=${RGPOT_SC_LIB:-$WORK/rgpot/lib/librgpot.so}
if [ ! -f "$LIB" ]; then
  bash "$HERE/build_rgpot.sh"
fi

PY=$(command -v python3)
user=$(id -un)
mkdir -p "$WORK/userdir" "$WORK/scratch" "$WORK/logs"
cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/userdir
export TMPDIR=$WORK/scratch
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export RGPOT_SC_LIB=$LIB
export LD_LIBRARY_PATH=$(dirname "$LIB")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd $WORK
exec taskset -c 0 $PY -u $HERE/run_panel.py \\
  --rows $OUT/rows.csv \\
  --summary $OUT/summary.csv \\
  --pairs $OUT/pairs.csv \\
  --readme $OUT/README.md \\
  --builds $OUT/builds.txt \\
  --rgpot-manifest $WORK/rgpot/MANIFEST \\
  --logs $WORK/logs \\
  --bin-a $WORK/bin-a --tree-a $WORK/arm-a \\
  --bin-b $WORK/bin-b --tree-b $WORK/arm-b \\
  "\$@"
EOF
chmod +x "$WORK/inside.sh"

others=""
cpus=$(nproc --all)
if [ "${PANEL_EVACUATE:-1}" = 1 ] && [ "$cpus" -gt 1 ]; then
  others="1-$((cpus - 1))"
fi

if sudo -n true 2> /dev/null; then
  if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
    echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts > /dev/null
  fi
  exec sudo -n unshare --uts sh -c '
    hostname bench
    others=$1; user=$2; script=$3; shift 3
    if [ -n "$others" ]; then
      for p in /proc/[0-9]*; do
        taskset -a -pc "$others" "${p#/proc/}" > /dev/null 2>&1 || true
      done
    fi
    exec sudo -u "$user" "$script" "$@"
  ' _ "$others" "$user" "$WORK/inside.sh" "$@"
fi
exec unshare --user --map-root-user --uts sh -c 'hostname bench; script=$1; shift; exec "$script" "$@"' _ "$WORK/inside.sh" "$@"
