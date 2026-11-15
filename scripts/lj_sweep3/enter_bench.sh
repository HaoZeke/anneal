#!/usr/bin/env bash
# Copyright (c) anneal developers
# Same-session develop against this tree. Hostname bench, one process.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/lj-sweep3
mkdir -p "$WORK/home" "$WORK/tmp" "$WORK/logs" "$ROOT/bench/lj-sweep3"

export LJOS_SEAT=anneal-paper
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1

git -C "$ROOT" rev-parse HEAD > "$WORK/head-rev"
git -C "$WORK/develop-src" rev-parse HEAD > "$WORK/develop-rev"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u ubuntu env LJOS_SEAT=anneal-paper HOME=$WORK/home TMPDIR=$WORK/tmp OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/python3 -u $HERE/run_session.py \
  --head-bin $WORK/head-target/release/examples/h2h_gmin_proof \
  --develop-bin $WORK/develop-target/release/examples/h2h_gmin_proof \
  --head-rev $WORK/head-rev \
  --develop-rev $WORK/develop-rev \
  --rows $ROOT/bench/lj-sweep3/rows.csv \
  --summary $ROOT/bench/lj-sweep3/summary.csv \
  --logs $WORK/logs \
  --gmin-rows $ROOT/bench/lj-sweep2/rows.csv \
  --head-examples $WORK/head-target/release/examples \
  --develop-examples $WORK/develop-target/release/examples \
  --pins $ROOT/bench/lj-sweep3 \
  \"\$@\"" _ "$@"
