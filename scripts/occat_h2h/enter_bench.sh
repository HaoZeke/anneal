#!/usr/bin/env bash
# Copyright (c) anneal developers
# OCCat, anneal, and GMIN. One UTS namespace, hostname bench, core 0.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/occat-h2h

if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK/runs" "$WORK/home" "$WORK/tmp" "$ROOT/bench/occat-h2h"

if [[ ! -x /tmp/OCCat_release/bin/occat ]]; then
  rm -rf /tmp/OCCat_release
  git clone --depth 1 https://github.com/LEDlamar/OCCat_release.git /tmp/OCCat_release
fi

if [[ ! -x $WORK/target/release/examples/h2h_gmin_proof || ! -x $WORK/target/release/examples/h2h_metal ]]; then
  mkdir -p "$WORK/target" "$WORK/rustc-tmp"
  (
    cd "$ROOT"
    export CARGO_TARGET_DIR=$WORK/target
    export TMPDIR=$WORK/rustc-tmp
    cargo +1.88.0 build --release --locked --example h2h_gmin_proof --example h2h_metal -j 1
  )
fi

if [[ ! -x /work/gmin-h2h/bin/GMIN ]]; then
  bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
fi

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
export GOMAXPROCS=1
cd $WORK
exec taskset -c 0 python3 -u $HERE/run_campaign.py \\
  --seeds $ROOT/bench/occat-h2h/seeds.csv \\
  --summary $ROOT/bench/occat-h2h/summary.csv \\
  --tests $ROOT/bench/occat-h2h/tests.csv \\
  --work $WORK/runs
EOF
chmod +x "$WORK/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/occat-h2h/inside.sh "$@"' _ "$@"
