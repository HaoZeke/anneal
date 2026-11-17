#!/usr/bin/env bash
# Copyright (c) anneal developers
# Seeds 6-11 of (H2O)12, CREST cap, plain arm then plateau arm.
# One UTS namespace, hostname bench, one thread.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REV=$(git -C "$ROOT" rev-parse HEAD)
exec sudo -n unshare --uts sh -c '
  hostname bench
  . /work/h2o20-moves/bench-env.sh
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export WATER12_REVISION='"$REV"'
  exec taskset -c 0 python3 -u '"$HERE"'/panel.py
'
