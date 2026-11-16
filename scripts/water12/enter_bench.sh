#!/usr/bin/env bash
# Copyright (c) anneal developers
# One UTS namespace, hostname bench, one thread. Extra arguments are
# m_molecules, budget, seed, and an output directory. H2H_PLATEAU=1 is
# the opt-in plateau population. H2H_START_XYZ, when set, is that seed's
# start.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BIN=${WATER12_BIN:-/work/water12/bin/gfn2_water_h2h}

if [[ $# -lt 4 ]]; then
  echo "usage: enter_bench.sh molecules budget seed outdir [start.xyz]" >&2
  exit 2
fi

exec sudo -n unshare --uts sh -c '
  hostname bench
  . /work/h2o20-moves/bench-env.sh
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export H2H_PLATEAU=1
  if [ -n "$5" ]; then
    export H2H_START_XYZ=$5
  fi
  exec taskset -c 0 "$0" "$1" "$2" "$3" "$4"
' "$BIN" "$1" "$2" "$3" "$4" "${5:-}"
