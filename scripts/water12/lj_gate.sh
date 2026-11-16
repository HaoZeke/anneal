#!/usr/bin/env bash
# Copyright (c) anneal developers
# LJ38, LJ55 and LJ75, twelve seeds, paper budgets, plain for_cluster
# against the same run with the plateau population. One thread, hostname
# bench. The published pin stays on the plain arm.
set -euo pipefail

BIN=${WATER12_LJ:-/work/water12/bin/lj_cluster_search}
OUT=${1:-/work/water12/lj}
mkdir -p "$OUT"

run_one() {
  local n=$1 budget=$2 tag=$3
  shift 3
  local dir=$OUT/lj$n-$tag
  mkdir -p "$dir"
  sudo -n unshare --uts sh -c '
    hostname bench
    . /work/h2o20-moves/bench-env.sh
    export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
    export SEED_OFFSET=0
    cd "$1"
    exec taskset -c 0 "$2" "$3" "$4" 12 $5
  ' sh "$dir" "$BIN" "$n" "$budget" "$*" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
}

run_one 38 400000 plain
run_one 38 400000 plateau plateau
run_one 55 1000000 plain
run_one 55 1000000 plateau plateau
run_one 75 4000000 plain
run_one 75 4000000 plateau plateau
