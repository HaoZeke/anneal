#!/usr/bin/env bash
# Copyright (c) anneal developers
# Same-session plain for_cluster against the plateau population.
# LJ38, LJ55, LJ75, LJ98, LJ104, LJ150 and Au55, twelve seeds, paper budgets.
# Hostname bench, one thread.
set -euo pipefail

LJ=${WATER12_LJ:-/work/water12/bin/lj_cluster_search}
METAL=${WATER12_METAL:-/work/water12/bin/h2h_metal}
OUT=${1:-/work/water12/gate2}
mkdir -p "$OUT"

run_lj() {
  local n=$1 budget=$2 tag=$3 opt=$4
  local dir=$OUT/lj${n}-${tag}
  mkdir -p "$dir"
  sudo -n unshare --uts sh -c '
    hostname bench
    . /work/h2o20-moves/bench-env.sh
    export OMP_NUM_THREADS=1
    export RAYON_NUM_THREADS=1
    export SEED_OFFSET=0
    cd "$1"
    exec taskset -c 0 "$2" "$3" "$4" 12 $5
  ' sh "$dir" "$LJ" "$n" "$budget" "$opt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
  echo "END lj$n $tag $(date -u +%H:%M:%S)"
}

run_au() {
  local tag=$1 plateau=$2
  local dir=$OUT/au55-${tag}
  mkdir -p "$dir"
  : > "$dir/stdout.txt"
  local s
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    sudo -n unshare --uts sh -c '
      hostname bench
      . /work/h2o20-moves/bench-env.sh
      export OMP_NUM_THREADS=1
      export RAYON_NUM_THREADS=1
      if [ "$3" = 1 ]; then export H2H_PLATEAU=1; fi
      exec taskset -c 0 "$1" run au55 "$2"
    ' sh "$METAL" "$s" "$plateau" >> "$dir/stdout.txt" 2>> "$dir/stderr.txt"
  done
  echo "END au55 $tag $(date -u +%H:%M:%S)"
}

run_lj 38 400000 plain ""
run_lj 38 400000 plateau plateau
run_lj 55 1000000 plain ""
run_lj 55 1000000 plateau plateau
run_lj 75 4000000 plain ""
run_lj 75 4000000 plateau plateau
run_lj 98 4000000 plain ""
run_lj 98 4000000 plateau plateau
run_lj 104 4000000 plain ""
run_lj 104 4000000 plateau plateau
run_lj 150 4000000 plain ""
run_lj 150 4000000 plateau plateau
run_au plain 0
run_au plateau 1
echo GATE2_DONE
