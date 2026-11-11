#!/bin/bash
# Copyright (c) anneal developers
# Held-out LJ104 head-to-head. Seeds 5000-5029, one process
# at a time, pinned to cpu 0.
#
# Usage: run_heldout.sh WORK BIN
# WORK receives logs. BIN holds lj_dls, lj_requench,
# h2h_gmin_proof and GMIN. The caller chooses both
# directories. This file names no machine directory.
#
# The standing measurement of this protocol is arm D
# 24/30, median 56.984 s, best -582.0866420676, against
# GMIN 0/30. This script does not rewrite those logs.
set -eu
work=$1
bin=$2
budget=4000000
stop=-582.086641
target=-582.086642
mkdir -p "$work/dls" "$work/gmin" "$work/requench"
run_dls() {
  local seed=$1
  local dir=$work/dls/$seed
  local log=$dir/seed.log
  if [[ -e $log ]]; then
    echo "skip dls $seed"
    return
  fi
  mkdir -p "$dir"
  local prefix=$dir/seed
  {
    echo "n=104"
    echo "seed=$seed"
    echo "tie=lowest-index"
    echo "budget=$budget"
    echo "stop=$stop"
    taskset -c 0 "$bin/lj_dls" 104 "$budget" "$seed" \
      lowest-index "$stop" "$prefix"
    echo "dls_exit=$?"
  } > "$log" 2> "$dir/seed.err"
  if [[ -f $prefix.xyz ]]; then
    "$bin/lj_requench" "$prefix.xyz" \
      > "$work/requench/dls-$seed.txt" \
      2> "$work/requench/dls-$seed.err"
  fi
  echo "done dls $seed"
}
run_gmin() {
  local seed=$1
  local dir=$work/gmin/$seed
  if [[ -e $dir/stdout.txt ]]; then
    echo "skip gmin $seed"
    return
  fi
  mkdir -p "$dir"
  "$bin/h2h_gmin_proof" coords lj104 "$seed" "$dir/coords"
  cat > "$dir/data" << EOF
SLOPPYCONV 5.0D-3
TIGHTCONV 1.0D-3
EDIFF 0.003
MAXIT 600 500
SAVE 2
CENTRE
SORT
TEMPERATURE 1.5
STEPS 90000000 1.0
STEP 0.40 0.51
NEWRESTART 40
AVOID 4.0 20 F

RANSEED ${seed}
EOF
  (
    cd "$dir"
    H2H_BUDGET="$budget" \
    H2H_TARGET="$target" \
    H2H_TOL=1e-6 \
    H2H_SEED="$seed" \
    H2H_SYSTEM=lj104 \
    H2H_ROW="$dir/row.txt" \
      taskset -c 0 "$bin/GMIN" > stdout.txt 2> stderr.txt
  )
  if [[ -f $dir/lowest ]]; then
    {
      echo 104
      echo "gmin lowest"
      awk 'NR>2 && $1=="DU" {printf "X %.10f %.10f %.10f\n", $2, $3, $4}' \
        "$dir/lowest"
    } > "$dir/lowest.xyz"
    "$bin/lj_requench" "$dir/lowest.xyz" \
      > "$work/requench/gmin-$seed.txt" \
      2> "$work/requench/gmin-$seed.err" || true
  fi
  echo "done gmin $seed"
}
for seed in $(seq 5000 5029); do
  run_dls "$seed"
  run_gmin "$seed"
done
echo "HELDOUT_DONE"
