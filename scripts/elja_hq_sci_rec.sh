#!/usr/bin/env bash
# Submit scientific recommended vs for_cluster on the live Elja HyperQueue
# server at paper budgets. One HQ task = one seed.
#
#   LJ38: 4e5 x 72
#   LJ55: 1e6 x 48
#   LJ75: 4e6 x 48
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
OUT=${LJ_OUT:-$HOME/ljwork/hq-sci-rec}
ONE=$ROOT/scripts/elja_hq_one.sh
mkdir -p "$OUT"

submit_arm() {
  local n=$1 budget=$2 seeds=$3 arm=$4
  local last=$((seeds - 1))
  hq submit \
    --name "lj${n}-${arm}" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_${arm}_%{TASK_ID}.out" \
    --stderr "$OUT/lj${n}_${arm}_%{TASK_ID}.err" \
    -- "$ONE" "$n" "$budget" "$arm"
}

submit_arm 38 400000 72 rec
submit_arm 38 400000 72 base
submit_arm 55 1000000 48 rec
submit_arm 55 1000000 48 base
submit_arm 75 4000000 48 rec
submit_arm 75 4000000 48 base

echo "submitted. logs in $OUT"
hq job list
