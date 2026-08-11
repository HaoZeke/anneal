#!/usr/bin/env bash
# Scientific recommended remesure of the featomic patch leftover.
# Paper budgets. Rec only: for_cluster is unchanged (Atomic, no SOAP).
#   LJ38: 4e5 x 72
#   LJ55: 1e6 x 48
#   LJ75: 4e6 x 48
# Does not touch HQ job 2.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
OUT=${LJ_OUT:-$HOME/ljwork/hq-sci-feat}
ONE=$ROOT/scripts/elja_hq_one.sh
mkdir -p "$OUT"

IDFILE=$OUT/hq_job_ids.txt
: >"$IDFILE"
: >"$OUT/hq_submit.log"

submit_arm() {
  local n=$1 budget=$2 seeds=$3 arm=$4
  local last=$((seeds - 1))
  hq submit \
    --name "feat-lj${n}-${arm}" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_${arm}_%{TASK_ID}.out" \
    --stderr "$OUT/lj${n}_${arm}_%{TASK_ID}.err" \
    -- "$ONE" "$n" "$budget" "$arm" | tee -a "$OUT/hq_submit.log"
  awk '/job ID:/{print $NF}' "$OUT/hq_submit.log" | tail -1 >>"$IDFILE"
}

submit_arm 38 400000 72 rec
submit_arm 55 1000000 48 rec
submit_arm 75 4000000 48 rec

echo "submitted. logs in $OUT"
cat "$IDFILE"
hq job list
