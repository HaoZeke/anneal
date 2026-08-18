#!/usr/bin/env bash
# Three occupancy ensembles on the live HyperQueue server.
# One task = one system, every replica live. Paper budgets.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
OUT=${LJ_OUT:-$HOME/ljwork/hq-occ}
ONE=$ROOT/scripts/elja_hq_occ_one.sh
mkdir -p "$OUT"
IDFILE=$OUT/hq_job_ids.txt
: >"$IDFILE"
: >"$OUT/hq_submit.log"

submit_one() {
  local n=$1 budget=$2 ensemble=$3 limit=$4
  hq submit \
    --name "lj${n}-occ" \
    --cpus 48 \
    --time-limit="$limit" \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_occ.out" \
    --stderr "$OUT/lj${n}_occ.err" \
    -- "$ONE" "$n" "$budget" "$ensemble" | tee -a "$OUT/hq_submit.log"
  awk '/job ID:/{print $NF}' "$OUT/hq_submit.log" | tail -1 >>"$IDFILE"
}

submit_one 38 400000 0 2d
submit_one 75 4000000 0 4d
submit_one 98 4000000 0 4d
echo "submitted. logs in $OUT"
hq job list
