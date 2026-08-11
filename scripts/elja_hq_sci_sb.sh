#!/usr/bin/env bash
# Production: SOAP packing superbasin (CV-GO) + packing-mean hop.
# Paper budgets. Does not cancel running HQ jobs. Uses LJ_BIN if set.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
OUT=${LJ_OUT:-$HOME/ljwork/hq-sci-sb}
ONE=$ROOT/scripts/elja_hq_one.sh
mkdir -p "$OUT"
IDFILE=$OUT/hq_job_ids.txt
: >"$IDFILE"
: >"$OUT/hq_submit.log"

submit_arm() {
  local n=$1 budget=$2 seeds=$3
  local last=$((seeds - 1))
  hq submit \
    --name "sb-lj${n}-rec" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_rec_%{TASK_ID}.out" \
    --stderr "$OUT/lj${n}_rec_%{TASK_ID}.err" \
    -- bash -lc "export LJ_BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search-sb}; exec \"$ONE\" $n $budget rec" | tee -a "$OUT/hq_submit.log"
  awk '/job ID:/{print $NF}' "$OUT/hq_submit.log" | tail -1 >>"$IDFILE"
}

submit_arm 38 400000 72
submit_arm 55 1000000 48
submit_arm 75 4000000 48
echo "submitted. logs in $OUT"
cat "$IDFILE"
hq job list
