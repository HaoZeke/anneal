#!/usr/bin/env bash
# CSA bank remesure: recommended hop + Lee Dcut bank + mix + EI.
# Budget is split across the bank, not multiplied.
# Does not touch HQ jobs 2, 87, 89-91.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
OUT=${LJ_OUT:-$HOME/ljwork/hq-sci-bank}
ONE=$ROOT/scripts/elja_hq_one.sh
export BANK_CAPACITY=${BANK_CAPACITY:-30}
export BANK_SLICE=${BANK_SLICE:-3000}
export BANK_MIX=${BANK_MIX:-0.5}
export BANK_DCUT_FLOOR=${BANK_DCUT_FLOOR:-0.4}
export BANK_MIX_IMAGES=${BANK_MIX_IMAGES:-20}
export BANK_RANDOM=${BANK_RANDOM:-10}
export BANK_DEADLOCK_ITERS=${BANK_DEADLOCK_ITERS:-3}
export BANK_DEADLOCK=${BANK_DEADLOCK:-50}
mkdir -p "$OUT"
IDFILE=$OUT/hq_job_ids.txt
: >"$IDFILE"
: >"$OUT/hq_submit.log"

submit_arm() {
  local n=$1 budget=$2 seeds=$3
  local last=$((seeds - 1))
  hq submit \
    --name "bank-lj${n}-rec" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_rec_%{TASK_ID}.out" \
    --stderr "$OUT/lj${n}_rec_%{TASK_ID}.err" \
    -- bash -lc "BANK_CAPACITY=${BANK_CAPACITY} BANK_SLICE=${BANK_SLICE} BANK_MIX=${BANK_MIX} BANK_DCUT_FLOOR=${BANK_DCUT_FLOOR} BANK_MIX_IMAGES=${BANK_MIX_IMAGES} BANK_RANDOM=${BANK_RANDOM} BANK_DEADLOCK_ITERS=${BANK_DEADLOCK_ITERS} BANK_DEADLOCK=${BANK_DEADLOCK} exec ${ONE} ${n} ${budget} rec,bank,acq" | tee -a "$OUT/hq_submit.log"
  awk '/job ID:/{print $NF}' "$OUT/hq_submit.log" | tail -1 >>"$IDFILE"
}

submit_arm 38 400000 72
submit_arm 55 1000000 48
submit_arm 75 4000000 48
echo "submitted. logs in $OUT"
cat "$IDFILE"
hq job list
