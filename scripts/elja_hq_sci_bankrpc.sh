#!/usr/bin/env bash
# One Cap'n Proto bank, many recommended chains. Paper budgets.
# Does not cancel running HQ jobs.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
OUT=${LJ_OUT:-$HOME/ljwork/hq-sci-bankrpc}
ONE=$ROOT/scripts/elja_hq_one.sh
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SRV=$ROOT/target/release/examples/bank_server
PORT=${BANK_PORT:-7424}
HOST=${BANK_HOST:-$(hostname)}
CAP=${BANK_CAPACITY:-30}
mkdir -p "$OUT"
IDFILE=$OUT/hq_job_ids.txt
: >"$IDFILE"
: >"$OUT/hq_submit.log"

if [[ ! -x $SRV ]]; then
  echo "missing $SRV; build with --features featomic,bank-rpc" >&2
  exit 1
fi

if ! ss -ltn | grep -q ":${PORT} "; then
  nohup "$SRV" "0.0.0.0:${PORT}" "$CAP" >"$OUT/bank_server.log" 2>&1 &
  echo $! >"$OUT/bank_server.pid"
  sleep 1
fi
export BANK_RPC="${HOST}:${PORT}"
echo "BANK_RPC=$BANK_RPC" | tee -a "$OUT/hq_submit.log"

submit_arm() {
  local n=$1 budget=$2 seeds=$3
  local last=$((seeds - 1))
  hq submit \
    --name "rpc-lj${n}-rec" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$OUT" \
    --stdout "$OUT/lj${n}_rec_%{TASK_ID}.out" \
    --stderr "$OUT/lj${n}_rec_%{TASK_ID}.err" \
    -- bash -lc "export BANK_RPC=${BANK_RPC} LJ_BIN=${BIN} BANK_SLICE=${BANK_SLICE:-3000}; exec ${ONE} ${n} ${budget} rec" \
    | tee -a "$OUT/hq_submit.log"
  awk '/job ID:/{print $NF}' "$OUT/hq_submit.log" | tail -1 >>"$IDFILE"
}

submit_arm 38 400000 72
submit_arm 55 1000000 48
submit_arm 75 4000000 48
echo "submitted. logs in $OUT"
cat "$IDFILE"
hq job list
