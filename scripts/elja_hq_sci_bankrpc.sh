#!/usr/bin/env bash
# The LJ production path: one SOAP bank per N, no cut-and-splice.
# Chains share wells over Cap'n; a raised class is left by archive-null
# hop, not a random cluster.
#
#   start   — banks only (7438/7455/7475)
#   submit  — clients only (does not start banks)
#   all     — start then submit
#
# Default is start (safe). Does not cancel running HQ jobs.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
BASE=${LJ_OUT:-$HOME/ljwork/hq-sci-n}
ONE=$ROOT/scripts/elja_hq_one.sh
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SRV=$ROOT/target/release/examples/bank_server
HOST=${BANK_HOST:-$(hostname)}
CAP=${BANK_CAPACITY:-30}

if [[ ! -x $SRV ]]; then
  echo "missing $SRV; build with --features featomic,ira,bank-rpc" >&2
  exit 1
fi
if ldd "$SRV" | grep -q "not found"; then
  echo "bank_server unresolved libs" >&2
  ldd "$SRV"
  exit 1
fi
if ! ldd "$SRV" | grep -q libira; then
  echo "bank_server is not linked to libira; rebuild with --features ira" >&2
  exit 1
fi

export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"

start_bank() {
  local port=$1 out=$2
  mkdir -p "$out"
  if ! ss -ltn | grep -q ":${port} "; then
    nohup "$SRV" "0.0.0.0:${port}" "$CAP" >"$out/bank_server.log" 2>&1 &
    echo $! >"$out/bank_server.pid"
    sleep 1
  fi
  if ! grep -q "bank identity: IRA" "$out/bank_server.log"; then
    echo "bank_server on $port did not print IRA identity" >&2
    cat "$out/bank_server.log" >&2
    exit 1
  fi
}

submit_arm() {
  local n=$1 budget=$2 seeds=$3 port=$4
  local out=${BASE}${n}
  local last=$((seeds - 1))
  mkdir -p "$out"
  : >"$out/hq_submit.log"
  echo "BANK_RPC=${HOST}:${port} n=${n}" | tee -a "$out/hq_submit.log"
  hq submit \
    --name "n${n}-lj${n}-rec" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$out" \
    --stdout "$out/lj${n}_rec_%{TASK_ID}.out" \
    --stderr "$out/lj${n}_rec_%{TASK_ID}.err" \
    -- bash -lc "export BANK_RPC=${HOST}:${port} LJ_BIN=${BIN} BANK_SLICE=${BANK_SLICE:-3000}; exec ${ONE} ${n} ${budget} rec" \
    | tee -a "$out/hq_submit.log"
}

cmd=${1:-start}
case $cmd in
  start)
    start_bank 7438 "${BASE}38"
    start_bank 7455 "${BASE}55"
    start_bank 7475 "${BASE}75"
    echo "banks 7438/7455/7475"
    ;;
  submit)
    submit_arm 38 400000 72 7438
    submit_arm 55 1000000 48 7455
    submit_arm 75 4000000 48 7475
    echo "submitted 38/55/75 SOAP clients"
    hq job list
    ;;
  all)
    start_bank 7438 "${BASE}38"
    start_bank 7455 "${BASE}55"
    start_bank 7475 "${BASE}75"
    submit_arm 38 400000 72 7438
    submit_arm 55 1000000 48 7455
    submit_arm 75 4000000 48 7475
    echo "banks + clients"
    hq job list
    ;;
  *)
    echo "usage: $0 start|submit|all" >&2
    exit 2
    ;;
esac
