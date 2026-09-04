#!/usr/bin/env bash
# Occupancy ensemble that talks: one catalog coordinator, shared bias,
# every replica starts on the sealed LJ75 ico well.
set -euo pipefail
export PATH="/usr/bin:${HOME}/.cargo/bin:${PATH}"
export IRA_LIB_DIR="${IRA_LIB_DIR:-${HOME}/ira/lib}"
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ROOT="${1:-${HOME}/Git/Github/Rust/anneal}"
cd "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${ROOT}/logs/occ-talk-${STAMP}"
mkdir -p "${OUT}/state" "${OUT}/traces"
LOG="${OUT}/wrapper.log"
exec > >(tee -a "${LOG}") 2>&1
echo "commit=$(git rev-parse HEAD)"
echo "out=${OUT}"
cargo build --release --example lj_cluster_search --example catalog_server --features bank-rpc,ira
BIN=./target/release/examples/lj_cluster_search
SERVER=./target/release/examples/catalog_server
N=75
REPLICAS=8
CAPACITY=30
RADIUS=4.80132341502328e-07
BUDGET=4000000
TOTAL=$((BUDGET * REPLICAS))
CAMPAIGN=lj75-ico-talk
ENSEMBLE="talk-${STAMP}"
REPLICA_LIST=$(seq -s, 0 $((REPLICAS - 1)))

"$SERVER" \
  127.0.0.1:0 \
  "$N" \
  "$CAPACITY" \
  "$RADIUS" \
  "$TOTAL" \
  "$CAMPAIGN" \
  "$ENSEMBLE" \
  "$REPLICA_LIST" \
  "$OUT/state" \
  >"$OUT/coordinator.jsonl" 2>"$OUT/coordinator.err" &
server_pid=$!
endpoint=
for _ in $(seq 1 600); do
  endpoint=$(grep -o '"addr":"[^"]*"' "$OUT/coordinator.jsonl" 2>/dev/null \
    | awk -F '"' 'NR == 1 { print $4 }' || true)
  if [[ -n $endpoint ]]; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "coordinator died" >&2
    cat "$OUT/coordinator.err" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ -z $endpoint ]]; then
  echo "coordinator published no address" >&2
  exit 1
fi
echo "catalog_rpc=${endpoint} replicas=${REPLICAS} shared_bias=1 start=ico"

pids=
for replica in $(seq 0 $((REPLICAS - 1))); do
  (
    export CATALOG_CAMPAIGN="$CAMPAIGN"
    export CATALOG_ENSEMBLE="$ENSEMBLE"
    export CATALOG_REPLICA="$replica"
    export CATALOG_RPC="$endpoint"
    export CATALOG_SHARED_BIAS=1
    export CATALOG_SHARING=shared
    export CATALOG_SLICE=500
    export CATALOG_POPULATION_INTERVAL=50000
    export CATALOG_MAX_HOPS=60000
    export CATALOG_START_FILE="${ROOT}/tests/fixtures/lj75_ico.xyz"
    export CATALOG_START_REPLICA=all
    export CATALOG_TRACE="$OUT/traces/replica-${replica}.jsonl"
    export SEED_OFFSET=$((1800000 + replica))
    exec "$BIN" "$N" "$BUDGET" 1 rec
  ) >"$OUT/replica-${replica}.out" 2>"$OUT/replica-${replica}.err" &
  pids="${pids} $!"
done
fail=0
for pid in ${pids}; do
  wait "${pid}" || fail=1
done
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
if /usr/bin/grep -qE '^  Marks -397\.492331' "$OUT"/replica-*.out; then
  echo "MARKS_HIT"
else
  echo "MARKS_MISS"
  fail=1
fi
echo "EXIT:${fail}"
