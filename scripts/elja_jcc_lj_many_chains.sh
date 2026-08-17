#!/usr/bin/env bash
# Many short LJ catalog replicas in one Slurm allocation.
# Workers launch in waves so one 32-core node does not OOM.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_lj_many_chains.sh requires a Slurm allocation" >&2
  exit 1
fi

N=${1:?LJ site count}
PER_REPLICA_BUDGET=${2:?force evaluations per replica}
ENSEMBLE_INDEX=${3:?ensemble index}
CENSUS_RADIUS=${4:?calibrated census radius}
REPLICAS=${CATALOG_REPLICAS:-300}
WAVE=${CATALOG_WAVE:-24}
MAX_HOPS=${CATALOG_MAX_HOPS:-}
if [[ -n $MAX_HOPS ]]; then
  export CATALOG_MAX_HOPS="$MAX_HOPS"
fi

ROOT=${LJ_ROOT:-$HOME/anneal-dev}
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CALIBRATION=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}/results_jcc/calibration/lj${N}.json
CAMPAIGN=${CATALOG_CAMPAIGN:-lj38-many}
ENSEMBLE="lj${N}-shared-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/jcc}
OUT="$OUT_ROOT/$CAMPAIGN/lj${N}/shared/$ENSEMBLE"
CAPACITY=${CATALOG_CAPACITY:-30}
SLICE=${CATALOG_SLICE:-500}
TRANSPORT_NOISE=${CATALOG_TRANSPORT_NOISE:-0.05}
TRANSPORT_RADIUS=${CATALOG_TRANSPORT_RADIUS:-$(awk -v n="$N" 'BEGIN { printf "%.17g", sqrt(n) }')}
POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=${SEED_OFFSET_BASE:-400000}
REPLICA_LIST=$(seq -s, 0 $((REPLICAS - 1)))
BRAIN_PORT_BASE=${CATALOG_BRAIN_PORT_BASE:-$((27000 + ${SLURM_JOB_ID:-0} % 2000))}
brain_peers() {
  local me=$1
  local lo=$2
  local hi=$3
  local parts=()
  local r
  for r in $(seq "$lo" $((hi - 1))); do
    if (( r != me )); then
      parts+=("${r}=tcp://127.0.0.1:$((BRAIN_PORT_BASE + r))")
    fi
  done
  local IFS=,
  printf '%s' "${parts[*]}"
}

if [[ -e $OUT ]]; then
  echo "ensemble output already exists: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$OUT/traces" "$OUT/workers" "$OUT/state"

export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export GCCLIB=${GCCLIB:-$HOME/mkl-lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"

for executable in "$BIN" "$SERVER"; do
  if [[ ! -x $executable ]]; then
    echo "missing executable: $executable" >&2
    exit 2
  fi
  if ldd "$executable" | grep -q "not found"; then
    echo "unresolved libraries in $executable" >&2
    ldd "$executable" >&2
    exit 2
  fi
done
if [[ ! -f $SOURCE_COMMIT_FILE ]]; then
  echo "missing source commit record: $SOURCE_COMMIT_FILE" >&2
  exit 2
fi
IFS= read -r SOURCE_COMMIT <"$SOURCE_COMMIT_FILE"
if [[ ! $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit record must contain one full Git object ID" >&2
  exit 2
fi
if [[ ! -f $CALIBRATION ]]; then
  echo "missing census-radius calibration: $CALIBRATION" >&2
  exit 2
fi

server_pid=
stop_server() {
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT

"$SERVER" \
  127.0.0.1:0 \
  "$N" \
  "$CAPACITY" \
  "$CENSUS_RADIUS" \
  "$TOTAL_BUDGET" \
  "$CAMPAIGN" \
  "$ENSEMBLE" \
  "$REPLICA_LIST" \
  "$OUT/state" \
  >"$OUT/coordinator.jsonl" 2>"$OUT/coordinator.err" &
server_pid=$!
endpoint=
for _ in $(seq 1 200); do
  endpoint=$(grep -o '"addr":"[^"]*"' "$OUT/coordinator.jsonl" 2>/dev/null | head -1 | cut -d '"' -f4 || true)
  if [[ -n $endpoint ]]; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "catalog coordinator exited during startup" >&2
    cat "$OUT/coordinator.err" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ -z $endpoint ]]; then
  echo "catalog coordinator did not publish its address" >&2
  exit 1
fi

status=0
replica=0
while (( replica < REPLICAS )); do
  pids=()
  wave_start=$replica
  wave_end=$((replica + WAVE))
  if (( wave_end > REPLICAS )); then
    wave_end=$REPLICAS
  fi
  while (( replica < wave_end )); do
    seed=$((SEED_BASE + replica))
    worker=$OUT/workers/replica-${replica}
    mkdir -p "$worker"
    (
      export CATALOG_CAMPAIGN="$CAMPAIGN"
      export CATALOG_ENSEMBLE="$ENSEMBLE"
      export CATALOG_REPLICA="$replica"
      export CATALOG_SLICE="$SLICE"
      export CATALOG_TRANSPORT_NOISE="$TRANSPORT_NOISE"
      export CATALOG_TRANSPORT_RADIUS="$TRANSPORT_RADIUS"
      export CATALOG_POPULATION_INTERVAL="$POPULATION_INTERVAL"
      if [[ -n $MAX_HOPS ]]; then
        export CATALOG_MAX_HOPS="$MAX_HOPS"
      fi
      export CATALOG_TRACE="$OUT/traces/replica-${replica}.jsonl"
      export ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json
      export SEED_OFFSET="$seed"
      export CATALOG_RPC="$endpoint"
      export CATALOG_BRAIN_LISTEN="tcp://127.0.0.1:$((BRAIN_PORT_BASE + replica))"
      export CATALOG_BRAIN_PEERS="$(brain_peers "$replica" "$wave_start" "$wave_end")"
      exec "$BIN" "$N" "$PER_REPLICA_BUDGET" 1 rec
    ) >"$OUT/workers/replica-${replica}.out" 2>"$OUT/workers/replica-${replica}.err" &
    pids+=("$!")
    replica=$((replica + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "a replica in wave ending at $replica failed" >&2
    exit "$status"
  fi
done

if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "catalog coordinator exited before the ensemble terminal boundary" >&2
  cat "$OUT/coordinator.err" >&2
  exit 1
fi
stop_server
server_pid=

solved=0
best_all=
for r in $(seq 0 $((REPLICAS - 1))); do
  output="$OUT/workers/replica-${r}.out"
  test -s "$output"
  test -s "$OUT/traces/replica-${r}.jsonl"
  grep -q "budget ${PER_REPLICA_BUDGET} " "$output"
  line=$(grep -E "seed [0-9]+: best " "$output" | tail -1 || true)
  energy=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="best") print $(i+1)}')
  if [[ -n $energy ]]; then
    if [[ -z $best_all ]] || awk -v a="$energy" -v b="$best_all" 'BEGIN { exit !(a < b) }'; then
      best_all=$energy
    fi
  fi
  if grep -q "1/1 solved" "$output"; then
    solved=$((solved + 1))
  fi
done
printf 'replicas %s solved %s best %s source %s\n' "$REPLICAS" "$solved" "${best_all:-none}" "$SOURCE_COMMIT" >"$OUT/TERMINAL_OK"
cat "$OUT/TERMINAL_OK"
