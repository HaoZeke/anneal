#!/usr/bin/env bash
# Many LJ catalog replicas in one Slurm allocation. Shared runs keep the
# synchronous population live; private runs give every replica its own catalog.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_lj_many_chains.sh requires a Slurm allocation" >&2
  exit 1
fi

N=${1:?LJ site count}
PER_REPLICA_BUDGET=${2:?force evaluations per replica}
ENSEMBLE_INDEX=${3:?ensemble index}
CENSUS_RADIUS=${4:?calibrated census radius}
ARM=${5:-shared}
case $ARM in
  shared|private) ;;
  *)
    echo "arm must be shared or private" >&2
    exit 2
    ;;
esac
REPLICAS=${CATALOG_REPLICAS:-300}
WAVE=${CATALOG_WAVE:-24}
MAX_HOPS=${CATALOG_MAX_HOPS:-}
if [[ $ARM == shared && $WAVE -ne $REPLICAS ]]; then
  echo "shared population requires CATALOG_WAVE=CATALOG_REPLICAS" >&2
  exit 2
fi
if [[ -n $MAX_HOPS ]]; then
  export CATALOG_MAX_HOPS="$MAX_HOPS"
fi

ROOT=${LJ_ROOT:-$HOME/anneal-dev}
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CALIBRATION=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}/results_jcc/calibration/lj${N}.json
CAMPAIGN=${CATALOG_CAMPAIGN:-lj38-many}
ENSEMBLE="lj${N}-${ARM}-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/jcc}
OUT="$OUT_ROOT/$CAMPAIGN/lj${N}/$ARM/$ENSEMBLE"
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

server_pids=()
server_prefixes=()
private_endpoints=()
shared_endpoint=
last_started_endpoint=
stop_servers() {
  local pid
  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${server_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  server_pids=()
  server_prefixes=()
}
trap stop_servers EXIT

start_catalog() {
  local prefix=$1
  local state_dir=$2
  local replica_list=$3
  local server_budget=$4
  local pid endpoint=
  mkdir -p "$state_dir"
  "$SERVER" \
    127.0.0.1:0 \
    "$N" \
    "$CAPACITY" \
    "$CENSUS_RADIUS" \
    "$server_budget" \
    "$CAMPAIGN" \
    "$ENSEMBLE" \
    "$replica_list" \
    "$state_dir" \
    >"${prefix}.jsonl" 2>"${prefix}.err" &
  pid=$!
  server_pids+=("$pid")
  server_prefixes+=("$prefix")
  for _ in $(seq 1 200); do
    endpoint=$(grep -o '"addr":"[^"]*"' "${prefix}.jsonl" 2>/dev/null \
      | head -1 | cut -d '"' -f4 || true)
    if [[ -n $endpoint ]]; then
      last_started_endpoint=$endpoint
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "catalog coordinator exited during startup: $prefix" >&2
      cat "${prefix}.err" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "catalog coordinator did not publish its address: $prefix" >&2
  exit 1
}

if [[ $ARM == shared ]]; then
  start_catalog "$OUT/coordinator" "$OUT/state" "$REPLICA_LIST" "$TOTAL_BUDGET"
  shared_endpoint=$last_started_endpoint
  catalog_topology=shared
else
  for replica in $(seq 0 $((REPLICAS - 1))); do
    start_catalog \
      "$OUT/coordinator-replica-$replica" \
      "$OUT/state/replica-$replica" \
      "$replica" \
      "$PER_REPLICA_BUDGET"
    private_endpoints[replica]=$last_started_endpoint
  done
  catalog_topology=private_per_replica
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
      export RAYON_NUM_THREADS=1
      export OMP_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      if [[ $ARM == shared ]]; then
        export CATALOG_RPC="$shared_endpoint"
        export CATALOG_BRAIN_LISTEN="tcp://127.0.0.1:$((BRAIN_PORT_BASE + replica))"
        CATALOG_BRAIN_PEERS=$(brain_peers "$replica" "$wave_start" "$wave_end")
        export CATALOG_BRAIN_PEERS
      else
        export CATALOG_RPC=${private_endpoints[$replica]}
        unset CATALOG_BRAIN_LISTEN CATALOG_BRAIN_PEERS
      fi
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

for index in "${!server_pids[@]}"; do
  if ! kill -0 "${server_pids[$index]}" 2>/dev/null; then
    wait "${server_pids[$index]}" || true
    echo "catalog coordinator exited before the ensemble terminal boundary: ${server_prefixes[$index]}" >&2
    cat "${server_prefixes[$index]}.err" >&2
    exit 1
  fi
done
stop_servers

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
printf 'replicas %s solved %s best %s arm %s topology %s source %s\n' \
  "$REPLICAS" "$solved" "${best_all:-none}" "$ARM" "$catalog_topology" \
  "$SOURCE_COMMIT" >"$OUT/TERMINAL_OK"
cat "$OUT/TERMINAL_OK"
