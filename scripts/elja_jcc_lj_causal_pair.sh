#!/usr/bin/env bash
# Matched shared-catalog and private-catalog LJ ensembles in one allocation.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_lj_causal_pair.sh requires a Slurm allocation" >&2
  exit 1
fi

N=${1:?LJ site count}
PER_REPLICA_BUDGET=${2:?force evaluations per replica}
ENSEMBLE_INDEX=${3:?ensemble index or slurm-array}
CENSUS_RADIUS=${4:?calibrated census radius}
ARM=${5:?shared or private}
if [[ $ENSEMBLE_INDEX == slurm-array ]]; then
  ENSEMBLE_INDEX=${SLURM_ARRAY_TASK_ID:?Slurm array index}
fi
case $ARM in
  shared|private) ;;
  *)
    echo "arm must be shared or private" >&2
    exit 2
    ;;
esac

ROOT=${LJ_ROOT:-$HOME/anneal-build}
CAMPAIGN_ENV_BIN=${CAMPAIGN_ENV_BIN:-$ROOT/target/release/examples/campaign_env}
if [[ -n ${CATALOG_CONFIG:-} ]]; then
  if [[ ! -x $CAMPAIGN_ENV_BIN ]]; then
    echo "missing campaign resolver: $CAMPAIGN_ENV_BIN" >&2
    exit 2
  fi
  campaign_env_text=$("$CAMPAIGN_ENV_BIN" "$CATALOG_CONFIG")
  while IFS=$'\t' read -r name value; do
    [[ -n $name ]] || continue
    if [[ -v $name && ${!name} != "$value" ]]; then
      echo "$name=${!name} conflicts with $CATALOG_CONFIG value $value" >&2
      exit 2
    fi
    printf -v "$name" '%s' "$value"
    export "${name?}"
  done <<<"$campaign_env_text"
fi

REPLICAS=${CATALOG_REPLICAS:-4}
WAVE=${CATALOG_WAVE:-4}
MAX_HOPS=${CATALOG_MAX_HOPS:-}
for integer in "$N" "$PER_REPLICA_BUDGET" "$ENSEMBLE_INDEX" "$REPLICAS" "$WAVE"; do
  if [[ ! $integer =~ ^[0-9]+$ ]]; then
    echo "site count, budgets, indices, replicas, and wave must be unsigned integers" >&2
    exit 2
  fi
done
if ((N == 0 || PER_REPLICA_BUDGET == 0 || REPLICAS == 0 || WAVE == 0 || WAVE > REPLICAS)); then
  echo "site count, budget, replicas, and wave must be positive; wave cannot exceed replicas" >&2
  exit 2
fi
if [[ -n $MAX_HOPS ]]; then
  if [[ ! $MAX_HOPS =~ ^[1-9][0-9]*$ ]]; then
    echo "CATALOG_MAX_HOPS must be a positive integer" >&2
    exit 2
  fi
  export CATALOG_MAX_HOPS="$MAX_HOPS"
fi

BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CALIBRATION=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}/results_jcc/calibration/lj${N}.json
CAMPAIGN=${CATALOG_CAMPAIGN:-lj-catalog-causal}
ENSEMBLE="lj${N}-${ARM}-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/jcc}
OUT="$OUT_ROOT/$CAMPAIGN/lj${N}/$ARM/$ENSEMBLE"
CAPACITY=${CATALOG_CAPACITY:-30}
SLICE=${CATALOG_SLICE:-500}
TRANSPORT_NOISE=${CATALOG_TRANSPORT_NOISE:-0.05}
TRANSPORT_RADIUS=${CATALOG_TRANSPORT_RADIUS:-$(awk -v n="$N" 'BEGIN { printf "%.17g", sqrt(n) }')}
POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=$((${SEED_OFFSET_BASE:-400000} + ENSEMBLE_INDEX * REPLICAS))
REPLICA_LIST=$(seq -s, 0 $((REPLICAS - 1)))
BRAIN_PORT_BASE=${CATALOG_BRAIN_PORT_BASE:-$((27000 + SLURM_JOB_ID % 2000))}

brain_peers() {
  local me=$1
  local lo=$2
  local hi=$3
  local parts=()
  local replica
  for replica in $(seq "$lo" $((hi - 1))); do
    if ((replica != me)); then
      parts+=("${replica}=tcp://127.0.0.1:$((BRAIN_PORT_BASE + replica))")
    fi
  done
  local IFS=,
  printf '%s' "${parts[*]}"
}

if [[ -e $OUT ]]; then
  echo "ensemble output already exists: $OUT" >&2
  exit 1
fi
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

mkdir -p "$OUT/traces" "$OUT/workers" "$OUT/state"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export GCCLIB=${GCCLIB:-$HOME/mkl-lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"

server_pids=()
server_prefixes=()
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
  local total_work=$2
  local server_ensemble=$3
  local replicas=$4
  local state=$5
  local pid endpoint=
  mkdir -p "$state"
  "$SERVER" \
    127.0.0.1:0 \
    "$N" \
    "$CAPACITY" \
    "$CENSUS_RADIUS" \
    "$total_work" \
    "$CAMPAIGN" \
    "$server_ensemble" \
    "$replicas" \
    "$state" \
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

require_live_servers() {
  local index
  for index in "${!server_pids[@]}"; do
    if ! kill -0 "${server_pids[$index]}" 2>/dev/null; then
      wait "${server_pids[$index]}" || true
      echo "catalog coordinator exited before its replicas: ${server_prefixes[$index]}" >&2
      cat "${server_prefixes[$index]}.err" >&2
      exit 1
    fi
  done
}

shared_endpoint=
if [[ $ARM == shared ]]; then
  start_catalog \
    "$OUT/coordinator-shared" \
    "$TOTAL_BUDGET" \
    "$ENSEMBLE" \
    "$REPLICA_LIST" \
    "$OUT/state/shared"
  shared_endpoint=$last_started_endpoint
  catalog_topology=shared
  brain_topology=wave_peers
else
  catalog_topology=private_per_replica
  brain_topology=singleton
fi

status=0
replica=0
while ((replica < REPLICAS)); do
  pids=()
  private_endpoints=()
  private_ensembles=()
  wave_start=$replica
  wave_end=$((replica + WAVE))
  if ((wave_end > REPLICAS)); then
    wave_end=$REPLICAS
  fi

  if [[ $ARM == private ]]; then
    for private_replica in $(seq "$wave_start" $((wave_end - 1))); do
      private_ensemble="${ENSEMBLE}-replica-$(printf '%04d' "$private_replica")"
      start_catalog \
        "$OUT/coordinator-replica-${private_replica}" \
        "$PER_REPLICA_BUDGET" \
        "$private_ensemble" \
        "$private_replica" \
        "$OUT/state/replica-${private_replica}"
      private_endpoints[$private_replica]=$last_started_endpoint
      private_ensembles[$private_replica]=$private_ensemble
    done
  fi

  while ((replica < wave_end)); do
    seed=$((SEED_BASE + replica))
    worker=$OUT/workers/replica-${replica}
    mkdir -p "$worker"
    if [[ $ARM == shared ]]; then
      worker_endpoint=$shared_endpoint
      worker_ensemble=$ENSEMBLE
      worker_brain_peers=$(brain_peers "$replica" "$wave_start" "$wave_end")
    else
      worker_endpoint=${private_endpoints[$replica]}
      worker_ensemble=${private_ensembles[$replica]}
      worker_brain_peers=
    fi
    (
      export CATALOG_CAMPAIGN="$CAMPAIGN"
      export CATALOG_ENSEMBLE="$worker_ensemble"
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
      export CATALOG_RPC="$worker_endpoint"
      export CATALOG_BRAIN_LISTEN="tcp://127.0.0.1:$((BRAIN_PORT_BASE + replica))"
      CATALOG_BRAIN_PEERS=$worker_brain_peers
      export CATALOG_BRAIN_PEERS
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
  if ((status != 0)); then
    echo "a replica in wave ending at $replica failed" >&2
    exit "$status"
  fi
  require_live_servers
  if [[ $ARM == private ]]; then
    stop_servers
  fi
done

if [[ $ARM == shared ]]; then
  require_live_servers
  stop_servers
fi

solved=0
best_all=
for replica in $(seq 0 $((REPLICAS - 1))); do
  output="$OUT/workers/replica-${replica}.out"
  test -s "$output"
  test -s "$OUT/traces/replica-${replica}.jsonl"
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

binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
server_sha256=$(sha256sum "$SERVER" | awk '{print $1}')
runner_sha256=$(sha256sum "$0" | awk '{print $1}')
calibration_sha256=$(sha256sum "$CALIBRATION" | awk '{print $1}')
campaign_config_sha256=none
if [[ -n ${CATALOG_CONFIG:-} ]]; then
  campaign_config_sha256=$(sha256sum "$CATALOG_CONFIG" | awk '{print $1}')
fi
{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'system=lj%s\n' "$N"
  printf 'arm=%s\n' "$ARM"
  printf 'ensemble=%s\n' "$ENSEMBLE"
  printf 'catalog_topology=%s\n' "$catalog_topology"
  printf 'brain_topology=%s\n' "$brain_topology"
  printf 'replicas=%s\n' "$REPLICAS"
  printf 'wave=%s\n' "$WAVE"
  printf 'per_replica_budget=%s\n' "$PER_REPLICA_BUDGET"
  printf 'aggregate_budget=%s\n' "$TOTAL_BUDGET"
  printf 'seed_base=%s\n' "$SEED_BASE"
  printf 'slice=%s\n' "$SLICE"
  printf 'max_hops=%s\n' "${MAX_HOPS:-none}"
  printf 'population_interval=%s\n' "$POPULATION_INTERVAL"
  printf 'census_radius=%s\n' "$CENSUS_RADIUS"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'binary_sha256=%s\n' "$binary_sha256"
  printf 'server_sha256=%s\n' "$server_sha256"
  printf 'runner_sha256=%s\n' "$runner_sha256"
  printf 'calibration_sha256=%s\n' "$calibration_sha256"
  printf 'campaign_config_sha256=%s\n' "$campaign_config_sha256"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
} >"$OUT/run.manifest"

printf 'replicas %s solved %s best %s source %s arm %s\n' \
  "$REPLICAS" "$solved" "${best_all:-none}" "$SOURCE_COMMIT" "$ARM" \
  >"$OUT/TERMINAL_OK"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >SHA256SUMS)
cat "$OUT/TERMINAL_OK"
