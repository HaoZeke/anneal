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
ARM=${5:?shared or control}
if [[ $ENSEMBLE_INDEX == slurm-array ]]; then
  ENSEMBLE_INDEX=${SLURM_ARRAY_TASK_ID:?Slurm array index}
fi
case $ARM in
  shared)
    SHARING_MODE=shared
    TRACE_SHARING=true
    ;;
  control)
    SHARING_MODE=private
    TRACE_SHARING=false
    ;;
  *)
    echo "arm must be shared or control" >&2
    exit 2
    ;;
esac

require_protocol_value() {
  local name=$1
  local expected=$2
  if [[ -v $name && ${!name} != "$expected" ]]; then
    echo "$name=${!name} conflicts with the JCC LJ protocol value $expected" >&2
    exit 2
  fi
  printf -v "$name" '%s' "$expected"
  export "$name"
}

reject_protocol_variable() {
  local name=$1
  if [[ -v $name ]]; then
    echo "$name is not part of the JCC LJ causal protocol" >&2
    exit 2
  fi
}

require_protocol_value CATALOG_SHARED_SCREEN 1
require_protocol_value CATALOG_SHARED_BIAS 0
require_protocol_value CATALOG_ENTROPIC_BIAS 0
require_protocol_value CATALOG_HISTO_SCREEN 0
require_protocol_value CATALOG_SEAM_LADDER 1
require_protocol_value CATALOG_FRONTIER_EXCHANGE 1
require_protocol_value CATALOG_COOP_WELLS 1
require_protocol_value CATALOG_BRIDGE 0
require_protocol_value CATALOG_DIFFICULTY 0
require_protocol_value CATALOG_PACKING_PAVE 0
require_protocol_value CATALOG_SEAM_PATIENCE 800
require_protocol_value CATALOG_SEAM_TRACE 0
require_protocol_value CATALOG_HISTO_RADIUS 1.4
require_protocol_value CATALOG_PROBE_INTERVAL 8
require_protocol_value CATALOG_PROBE_SCALE 0.2
require_protocol_value CATALOG_BRIDGE_INTERVAL 64
require_protocol_value QUENCH_NOISE 0

reject_protocol_variable CATALOG_CONFIG
reject_protocol_variable CATALOG_TEMP_LADDER
reject_protocol_variable CATALOG_MD_ENGINE
reject_protocol_variable CATALOG_MD_BIN
reject_protocol_variable CATALOG_MD_INTERVAL
reject_protocol_variable CATALOG_MD_STEPS
reject_protocol_variable CATALOG_MD_TEMP
reject_protocol_variable CATALOG_START_FILE
reject_protocol_variable CATALOG_START_REPLICA
reject_protocol_variable CATALOG_BRAIN_PUBLISH

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

ROOT=${LJ_ROOT:-$HOME/anneal-build}
SOURCE_ROOT=${JCC_SOURCE_ROOT:-$ROOT}
REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CALIBRATION=$REPRO_ROOT/results_jcc/calibration/lj${N}.json
QUALIFIER=${JCC_QUALIFIER:-$REPRO_ROOT/workflow/jcc/validate_hard_lj_qualification.py}
QUALIFIER_PYTHON=${JCC_QUALIFIER_PYTHON:-$REPRO_ROOT/.pixi/envs/default/bin/python}
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
HEAD=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
if [[ $SOURCE_COMMIT != "$HEAD" ]]; then
  echo "SOURCE_COMMIT=$SOURCE_COMMIT does not match HEAD=$HEAD" >&2
  exit 2
fi
if ! (cd "$SOURCE_ROOT" && git diff --quiet HEAD --); then
  echo "tracked source differs from HEAD=$HEAD" >&2
  (cd "$SOURCE_ROOT" && git status --short >&2)
  exit 2
fi
if [[ ! -s $ROOT/BUILD_SHA256SUMS ]]; then
  echo "missing LJ build artifact seal: $ROOT/BUILD_SHA256SUMS" >&2
  exit 2
fi
(cd "$ROOT" && sha256sum -c BUILD_SHA256SUMS)
if [[ ! -f $CALIBRATION ]]; then
  echo "missing census-radius calibration: $CALIBRATION" >&2
  exit 2
fi
if [[ ! -f $QUALIFIER ]]; then
  echo "missing hard-LJ qualifier: $QUALIFIER" >&2
  exit 2
fi
if [[ ! -x $QUALIFIER_PYTHON ]]; then
  echo "missing qualification Python: $QUALIFIER_PYTHON" >&2
  exit 2
fi

mkdir -p "$OUT/traces" "$OUT/workers" "$OUT/state"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export GCCLIB=${GCCLIB:-$HOME/mkl-lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

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
  wave_start=$replica
  wave_end=$((replica + WAVE))
  if ((wave_end > REPLICAS)); then
    wave_end=$REPLICAS
  fi

  if [[ $ARM == control ]]; then
    for private_replica in $(seq "$wave_start" $((wave_end - 1))); do
      start_catalog \
        "$OUT/coordinator-replica-${private_replica}" \
        "$PER_REPLICA_BUDGET" \
        "$ENSEMBLE" \
        "$private_replica" \
        "$OUT/state/replica-${private_replica}"
      private_endpoints[private_replica]=$last_started_endpoint
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
      worker_ensemble=$ENSEMBLE
      worker_brain_peers=
    fi
    (
      export CATALOG_CAMPAIGN="$CAMPAIGN"
      export CATALOG_ENSEMBLE="$worker_ensemble"
      export CATALOG_REPLICA="$replica"
      export CATALOG_SHARING="$SHARING_MODE"
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
  if [[ $ARM == control ]]; then
    stop_servers
  fi
done

if [[ $ARM == shared ]]; then
  require_live_servers
  stop_servers
fi

solved=0
best_all=
qualification_traces=()
for replica in $(seq 0 $((REPLICAS - 1))); do
  output="$OUT/workers/replica-${replica}.out"
  errors="$OUT/workers/replica-${replica}.err"
  trace="$OUT/traces/replica-${replica}.jsonl"
  resolved_config="$OUT/workers/replica-${replica}/resolved-config.json"
  test -s "$output"
  test -s "$trace"
  test -s "$resolved_config"
  if [[ -s $errors ]]; then
    echo "replica $replica wrote error output" >&2
    cat "$errors" >&2
    exit 1
  fi
  qualification_traces+=("$trace")
  if ! head -n 1 "$trace" \
    | grep -F -q "\"ensemble\":\"$ENSEMBLE\",\"sharing\":$TRACE_SHARING"; then
    echo "trace manifest does not match $ENSEMBLE sharing=$TRACE_SHARING: $trace" >&2
    exit 1
  fi
  if ! grep -F -q '"kind":"slice"' "$trace"; then
    echo "trace has no analyzer slice: $trace" >&2
    exit 1
  fi
  if grep -F -q '"kind":"rpc_fallback"' "$trace"; then
    echo "trace records catalog RPC fallback: $trace" >&2
    exit 1
  fi
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

"$QUALIFIER_PYTHON" "$QUALIFIER" \
  --output "$OUT/qualification.json" \
  "${qualification_traces[@]}"

for replica in $(seq 1 $((REPLICAS - 1))); do
  cmp -s "$OUT/workers/replica-0/resolved-config.json" \
    "$OUT/workers/replica-${replica}/resolved-config.json"
done

cp "$OUT/workers/replica-0/resolved-config.json" "$OUT/resolved-config.json"
resolved_config_sha256=$(sha256sum "$OUT/resolved-config.json" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
server_sha256=$(sha256sum "$SERVER" | awk '{print $1}')
runner_sha256=$(sha256sum "$0" | awk '{print $1}')
qualifier_sha256=$(sha256sum "$QUALIFIER" | awk '{print $1}')
calibration_sha256=$(sha256sum "$CALIBRATION" | awk '{print $1}')
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
  printf 'catalog_capacity=%s\n' "$CAPACITY"
  printf 'slice=%s\n' "$SLICE"
  printf 'coordination_protocol=%s\n' 'jcc-lj-causal-v1'
  printf 'shared_screen=%s\n' "$CATALOG_SHARED_SCREEN"
  printf 'shared_bias=%s\n' "$CATALOG_SHARED_BIAS"
  printf 'entropic_bias=%s\n' "$CATALOG_ENTROPIC_BIAS"
  printf 'histo_screen=%s\n' "$CATALOG_HISTO_SCREEN"
  printf 'seam_ladder=%s\n' "$CATALOG_SEAM_LADDER"
  printf 'frontier_exchange=%s\n' "$CATALOG_FRONTIER_EXCHANGE"
  printf 'coop_wells=%s\n' "$CATALOG_COOP_WELLS"
  printf 'bridge=%s\n' "$CATALOG_BRIDGE"
  printf 'difficulty=%s\n' "$CATALOG_DIFFICULTY"
  printf 'packing_pave=%s\n' "$CATALOG_PACKING_PAVE"
  printf 'seam_patience=%s\n' "$CATALOG_SEAM_PATIENCE"
  printf 'seam_trace=%s\n' "$CATALOG_SEAM_TRACE"
  printf 'histo_radius=%s\n' "$CATALOG_HISTO_RADIUS"
  printf 'probe_interval=%s\n' "$CATALOG_PROBE_INTERVAL"
  printf 'probe_scale=%s\n' "$CATALOG_PROBE_SCALE"
  printf 'bridge_interval=%s\n' "$CATALOG_BRIDGE_INTERVAL"
  printf 'quench_noise=%s\n' "$QUENCH_NOISE"
  printf 'temperature_ladder=%s\n' 'none'
  printf 'md_engine=%s\n' 'none'
  printf 'transport_noise=%s\n' "$TRANSPORT_NOISE"
  printf 'transport_radius=%s\n' "$TRANSPORT_RADIUS"
  printf 'max_hops=%s\n' "${MAX_HOPS:-none}"
  printf 'population_interval=%s\n' "$POPULATION_INTERVAL"
  printf 'census_radius=%s\n' "$CENSUS_RADIUS"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'resolved_config_sha256=%s\n' "$resolved_config_sha256"
  printf 'binary_sha256=%s\n' "$binary_sha256"
  printf 'server_sha256=%s\n' "$server_sha256"
  printf 'runner_sha256=%s\n' "$runner_sha256"
  printf 'qualifier_sha256=%s\n' "$qualifier_sha256"
  printf 'calibration_sha256=%s\n' "$calibration_sha256"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
} >"$OUT/run.manifest"

printf 'replicas %s solved %s best %s source %s arm %s\n' \
  "$REPLICAS" "$solved" "${best_all:-none}" "$SOURCE_COMMIT" "$ARM" \
  >"$OUT/TERMINAL_OK"
mapfile -d '' artifact_files < <(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z)
(cd "$OUT" && sha256sum "${artifact_files[@]}" >SHA256SUMS)
cat "$OUT/TERMINAL_OK"
