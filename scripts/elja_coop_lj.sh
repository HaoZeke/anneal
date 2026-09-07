#!/usr/bin/env bash
# One cooperative LJ ensemble on one Elja node: a catalog coordinator and
# CATALOG_REPLICAS worker processes, each with its own charged budget.
#
# Usage (under Slurm): elja_coop_lj.sh N PER_REPLICA_BUDGET INDEX CENSUS_RADIUS
#
# Environment: LJ_ROOT (staging tree), LJ_BIN, CATALOG_SERVER_BIN,
# CATALOG_CAMPAIGN (output subdirectory), CATALOG_REPLICAS (default 48),
# CATALOG_WORKER_OPTS (mechanism list, default rec), CATALOG_SLICE,
# CATALOG_POPULATION_INTERVAL, CATALOG_MAX_HOPS, plus every CENSUS_BUS_* and
# GOSSIP* variable the workers read. Output under $LJ_OUT/$CAMPAIGN/ljN/shared/.
#
# Derived from elja_jcc_lj_ensemble.sh without the qualification stage; the
# coordinator's replica list and the worker count follow CATALOG_REPLICAS.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_coop_lj.sh requires a Slurm allocation" >&2
  exit 1
fi
N=${1:?LJ site count}
PER_REPLICA_BUDGET=${2:?per-replica charged budget}
ENSEMBLE_INDEX=${3:?ensemble index}
CENSUS_RADIUS=${4:?census radius}
ROOT=${LJ_ROOT:-$HOME/anneal-psym}
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CAMPAIGN=${CATALOG_CAMPAIGN:-lj${N}-coop}
ENSEMBLE="lj${N}-shared-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/coop}
OUT="$OUT_ROOT/$CAMPAIGN/lj${N}/shared/$ENSEMBLE"
CAPACITY=${CATALOG_CAPACITY:-30}
SLICE=${CATALOG_SLICE:-500}
TRANSPORT_NOISE=${CATALOG_TRANSPORT_NOISE:-0.05}
TRANSPORT_RADIUS=${CATALOG_TRANSPORT_RADIUS:-$(awk -v n="$N" 'BEGIN { printf "%.17g", sqrt(n) }')}
POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
MINIMUM_POPULATION_INTERVAL=$((2 * SLICE + 2))
if (( POPULATION_INTERVAL < MINIMUM_POPULATION_INTERVAL )); then
  POPULATION_INTERVAL=$MINIMUM_POPULATION_INTERVAL
fi
REPLICAS=${CATALOG_REPLICAS:-48}
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=$(( ${SEED_OFFSET_BASE:-0} + ENSEMBLE_INDEX * REPLICAS ))
WORKER_OPTS=${CATALOG_WORKER_OPTS:-rec}
if [[ -e $OUT ]]; then
  echo "ensemble output already exists: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$OUT/traces" "$OUT/workers" "$OUT/state"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
for executable in "$BIN" "$SERVER"; do
  if [[ ! -x $executable ]]; then
    echo "missing executable: $executable" >&2
    exit 1
  fi
done
replica_list=$(seq -s, 0 $((REPLICAS - 1)))
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
  "$replica_list" \
  "$OUT/state" \
  >"$OUT/coordinator.jsonl" 2>"$OUT/coordinator.err" &
server_pid=$!
endpoint=
for _ in $(seq 1 100); do
  endpoint=$(grep -o '"addr":"[^"]*"' "$OUT/coordinator.jsonl" 2>/dev/null \
    | awk -F '"' 'NR == 1 { print $4 }' || true)
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
{
  printf 'campaign=%s\nsystem=lj%s\nensemble=%s\nreplicas=%s\nper_replica_budget=%s\n' \
    "$CAMPAIGN" "$N" "$ENSEMBLE" "$REPLICAS" "$PER_REPLICA_BUDGET"
  printf 'census_radius=%s\nslice=%s\npopulation_interval=%s\nworker_opts=%s\n' \
    "$CENSUS_RADIUS" "$SLICE" "$POPULATION_INTERVAL" "$WORKER_OPTS"
  printf 'source_commit=%s\nbinary_sha256=%s\nserver_sha256=%s\nslurm_job_id=%s\nhost=%s\n' \
    "$(cat "$SOURCE_COMMIT_FILE" 2>/dev/null || echo unknown)" \
    "$(sha256sum "$BIN" | awk '{print $1}')" "$(sha256sum "$SERVER" | awk '{print $1}')" \
    "$SLURM_JOB_ID" "$(hostname)"
  env | grep -E '^(CENSUS_BUS|GOSSIP|CATALOG_)' | sort
} >"$OUT/run.manifest"
pids=()
for replica in $(seq 0 $((REPLICAS - 1))); do
  seed=$((SEED_BASE + replica))
  worker=$OUT/workers/replica-${replica}
  mkdir -p "$worker"
  (
    export CATALOG_CAMPAIGN="$CAMPAIGN"
    export CATALOG_ENSEMBLE="$ENSEMBLE"
    export CATALOG_REPLICA="$replica"
    export CATALOG_REPLICAS="$REPLICAS"
    export CATALOG_SLICE="$SLICE"
    export CATALOG_TRANSPORT_NOISE="$TRANSPORT_NOISE"
    export CATALOG_TRANSPORT_RADIUS="$TRANSPORT_RADIUS"
    export CATALOG_POPULATION_INTERVAL="$POPULATION_INTERVAL"
    if [[ -n ${CATALOG_MAX_HOPS:-} ]]; then
      export CATALOG_MAX_HOPS
    fi
    export CATALOG_TRACE="$OUT/traces/replica-${replica}.jsonl"
    export ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json
    export SEED_OFFSET="$seed"
    export CATALOG_RPC="$endpoint"
    exec "$BIN" "$N" "$PER_REPLICA_BUDGET" 1 "$WORKER_OPTS"
  ) >"$OUT/workers/replica-${replica}.out" 2>"$OUT/workers/replica-${replica}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
stop_server
server_pid=
if (( status != 0 )); then
  echo "at least one LJ replica failed" >&2
  exit "$status"
fi
touch "$OUT/TERMINAL_OK"
printf '%s\n' "$OUT"
