#!/usr/bin/env bash
# One isolated four-replica LJ ensemble inside an Elja Slurm allocation.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_lj_ensemble.sh requires a Slurm allocation" >&2
  exit 1
fi

N=${1:?LJ site count}
PER_REPLICA_BUDGET=${2:?per-replica charged budget}
ARM=${3:?shared or control}
ENSEMBLE_INDEX=${4:?ensemble index}
if [[ $ENSEMBLE_INDEX == slurm-array ]]; then
  ENSEMBLE_INDEX=${SLURM_ARRAY_TASK_ID:?Slurm array index}
fi
CENSUS_RADIUS=${5:?calibrated census radius}

case "$ARM" in
  shared|control) ;;
  *)
    echo "arm must be shared or control" >&2
    exit 2
    ;;
esac

ROOT=${LJ_ROOT:-$HOME/anneal-build}
REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
SERVER=${CATALOG_SERVER_BIN:-$ROOT/target/release/examples/catalog_server}
QUALIFIER=${JCC_QUALIFIER:-$REPRO_ROOT/workflow/jcc/validate_hard_lj_qualification.py}
QUALIFIER_PYTHON=${JCC_QUALIFIER_PYTHON:-$REPRO_ROOT/.pixi/envs/default/bin/python}
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
CALIBRATION=$REPRO_ROOT/results_jcc/calibration/lj${N}.json
CAMPAIGN=${CATALOG_CAMPAIGN:-jcc-2026-development}
ENSEMBLE="lj${N}-${ARM}-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT_ROOT=${LJ_OUT:-$HOME/ljwork/jcc}
OUT="$OUT_ROOT/$CAMPAIGN/lj${N}/$ARM/$ENSEMBLE"
CAPACITY=${CATALOG_CAPACITY:-30}
SLICE=${CATALOG_SLICE:-500}
HOLE_SAMPLES=${CATALOG_HOLE_SAMPLES:-256}
POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
MINIMUM_POPULATION_INTERVAL=$((2 * SLICE + 2))
if (( POPULATION_INTERVAL < MINIMUM_POPULATION_INTERVAL )); then
  POPULATION_INTERVAL=$MINIMUM_POPULATION_INTERVAL
fi
REPLICAS=4
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=$(( ${SEED_OFFSET_BASE:-0} + ENSEMBLE_INDEX * REPLICAS ))

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
  if ldd "$executable" | grep -q "not found"; then
    echo "unresolved libraries in $executable" >&2
    ldd "$executable" >&2
    exit 1
  fi
done
if [[ ! -f $QUALIFIER ]]; then
  echo "missing hard-LJ qualifier: $QUALIFIER" >&2
  exit 1
fi
if [[ ! -x $QUALIFIER_PYTHON ]]; then
  echo "missing qualification Python: $QUALIFIER_PYTHON" >&2
  exit 1
fi
if [[ ! -f $SOURCE_COMMIT_FILE ]]; then
  echo "missing source commit record: $SOURCE_COMMIT_FILE" >&2
  exit 1
fi
if [[ ! -f $CALIBRATION ]]; then
  echo "missing census-radius calibration: $CALIBRATION" >&2
  exit 1
fi
IFS= read -r SOURCE_COMMIT <"$SOURCE_COMMIT_FILE"
if [[ ! $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit record must contain one full Git object ID" >&2
  exit 1
fi

server_pid=
stop_server() {
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT

endpoint=
if [[ $ARM == shared ]]; then
  "$SERVER" \
    127.0.0.1:0 \
    "$N" \
    "$CAPACITY" \
    "$CENSUS_RADIUS" \
    "$TOTAL_BUDGET" \
    "$CAMPAIGN" \
    "$ENSEMBLE" \
    0,1,2,3 \
    "$OUT/state" \
    >"$OUT/coordinator.jsonl" 2>"$OUT/coordinator.err" &
  server_pid=$!
  for _ in $(seq 1 100); do
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
fi

pids=()
for replica in 0 1 2 3; do
  seed=$((SEED_BASE + replica))
  worker=$OUT/workers/replica-${replica}
  args=rec
  if [[ $ARM == control ]]; then
    args=rec,catalog
  fi
  (
    export CATALOG_CAMPAIGN="$CAMPAIGN"
    export CATALOG_ENSEMBLE="$ENSEMBLE"
    export CATALOG_REPLICA="$replica"
    export CATALOG_SLICE="$SLICE"
    export CATALOG_HOLE_SAMPLES="$HOLE_SAMPLES"
    export CATALOG_POPULATION_INTERVAL="$POPULATION_INTERVAL"
    export CATALOG_TRACE="$OUT/traces/replica-${replica}.jsonl"
    export ANNEAL_RESOLVED_CONFIG=$worker.resolved-config.json
    export SEED_OFFSET="$seed"
    if [[ $ARM == shared ]]; then
      export CATALOG_RPC="$endpoint"
    else
      unset CATALOG_RPC BANK_RPC
    fi
    exec "$BIN" "$N" "$PER_REPLICA_BUDGET" 1 "$args"
  ) >"$OUT/workers/replica-${replica}.out" 2>"$OUT/workers/replica-${replica}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "at least one LJ replica failed" >&2
  exit "$status"
fi
for replica in 0 1 2 3; do
  test -s "$OUT/workers/replica-${replica}.resolved-config.json"
done
for replica in 1 2 3; do
  cmp -s "$OUT/workers/replica-0.resolved-config.json" \
    "$OUT/workers/replica-${replica}.resolved-config.json"
done

if [[ $ARM == shared ]] && ! kill -0 "$server_pid" 2>/dev/null; then
  wait "$server_pid" || true
  echo "catalog coordinator exited before the ensemble terminal boundary" >&2
  cat "$OUT/coordinator.err" >&2
  exit 1
fi

stop_server
server_pid=

for replica in 0 1 2 3; do
  output="$OUT/workers/replica-${replica}.out"
  trace="$OUT/traces/replica-${replica}.jsonl"
  test -s "$output"
  test -s "$trace"
  grep -q "charged ${PER_REPLICA_BUDGET}" "$output"
  grep -q '"kind":"manifest_header"' "$trace"
  if [[ $ARM == control ]]; then
    grep -q '"sharing":false' "$trace"
    if grep -q '"kind":"rpc_fallback"' "$trace"; then
      echo "control replica $replica recorded an RPC fallback" >&2
      exit 1
    fi
  else
    grep -q '"sharing":true' "$trace"
    if (( POPULATION_INTERVAL <= PER_REPLICA_BUDGET - 4 )); then
      grep -q '"kind":"population_ready"' "$trace"
    fi
  fi
done

if [[ $ARM == shared ]]; then
  awk -v expected="\"aggregate_charged\":${TOTAL_BUDGET}" '
    index($0, expected) { found = 1 }
    END { exit !found }
  ' "$OUT"/traces/replica-*.jsonl
fi

"$QUALIFIER_PYTHON" "$QUALIFIER" \
  --output "$OUT/qualification.json" \
  "$OUT/traces/replica-0.jsonl" \
  "$OUT/traces/replica-1.jsonl" \
  "$OUT/traces/replica-2.jsonl" \
  "$OUT/traces/replica-3.jsonl"

cp "$OUT/workers/replica-0.resolved-config.json" "$OUT/resolved-config.json"
resolved_config_sha256=$(sha256sum "$OUT/resolved-config.json" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
runner_sha256=$(sha256sum "$0" | awk '{print $1}')
qualifier_sha256=$(sha256sum "$QUALIFIER" | awk '{print $1}')
calibration_sha256=$(sha256sum "$CALIBRATION" | awk '{print $1}')

{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'system=lj%s\n' "$N"
  printf 'arm=%s\n' "$ARM"
  printf 'ensemble=%s\n' "$ENSEMBLE"
  printf 'replicas=%s\n' "$REPLICAS"
  printf 'per_replica_budget=%s\n' "$PER_REPLICA_BUDGET"
  printf 'aggregate_budget=%s\n' "$TOTAL_BUDGET"
  printf 'seed_base=%s\n' "$SEED_BASE"
  printf 'catalog_capacity=%s\n' "$CAPACITY"
  printf 'census_radius=%s\n' "$CENSUS_RADIUS"
  printf 'slice=%s\n' "$SLICE"
  printf 'hole_samples=%s\n' "$HOLE_SAMPLES"
  printf 'population_interval=%s\n' "$POPULATION_INTERVAL"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'requested_options=%s\n' "$args"
  printf 'resolved_config_sha256=%s\n' "$resolved_config_sha256"
  printf 'binary_sha256=%s\n' "$binary_sha256"
  printf 'runner_sha256=%s\n' "$runner_sha256"
  printf 'qualifier_sha256=%s\n' "$qualifier_sha256"
  printf 'calibration_sha256=%s\n' "$calibration_sha256"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  if [[ $ARM == shared ]]; then
    printf 'server_sha256=%s\n' "$(sha256sum "$SERVER" | awk '{print $1}')"
  fi
} >"$OUT/run.manifest"

touch "$OUT/TERMINAL_OK"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >SHA256SUMS)
printf '%s\n' "$OUT"
