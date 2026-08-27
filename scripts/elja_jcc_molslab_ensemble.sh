#!/usr/bin/env bash
# One isolated four-replica molecule or slab ensemble inside an Elja allocation.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_molslab_ensemble.sh requires a Slurm allocation" >&2
  exit 1
fi

SYSTEM=${1:?h2o2, h2o4, h2o6, or cuh2}
PER_REPLICA_BUDGET=${2:?per-replica charged budget}
ARM=${3:?shared or control}
ENSEMBLE_INDEX=${4:?ensemble index}
SOAP_MODE=${5:?flexible, rigid, or off}
if [[ $ENSEMBLE_INDEX == slurm-array ]]; then
  ENSEMBLE_INDEX=${SLURM_ARRAY_TASK_ID:?Slurm array index}
fi
case $ARM in
  shared|control) ;;
  *)
    echo "arm must be shared or control" >&2
    exit 2
    ;;
esac
case $SOAP_MODE in
  flexible|rigid|off) ;;
  *)
    echo "SOAP mode must be flexible, rigid, or off" >&2
    exit 2
    ;;
esac

ROOT=${LJ_ROOT:-$HOME/anneal-build}
SOURCE_ROOT=${JCC_SOURCE_ROOT:-$ROOT}
RGPOT=${RGPOT_ROOT:-$HOME/rgpot}
CAMPAIGN=${MOLSLAB_CAMPAIGN:-jcc-2026-development}
OUT_ROOT=${MOLSLAB_OUT:-$HOME/ljwork/jcc}
CAPACITY=${BANK_CAPACITY:-30}
SLICE=${BANK_SLICE:-500}
SYNC_INTERVAL=${BANK_SYNC:-1}
TARGET_TOLERANCE=1e-3
REPLICAS=4
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=$((${SEED_OFFSET_BASE:-0} + ENSEMBLE_INDEX * REPLICAS))
ENSEMBLE="${SYSTEM}-${SOAP_MODE}-${ARM}-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT="$OUT_ROOT/$CAMPAIGN/$SYSTEM/$ARM/$ENSEMBLE"
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
RGPOT_SOURCE_COMMIT_FILE=${JCC_RGPOT_SOURCE_COMMIT_FILE:-$ROOT/RGPOT_SOURCE_COMMIT}
RGPOT_PIXI_LOCK_SHA256_FILE=${JCC_RGPOT_PIXI_LOCK_SHA256_FILE:-$ROOT/RGPOT_PIXI_LOCK_SHA256}
SERVER=$ROOT/target/release/examples/bank_server
PEEK=$ROOT/target/release/examples/bank_peek
CON=$ROOT/examples/fixtures/cuh2_fcc_slab.con
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTBLIB=${XTBLIB:-$RGPOT/.pixi/envs/xtbbld/lib}

case $SYSTEM in
  h2o2)
    BIN=$ROOT/target/release/examples/molecular_cluster
    ENGINE=$ROOT/engines/libxtb_engine.so
    TARGET=-276.168547
    COMMAND_PREFIX=("$BIN" 2)
    ;;
  h2o4)
    BIN=$ROOT/target/release/examples/molecular_cluster
    ENGINE=$ROOT/engines/libxtb_engine.so
    TARGET=-553.064301
    COMMAND_PREFIX=("$BIN" 4)
    ;;
  h2o6)
    BIN=$ROOT/target/release/examples/molecular_cluster
    ENGINE=$ROOT/engines/libxtb_engine.so
    TARGET=-829.846965
    COMMAND_PREFIX=("$BIN" 6)
    ;;
  cuh2)
    BIN=$ROOT/target/release/examples/slab_adsorption
    ENGINE=$ROOT/engines/librgpot_cuh2.so
    TARGET=-415.971529
    COMMAND_PREFIX=("$BIN" "$CON")
    ;;
  *)
    echo "system must be h2o2, h2o4, h2o6, or cuh2" >&2
    exit 2
    ;;
esac

if [[ -e $OUT ]]; then
  echo "ensemble output already exists: $OUT" >&2
  exit 1
fi
for executable in "$BIN" "$SERVER" "$PEEK"; do
  if [[ ! -x $executable ]]; then
    echo "missing executable: $executable" >&2
    exit 1
  fi
done
if [[ ! -e $ENGINE ]]; then
  echo "missing engine: $ENGINE" >&2
  exit 1
fi
if [[ $SYSTEM == cuh2 && ! -f $CON ]]; then
  echo "missing slab input: $CON" >&2
  exit 1
fi
if [[ ! -f $SOURCE_COMMIT_FILE ]]; then
  echo "missing source commit record: $SOURCE_COMMIT_FILE" >&2
  exit 1
fi
IFS= read -r SOURCE_COMMIT <"$SOURCE_COMMIT_FILE"
if [[ ! $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit record must contain one full Git object ID" >&2
  exit 1
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
if [[ ! -s $ROOT/MOLSLAB_BUILD_SHA256SUMS ]]; then
  echo "missing molecular build artifact seal: $ROOT/MOLSLAB_BUILD_SHA256SUMS" >&2
  exit 2
fi
(cd "$ROOT" && sha256sum -c MOLSLAB_BUILD_SHA256SUMS)
if [[ ! -s $RGPOT_SOURCE_COMMIT_FILE || ! -s $RGPOT_PIXI_LOCK_SHA256_FILE ]]; then
  echo "missing rgpot source provenance below $ROOT" >&2
  exit 2
fi
IFS= read -r RGPOT_SOURCE_COMMIT <"$RGPOT_SOURCE_COMMIT_FILE"
IFS= read -r RGPOT_PIXI_LOCK_SHA256 <"$RGPOT_PIXI_LOCK_SHA256_FILE"
if [[ ! $RGPOT_SOURCE_COMMIT =~ ^[0-9a-f]{40}$ || ! $RGPOT_PIXI_LOCK_SHA256 =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid rgpot source provenance below $ROOT" >&2
  exit 2
fi
RGPOT_HEAD=$(git -C "$RGPOT" rev-parse HEAD)
if [[ $RGPOT_SOURCE_COMMIT != "$RGPOT_HEAD" ]]; then
  echo "RGPOT_SOURCE_COMMIT=$RGPOT_SOURCE_COMMIT does not match HEAD=$RGPOT_HEAD" >&2
  exit 2
fi
if ! git -C "$RGPOT" diff --quiet HEAD --; then
  echo "rgpot tracked source differs from HEAD=$RGPOT_HEAD" >&2
  git -C "$RGPOT" status --short >&2
  exit 2
fi
RGPOT_LOCK_ACTUAL=$(sha256sum "$RGPOT/pixi.lock" | awk '{print $1}')
if [[ $RGPOT_PIXI_LOCK_SHA256 != "$RGPOT_LOCK_ACTUAL" ]]; then
  echo "rgpot Pixi lock digest does not match $RGPOT/pixi.lock" >&2
  exit 2
fi
export LD_LIBRARY_PATH="${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
for executable in "$BIN" "$SERVER" "$PEEK"; do
  if ldd "$executable" | grep -q "not found"; then
    echo "unresolved libraries in $executable" >&2
    ldd "$executable" >&2
    exit 1
  fi
done

mkdir -p "$OUT/workers"
for replica in 0 1 2 3; do
  mkdir -p "$OUT/workers/replica-$replica"
done

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

start_bank() {
  local prefix=$1
  local pid endpoint=
  "$SERVER" 127.0.0.1:0 "$CAPACITY" >"${prefix}.out" 2>"${prefix}.err" &
  pid=$!
  server_pids+=("$pid")
  server_prefixes+=("$prefix")
  for _ in $(seq 1 100); do
    endpoint=$(grep -oE 'bank listening on [^ ]+' "${prefix}.err" 2>/dev/null \
      | head -1 | awk '{print $4}' || true)
    if [[ -n $endpoint ]]; then
      last_started_endpoint=$endpoint
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "bank server exited during startup: $prefix" >&2
      cat "${prefix}.err" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "bank server did not publish its allocated address: $prefix" >&2
  exit 1
}

if [[ $ARM == shared ]]; then
  start_bank "$OUT/bank-shared"
  shared_endpoint=$last_started_endpoint
  bank_topology=shared
else
  for replica in 0 1 2 3; do
    start_bank "$OUT/bank-private-$replica"
    private_endpoints[$replica]=$last_started_endpoint
  done
  bank_topology=private_per_replica
fi

pids=()
for replica in 0 1 2 3; do
  seed=$((SEED_BASE + replica))
  worker=$OUT/workers/replica-$replica
  (
    export SEED_OFFSET=$seed
    export TARGET_ENERGY=$TARGET
    export TARGET_TOL=$TARGET_TOLERANCE
    export BANK_SLICE=$SLICE
    export BANK_SYNC=$SYNC_INTERVAL
    export BANK_SHARING=$ARM
    export ANNEAL_SOAP_MODE=$SOAP_MODE
    export ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json
    if [[ $SYSTEM == h2o2 || $SYSTEM == h2o4 || $SYSTEM == h2o6 ]]; then
      export RGPOT_XTB_ENGINE=$ENGINE
      export RGPOT_XTB_TRACE=$worker/last-request.txt
    else
      export RGPOT_CUH2_LIBRARY=$ENGINE
    fi
    if [[ $ARM == shared ]]; then
      export BANK_RPC=$shared_endpoint
    else
      export BANK_RPC=${private_endpoints[$replica]}
    fi
    cd "$worker"
    exec "${COMMAND_PREFIX[@]}" "$PER_REPLICA_BUDGET" 1
  ) >"$worker/stdout.log" 2>"$worker/stderr.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if ((status != 0)); then
  echo "at least one molecule/slab replica failed" >&2
  exit "$status"
fi
for index in "${!server_pids[@]}"; do
  if ! kill -0 "${server_pids[$index]}" 2>/dev/null; then
    wait "${server_pids[$index]}" || true
    echo "bank server exited before the ensemble terminal boundary: ${server_prefixes[$index]}" >&2
    cat "${server_prefixes[$index]}.err" >&2
    exit 1
  fi
done
if [[ $ARM == shared ]]; then
  "$PEEK" "$shared_endpoint" >"$OUT/bank-snapshot.txt"
else
  for replica in 0 1 2 3; do
    "$PEEK" "${private_endpoints[$replica]}" >"$OUT/bank-snapshot-replica-$replica.txt"
  done
fi
stop_servers

label=$ARM
for replica in 0 1 2 3; do
  worker=$OUT/workers/replica-$replica
  output=$worker/stdout.log
  test -s "$output"
  test ! -s "$worker/stderr.log"
  if ! grep -F -q "capnp bank " "$output"; then
    echo "replica $replica did not connect to its bank" >&2
    exit 1
  fi
  if grep -F -q " own walk" "$output"; then
    echo "replica $replica fell back to an unbanked walk" >&2
    exit 1
  fi
  grep -q "charged ${PER_REPLICA_BUDGET}" "$output"
  grep -q "arm ${label}" "$output"
  grep -q "encounter target=" "$output"
  test -s "$worker/resolved-config.json"
  if grep -q "best inf eV" "$output"; then
    if find "$worker" -maxdepth 1 -name 'best_*.xyz' -print -quit | grep -q .; then
      echo "replica $replica wrote an unvalidated infinite-best structure" >&2
      exit 1
    fi
  else
    grep -q "verify e=" "$output"
  fi
done
for replica in 1 2 3; do
  cmp -s "$OUT/workers/replica-0/resolved-config.json" \
    "$OUT/workers/replica-$replica/resolved-config.json"
done

grep -h "encounter target=" "$OUT"/workers/replica-*/stdout.log >"$OUT/encounters.txt"
grep -h "verify e=" "$OUT"/workers/replica-*/stdout.log >"$OUT/verifications.txt" || true
cp "$OUT/workers/replica-0/resolved-config.json" "$OUT/resolved-config.json"
resolved_config_sha256=$(sha256sum "$OUT/resolved-config.json" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
engine_sha256=$(sha256sum "$ENGINE" | awk '{print $1}')
runner_sha256=$(sha256sum "$0" | awk '{print $1}')
server_sha256=$(sha256sum "$SERVER" | awk '{print $1}')
peek_sha256=$(sha256sum "$PEEK" | awk '{print $1}')
{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'system=%s\n' "$SYSTEM"
  printf 'arm=%s\n' "$ARM"
  printf 'soap_mode=%s\n' "$SOAP_MODE"
  printf 'requested_options=%s\n' "soap_mode=$SOAP_MODE,sharing=$ARM"
  printf 'ensemble=%s\n' "$ENSEMBLE"
  printf 'replicas=%s\n' "$REPLICAS"
  printf 'per_replica_budget=%s\n' "$PER_REPLICA_BUDGET"
  printf 'aggregate_budget=%s\n' "$TOTAL_BUDGET"
  printf 'seed_base=%s\n' "$SEED_BASE"
  printf 'target_energy=%s\n' "$TARGET"
  printf 'target_tolerance=%s\n' "$TARGET_TOLERANCE"
  printf 'bank_capacity=%s\n' "$CAPACITY"
  printf 'bank_slice=%s\n' "$SLICE"
  printf 'bank_topology=%s\n' "$bank_topology"
  printf 'bank_sync=charged_slices\n'
  printf 'bank_sync_interval=%s\n' "$SYNC_INTERVAL"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'rgpot_source_commit=%s\n' "$RGPOT_SOURCE_COMMIT"
  printf 'rgpot_pixi_lock_sha256=%s\n' "$RGPOT_PIXI_LOCK_SHA256"
  printf 'resolved_config_sha256=%s\n' "$resolved_config_sha256"
  printf 'binary_sha256=%s\n' "$binary_sha256"
  printf 'engine_sha256=%s\n' "$engine_sha256"
  printf 'runner_sha256=%s\n' "$runner_sha256"
  printf 'server_sha256=%s\n' "$server_sha256"
  printf 'peek_sha256=%s\n' "$peek_sha256"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  if [[ $SYSTEM == cuh2 ]]; then
    printf 'input_sha256=%s\n' "$(sha256sum "$CON" | awk '{print $1}')"
  fi
} >"$OUT/run.manifest"

touch "$OUT/TERMINAL_OK"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >SHA256SUMS)
printf '%s\n' "$OUT"
