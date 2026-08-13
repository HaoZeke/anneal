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

ROOT=${LJ_ROOT:-$HOME/anneal-build}
RGPOT=${RGPOT_ROOT:-$HOME/rgpot}
CAMPAIGN=${MOLSLAB_CAMPAIGN:-jcc-2026-development}
OUT_ROOT=${MOLSLAB_OUT:-$HOME/ljwork/jcc}
CAPACITY=${BANK_CAPACITY:-30}
SLICE=${BANK_SLICE:-500}
SYNC_INTERVAL=${BANK_SYNC:-1}
REPLICAS=4
TOTAL_BUDGET=$((PER_REPLICA_BUDGET * REPLICAS))
SEED_BASE=$((${SEED_OFFSET_BASE:-0} + ENSEMBLE_INDEX * REPLICAS))
ENSEMBLE="${SYSTEM}-${ARM}-$(printf '%04d' "$ENSEMBLE_INDEX")"
OUT="$OUT_ROOT/$CAMPAIGN/$SYSTEM/$ARM/$ENSEMBLE"
SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$ROOT/SOURCE_COMMIT}
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
export LD_LIBRARY_PATH="${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
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
  "$SERVER" 127.0.0.1:0 "$CAPACITY" >"$OUT/bank-server.out" 2>"$OUT/bank-server.err" &
  server_pid=$!
  for _ in $(seq 1 100); do
    endpoint=$(grep -oE 'bank listening on [^ ]+' "$OUT/bank-server.err" 2>/dev/null \
      | head -1 | awk '{print $4}' || true)
    if [[ -n $endpoint ]]; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "bank server exited during startup" >&2
      cat "$OUT/bank-server.err" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ -z $endpoint ]]; then
    echo "bank server did not publish its allocated address" >&2
    exit 1
  fi
fi

pids=()
for replica in 0 1 2 3; do
  seed=$((SEED_BASE + replica))
  worker=$OUT/workers/replica-$replica
  (
    export SEED_OFFSET=$seed
    export TARGET_ENERGY=$TARGET
    export TARGET_TOL=1e-3
    export BANK_SLICE=$SLICE
    export BANK_SYNC=$SYNC_INTERVAL
    if [[ $SYSTEM == h2o2 || $SYSTEM == h2o4 || $SYSTEM == h2o6 ]]; then
      export RGPOT_XTB_ENGINE=$ENGINE
      export RGPOT_XTB_TRACE=$worker/last-request.txt
    else
      export RGPOT_CUH2_LIBRARY=$ENGINE
    fi
    if [[ $ARM == shared ]]; then
      export BANK_RPC=$endpoint
    else
      unset BANK_RPC
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
if [[ $ARM == shared ]]; then
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    echo "bank server exited before the ensemble terminal boundary" >&2
    cat "$OUT/bank-server.err" >&2
    exit 1
  fi
  "$PEEK" "$endpoint" >"$OUT/bank-snapshot.txt"
  stop_server
  server_pid=
fi

label=nobank
if [[ $ARM == shared ]]; then
  label=bank
fi
for replica in 0 1 2 3; do
  worker=$OUT/workers/replica-$replica
  output=$worker/stdout.log
  test -s "$output"
  test ! -s "$worker/stderr.log"
  grep -q "charged ${PER_REPLICA_BUDGET}" "$output"
  grep -q "arm ${label}" "$output"
  grep -q "encounter target=" "$output"
  if grep -q "best inf eV" "$output"; then
    if find "$worker" -maxdepth 1 -name 'best_*.xyz' -print -quit | grep -q .; then
      echo "replica $replica wrote an unvalidated infinite-best structure" >&2
      exit 1
    fi
  else
    grep -q "verify e=" "$output"
  fi
done

grep -h "encounter target=" "$OUT"/workers/replica-*/stdout.log >"$OUT/encounters.txt"
grep -h "verify e=" "$OUT"/workers/replica-*/stdout.log >"$OUT/verifications.txt" || true
{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'system=%s\n' "$SYSTEM"
  printf 'arm=%s\n' "$ARM"
  printf 'ensemble=%s\n' "$ENSEMBLE"
  printf 'replicas=%s\n' "$REPLICAS"
  printf 'per_replica_budget=%s\n' "$PER_REPLICA_BUDGET"
  printf 'aggregate_budget=%s\n' "$TOTAL_BUDGET"
  printf 'seed_base=%s\n' "$SEED_BASE"
  printf 'target_energy=%s\n' "$TARGET"
  printf 'bank_capacity=%s\n' "$CAPACITY"
  printf 'bank_slice=%s\n' "$SLICE"
  printf 'bank_sync=charged_slices\n'
  printf 'bank_sync_interval=%s\n' "$SYNC_INTERVAL"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  sha256sum "$BIN"
  sha256sum "$ENGINE"
  sha256sum "$0"
  if [[ $SYSTEM == cuh2 ]]; then
    sha256sum "$CON"
  fi
  if [[ $ARM == shared ]]; then
    sha256sum "$SERVER"
    sha256sum "$PEEK"
  fi
} >"$OUT/run.manifest"

touch "$OUT/TERMINAL_OK"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >SHA256SUMS)
printf '%s\n' "$OUT"
