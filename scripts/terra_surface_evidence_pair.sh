#!/usr/bin/env bash
# Evaluation-matched independent ensembles with surface evidence as the only
# communication channel. This runner executes existing binaries; it builds none.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" && -z "${INVOCATION_ID:-}" ]]; then
  printf 'a bounded remote allocation is required\n' >&2
  exit 2
fi
: "${ANNEAL_PAIR_BIN:?set the built example directory}"
: "${ANNEAL_PAIR_OUTPUT:?set a fresh result directory}"
: "${ANNEAL_PAIR_SOURCE_REV:?set the binary source revision}"
: "${ANNEAL_PAIR_FEATURES:?set the binary feature set}"
n=${ANNEAL_PAIR_N:-13}
budget=${ANNEAL_PAIR_BUDGET:-20000}
replicas=${ANNEAL_PAIR_REPLICAS:-4}
ensembles=${ANNEAL_PAIR_ENSEMBLES:-1}
block=${ANNEAL_PAIR_BLOCK:-5}
for value in "$n" "$budget" "$replicas" "$ensembles" "$block"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'positive integer parameters required\n' >&2; exit 2; }
done
mkdir -- "$ANNEAL_PAIR_OUTPUT"
mkdir -- "$ANNEAL_PAIR_OUTPUT/bin"
for program in catalog_server lj_cluster_search; do
  test -x "$ANNEAL_PAIR_BIN/$program"
  cp -- "$ANNEAL_PAIR_BIN/$program" "$ANNEAL_PAIR_OUTPUT/bin/$program"
done
pair_bin="$ANNEAL_PAIR_OUTPUT/bin"
{
  printf 'source=%s\nfeatures=%s\nn=%s\nbudget=%s\nreplicas=%s\nensembles=%s\nblock=%s\n' \
    "$ANNEAL_PAIR_SOURCE_REV" "$ANNEAL_PAIR_FEATURES" "$n" "$budget" "$replicas" "$ensembles" "$block"
  printf 'options=catalog,surfaces,noclimb\nchannels=surface-evidence-only\n'
  sha256sum "$pair_bin/catalog_server" "$pair_bin/lj_cluster_search"
} >"$ANNEAL_PAIR_OUTPUT/manifest.txt"
server_pid=
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
roster=0
for ((replica=1; replica<replicas; replica++)); do roster+=",$replica"; done
for ((ensemble_index=0; ensemble_index<ensembles; ensemble_index++)); do
  for mode in private shared; do
    ensemble="surface-$n-$ensemble_index-$mode"
    endpoint=
    if [[ "$mode" == shared ]]; then
      server_log="$ANNEAL_PAIR_OUTPUT/server-$ensemble_index.log"
      env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        "$pair_bin/catalog_server" 127.0.0.1:0 "$n" 64 0.1 "$((budget * replicas))" \
        surface-evidence "$ensemble" "$roster" >"$server_log" 2>&1 &
      server_pid=$!
      for ((attempt=0; attempt<200; attempt++)); do
        endpoint=$(sed -n 's/.*"addr":"\([^"]*\)".*/\1/p' "$server_log" | sed -n '1p')
        [[ -n "$endpoint" ]] && break
        kill -0 "$server_pid" || { printf 'coordinator failed\n' >&2; exit 1; }
        sleep 0.1
      done
      [[ -n "$endpoint" ]] || { printf 'coordinator readiness failed\n' >&2; exit 1; }
    fi
    worker_pids=()
    for ((replica=0; replica<replicas; replica++)); do
      env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        CATALOG_CAMPAIGN=surface-evidence CATALOG_ENSEMBLE="$ensemble" \
        CATALOG_REPLICA="$replica" CATALOG_SHARING="$mode" CATALOG_RPC="$endpoint" \
        CATALOG_EVIDENCE_ONLY=1 SURFACES=kappa:0.7,mu:5 CATALOG_SLICE=500 \
        SURFACE_BLOCK="$block" SEED_OFFSET="$((ensemble_index * replicas + replica))" \
        "$pair_bin/lj_cluster_search" "$n" "$budget" 1 catalog,surfaces,noclimb \
        >"$ANNEAL_PAIR_OUTPUT/$mode-$ensemble_index-$replica.log" 2>&1 &
      worker_pids+=("$!")
    done
    status=0
    for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
    worker_pids=()
    [[ "$status" == 0 ]] || { printf '%s workers failed\n' "$mode" >&2; exit 1; }
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid"
      wait "$server_pid" || true
      server_pid=
    fi
  done
done
rg 'SURFACE_EVIDENCE|seed +[0-9].*best|policy: leaves' "$ANNEAL_PAIR_OUTPUT" -g '*.log'
