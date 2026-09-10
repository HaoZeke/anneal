#!/usr/bin/env bash
# Prove MinimumHistory observe/mark over nng Req/Rep. No catalog vat, no pump.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_history_nng_gate.sh: run under srun/sbatch" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-nng-rpc/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-nng-rpc/target}"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
cargo test --features history-nng --lib history_nng -- --nocapture
echo HISTORY_NNG_OK
