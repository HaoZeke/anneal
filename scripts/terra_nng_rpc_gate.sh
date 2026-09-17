#!/usr/bin/env bash
# Compile and gate the Cap'n-on-nng transport swap.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_nng_rpc_gate.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-nng-rpc/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-nng-rpc/target}"
mkdir -p "$CARGO_TARGET_DIR"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
echo "source=$(git rev-parse HEAD)"
# Carrier first, then bank and the catalog vat on that carrier.
cargo test --features bank-rpc --lib nng_rpc -- --nocapture
cargo test --features bank-rpc --lib bank_rpc -- --nocapture
cargo test --features bank-rpc --test catalog_rpc attach_binds_identity -- --nocapture --exact
cargo test --features bank-rpc --test catalog_rpc --test catalog_rpc_faults -- --test-threads=2
echo GATE_OK
