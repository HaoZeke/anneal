#!/usr/bin/env bash
# Values-only communicating replicas: cargo test box_hopping on terra.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_values_replicas.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-values/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-values/target}"
mkdir -p "$CARGO_TARGET_DIR"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
if [[ -d .git ]]; then
  echo "source=$(git rev-parse HEAD)"
else
  echo "source=unpinned-rsync"
fi
cargo test --lib box_hopping -- --nocapture
echo VALUES_REPLICAS_OK
