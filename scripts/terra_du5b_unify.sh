#!/usr/bin/env bash
# Box-search unify: pad gone; search splits on gradient.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_du5b_unify.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-box/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-box/target}"
mkdir -p "$CARGO_TARGET_DIR"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
if [[ -d .git ]]; then
  echo "source=$(git rev-parse HEAD)"
else
  echo "source=unpinned-rsync"
fi
if ! command -v rg >/dev/null; then
  echo "rg required for the pad check" >&2
  exit 1
fi
if rg -n 'n_points = dim\.div_ceil|fn box_hop_config' src/methods/box_hopping.rs; then
  echo "PAD_STILL_PRESENT" >&2
  exit 1
fi
if [[ -e src/methods/cutest_ensemble.rs ]]; then
  echo "CUTEST_MODULE_STILL_PRESENT" >&2
  exit 1
fi
echo "PAD_GONE"
cargo test --lib box_hopping -- --nocapture
echo BOX_UNIFY_OK
