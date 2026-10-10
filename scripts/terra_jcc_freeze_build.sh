#!/usr/bin/env bash
# Build the LJ cluster search binary for the frozen-recommended JCC campaign.
#
# The freeze is `Config::recommended` = for_cluster + Thompson + return screen
# + orbit completion, so every `rec` arm run against this binary measures the
# frozen method rather than the earlier LeanBurst stack.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_jcc_freeze_build.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${JCC_ROOT:-$HOME/Git/Github/Rust/anneal}
export CARGO_TARGET_DIR="${JCC_TARGET:-$ROOT/target}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
echo "recommended():"
sed -n '/pub fn recommended(n_points/,+6p' src/methods/cluster_hopping/config.rs

cargo build --release --features featomic,ira,bank-rpc --example lj_cluster_search

BIN="$CARGO_TARGET_DIR/release/examples/lj_cluster_search"
sha256sum "$BIN" | tee "$ROOT/JCC_FREEZE_EXE_SHA"
echo "built $BIN"
