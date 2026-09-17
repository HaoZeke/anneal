#!/usr/bin/env bash
# Recognition communicating arm compiles and its unit tests hold.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_tm8k_gate.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-recognition/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-recognition/target}"
mkdir -p "$CARGO_TARGET_DIR"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
if [[ -d .git ]]; then
  echo "source=$(git rev-parse HEAD)"
else
  echo "source=unpinned-rsync"
fi
if ! rg -n 'fn communicating' src/methods/cluster_hopping/config.rs; then
  echo "COMMUNICATING_PRESET_MISSING" >&2
  exit 1
fi
if ! rg -n 'SharedVisitPolicy::Recognition' src/methods/cluster_hopping.rs; then
  echo "RECOGNITION_NOT_WIRED" >&2
  exit 1
fi
cargo test --lib -- \
  recognition_keeps_a_peer_basin_new_for_this_chain \
  shared_history_marks_another_replicas_basin_as_known \
  communicating_is_orbit_without_depth_reward \
  recognition_does_not_count_a_peer_well_as_known_escape \
  a_shared_history_pays_deposits_for_the_other_chains_visits
echo RECOGNITION_OK
