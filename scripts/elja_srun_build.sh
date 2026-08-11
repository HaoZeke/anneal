#!/usr/bin/env bash
# One short-partition allocation: rebuild portable libira, then the
# featomic+ira+bank-rpc examples, then smoke on the same node.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
PART=${ELJA_PARTITION:-short}
CPUS=${ELJA_CPUS:-8}
TIME=${ELJA_TIME:-00:30:00}
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  bash "$ROOT/scripts/elja_rebuild_ira.sh"
  bash "$ROOT/scripts/elja_build_lj.sh"
  exit 0
fi
exec srun \
  --partition="$PART" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --time="$TIME" \
  --job-name=anneal-ira-build \
  --chdir="$ROOT" \
  bash "$ROOT/scripts/elja_srun_build.sh"
