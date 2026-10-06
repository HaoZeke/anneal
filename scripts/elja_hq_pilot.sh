#!/usr/bin/env bash
# One HyperQueue server and one Slurm node.
# QCG-PilotJob is the same shape (many tasks inside one allocation) and is
# not installed here. HyperQueue 0.19 is. Tasks do not use the home filer
# as their working directory: elja_hq_one.sh runs on /scratch/users/$USER.
#
#   scripts/elja_hq_pilot.sh
#   scripts/elja_hq_pilot.sh launch 75 4000000 1 rec
#
# launch is the only path that talks to Slurm. The default prints this
# usage and submits nothing.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/scripts/elja_scratch.sh"

usage() {
  echo "usage: scripts/elja_hq_pilot.sh launch N BUDGET SEEDS ARM" >&2
  echo "one Slurm allocation (--max-worker-count 1 --exclusive); tasks on /scratch/users/\$USER" >&2
}

if [[ ${1:-} != launch ]]; then
  usage
  exit 0
fi
echo "refusing: a seed is one Slurm job, not a pack of seeds on one node." >&2
echo "uv run --script scripts/elja_qcg_cell.py plan" >&2
exit 2
shift
N=${1:?n}
BUDGET=${2:?budget}
SEEDS=${3:?seeds}
ARM=${4:?arm}
if [[ $SEEDS -lt 1 ]]; then
  echo "seeds must be at least 1" >&2
  exit 2
fi
# A cap keeps one launch from filling the shared 30-running quota.
if [[ $SEEDS -gt 48 ]]; then
  echo "refusing $SEEDS seeds; one node holds at most 48" >&2
  exit 2
fi
BIN=${LJ_BIN:?set LJ_BIN to the built lj_cluster_search}
RECORD=${LJ_RECORD:-$HOME/ljwork/hq-record}
if [[ ! -x $BIN ]]; then
  echo "missing executable $BIN" >&2
  exit 2
fi
mkdir -p "$RECORD"
LAST=$((SEEDS - 1))
export HQ_SERVER_DIR=${HQ_SERVER_DIR:-$HOME/.hq-anneal}
if ! hq server info >/dev/null 2>&1; then
  hq server start
fi
# Dry-run is a real Slurm submit. --no-dry-run is the single allocation.
hq alloc add slurm \
  --name "anneal-lj${N}" \
  --time-limit 8h \
  --max-worker-count 1 \
  --workers-per-alloc 1 \
  --backlog 1 \
  --no-hyper-threading \
  --idle-timeout 10min \
  --no-dry-run \
  --worker-start-cmd 'mkdir -p /scratch/users/$USER/hq' \
  -- \
  --partition=s-normal \
  --account=chem-ui \
  --nodes=1 \
  --exclusive \
  --time=08:00:00
hq submit \
  --name "lj${N}-${ARM}" \
  --array="0-${LAST}" \
  --cpus 1 \
  --time-limit 8h \
  --stdout "/scratch/users/${USER}/hq/lj${N}_${ARM}_%{TASK_ID}.out" \
  --stderr "/scratch/users/${USER}/hq/lj${N}_${ARM}_%{TASK_ID}.err" \
  --env "LJ_BIN=${BIN}" \
  --env "LJ_RECORD=${RECORD}" \
  -- "$ROOT/scripts/elja_hq_one.sh" "$N" "$BUDGET" "$ARM"
