#!/usr/bin/env bash
# One seed inside a HyperQueue worker. The worker is one Slurm allocation.
# HQ_TASK_ID is the seed. The program runs on /scratch/users/$USER.
# One log is copied back when the seed finishes.
set -euo pipefail
N=${1:?n}
BUDGET=${2:?budget}
ARM=${3:?rec|base}
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/scripts/elja_scratch.sh"
elja_enter_scratch
trap elja_leave_scratch EXIT
export SEED_OFFSET=${SEED_OFFSET:-${HQ_TASK_ID:-0}}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
BIN=${LJ_BIN:?set LJ_BIN to the built lj_cluster_search}
RECORD=${LJ_RECORD:?set LJ_RECORD to the directory that receives the finished log}
cd "$ELJA_SCRATCH"
"$BIN" "$N" "$BUDGET" 1 "$ARM" >"$ELJA_SCRATCH/seed.log" 2>&1
elja_publish "$ELJA_SCRATCH/seed.log" "$RECORD/n${N}-${ARM}-seed${SEED_OFFSET}.log"
