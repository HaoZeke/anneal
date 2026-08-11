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
  BIN=${LJ_BIN:-$ROOT/target/release/examples/lj_cluster_search}
  if [[ -x $BIN ]] && ldd "$BIN" >/dev/null 2>&1; then
    # Compute images have no glibc-devel. rustc cannot link here.
    # The Fortran library is what -march=native poisons; smoke the
    # already-linked search binary against the srun libira.
    export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
    export LD_LIBRARY_PATH="${IRA_LIB_DIR}:/opt/ohpc/pub/compiler/gcc/12.4.0/lib64:${LD_LIBRARY_PATH:-}"
    echo "SMOKE_EXISTING $BIN"
    ldd "$BIN"
    "$BIN" 13 200 1 rec
    echo "SMOKE_OK host=$(hostname) job=$SLURM_JOB_ID"
    exit 0
  fi
  bash "$ROOT/scripts/elja_build_lj.sh"
  exit 0
fi
# Startfiles live on the login image only. Stage them before srun.
bash "$ROOT/scripts/elja_stage_sysroot.sh"
exec srun \
  --partition="$PART" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --time="$TIME" \
  --job-name=anneal-ira-build \
  --chdir="$ROOT" \
  bash "$ROOT/scripts/elja_srun_build.sh"
