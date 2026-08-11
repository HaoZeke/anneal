#!/usr/bin/env bash
# Paper CuH2 slab through in-process rgpot. Not potserv.
# cuh2_tiny at 2500 x 8 (tbl:soap-arm). Does not cancel running HQ jobs.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
BASE=${SLAB_OUT:-$HOME/ljwork/hq-sci-slab}
BIN=${SLAB_BIN:-$ROOT/target/release/examples/slab_adsorption}
CON=${SLAB_CON:-$ROOT/examples/fixtures/cuh2_tiny.con}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}

if [[ ! -x $BIN ]]; then
  echo "missing $BIN; run scripts/elja_build_rgpot_ex.sh login-cargo" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/librgpot_cuh2.so ]]; then
  echo "missing $ROOT/engines/librgpot_cuh2.so; run scripts/elja_build_rgpot_ex.sh engines" >&2
  exit 1
fi
if [[ ! -f $CON ]]; then
  echo "missing $CON" >&2
  exit 1
fi

mkdir -p "$BASE"
: >"$BASE/hq_submit.log"
echo "cuh2 eindir $CON budget=2500 seeds=8" | tee -a "$BASE/hq_submit.log"
hq submit \
  --name "cuh2-tiny-rec" \
  --array=0-7 \
  --cpus 1 \
  --time-limit=2h \
  --cwd "$BASE" \
  --stdout "$BASE/cuh2_tiny_rec_%{TASK_ID}.out" \
  --stderr "$BASE/cuh2_tiny_rec_%{TASK_ID}.err" \
  -- bash -lc "export SEED_OFFSET=\${HQ_TASK_ID:-0}
export RGPOT_CUH2_LIBRARY=${ROOT}/engines/librgpot_cuh2.so
export LD_LIBRARY_PATH=${IRA_LIB_DIR}:${GCCLIB}:\${LD_LIBRARY_PATH:-}
exec ${BIN} ${CON} 2500 1" \
  | tee -a "$BASE/hq_submit.log"
echo "submitted cuh2_tiny 2500x8"
hq job list
