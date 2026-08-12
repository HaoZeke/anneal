#!/usr/bin/env bash
# Paper water budgets through in-process rgpot GFN2-xTB. Not potserv.
# (H2O)4 at 2500 x 8; (H2O)6 at 4000 x 8. Does not cancel running HQ jobs.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
RGPOT=${RGPOT_ROOT:-$HOME/rgpot}
BASE=${MOL_OUT:-$HOME/ljwork/hq-sci-h2o}
BIN=${MOL_BIN:-$ROOT/target/release/examples/molecular_cluster}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTBLIB=${XTBLIB:-$RGPOT/.pixi/envs/xtbbld/lib}

if [[ ! -x $BIN ]]; then
  echo "missing $BIN; run scripts/elja_build_rgpot_ex.sh login-cargo" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/libxtb_engine.so ]]; then
  echo "missing $ROOT/engines/libxtb_engine.so; run scripts/elja_build_rgpot_ex.sh engines" >&2
  exit 1
fi

submit_arm() {
  local m=$1 budget=$2 seeds=$3
  local out=${BASE}${m}
  local last=$((seeds - 1))
  mkdir -p "$out"
  : >"$out/hq_submit.log"
  echo "xtb eindir (H2O)${m} budget=${budget} seeds=${seeds}" | tee -a "$out/hq_submit.log"
  hq submit \
    --name "h2o${m}-xtb-rec" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$out" \
    --stdout "$out/h2o${m}_rec_%{TASK_ID}.out" \
    --stderr "$out/h2o${m}_rec_%{TASK_ID}.err" \
    -- bash -lc "export SEED_OFFSET=\${HQ_TASK_ID:-0}
export RGPOT_XTB_ENGINE=${ROOT}/engines/libxtb_engine.so
export LD_LIBRARY_PATH=${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:\${LD_LIBRARY_PATH:-}
exec ${BIN} ${m} ${budget} 1" \
    | tee -a "$out/hq_submit.log"
}

submit_arm 2 2500 8
submit_arm 4 2500 8
submit_arm 6 4000 8
echo "submitted water dimer/tetramer/hexamer"
hq job list
