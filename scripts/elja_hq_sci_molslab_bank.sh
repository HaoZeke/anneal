#!/usr/bin/env bash
# Paired bank vs no-bank on water and the FCC Cu–H slab.
# Same binary, same seeds, same budget. BANK_RPC is the only difference.
# First-encounter charged evaluations are the comparison, not finish rate.
#
#   start        — banks only (7402/7404/7406/7412; not LJ 7438/7455/7475/7498)
#   submit-bank  — BANK_RPC clients
#   submit-ctrl  — no-bank control
#   submit       — both arms
#   all          — start then both arms
#
# Does not cancel running HQ jobs.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-build}
RGPOT=${RGPOT_ROOT:-$HOME/rgpot}
MOL=${MOL_BIN:-$ROOT/target/release/examples/molecular_cluster}
SLAB=${SLAB_BIN:-$ROOT/target/release/examples/slab_adsorption}
SRV=${BANK_BIN:-$ROOT/target/release/examples/bank_server}
CON=${SLAB_CON:-$ROOT/examples/fixtures/cuh2_fcc_slab.con}
HOST=${BANK_HOST:-$(hostname)}
CAP=${BANK_CAPACITY:-30}
SLICE=${BANK_SLICE:-200}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
XTBLIB=${XTBLIB:-$RGPOT/.pixi/envs/xtbbld/lib}
BASE=${MOLSLAB_OUT:-$HOME/ljwork/hq-sci}

if [[ ! -x $MOL || ! -x $SLAB ]]; then
  echo "missing examples; run scripts/elja_build_rgpot_ex.sh login-cargo" >&2
  exit 1
fi
if [[ ! -x $SRV ]]; then
  echo "missing $SRV; build with --features bank-rpc" >&2
  exit 1
fi
if [[ ! -e $ROOT/engines/libxtb_engine.so || ! -e $ROOT/engines/librgpot_cuh2.so ]]; then
  echo "missing engines; run scripts/elja_build_rgpot_ex.sh engines" >&2
  exit 1
fi
if [[ ! -f $CON ]]; then
  echo "missing $CON" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"

start_bank() {
  local port=$1 out=$2
  mkdir -p "$out"
  if ! ss -ltn | grep -q ":${port} "; then
    nohup "$SRV" "0.0.0.0:${port}" "$CAP" >"$out/bank_server.log" 2>&1 &
    echo $! >"$out/bank_server.pid"
    sleep 1
  fi
  if ! grep -q "listening" "$out/bank_server.log" && ! grep -q "bank identity" "$out/bank_server.log"; then
    # bank_server prints identity when IRA is linked; either line is a live server
    if ! kill -0 "$(cat "$out/bank_server.pid")" 2>/dev/null; then
      echo "bank_server on $port died" >&2
      cat "$out/bank_server.log" >&2
      exit 1
    fi
  fi
}

submit_mol() {
  local arm=$1 m=$2 budget=$3 seeds=$4 port=$5 target=$6
  local out=${BASE}-h2o${m}-${arm}
  local last=$((seeds - 1))
  local rpc=""
  if [[ $arm == bank ]]; then
    rpc="export BANK_RPC=${HOST}:${port} BANK_SLICE=${SLICE}"
  fi
  mkdir -p "$out"
  : >"$out/hq_submit.log"
  echo "h2o${m} arm=${arm} budget=${budget} seeds=${seeds} target=${target} rpc=${HOST}:${port}" | tee -a "$out/hq_submit.log"
  hq submit \
    --name "h2o${m}-${arm}" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=8h \
    --cwd "$out" \
    --stdout "$out/h2o${m}_${arm}_%{TASK_ID}.out" \
    --stderr "$out/h2o${m}_${arm}_%{TASK_ID}.err" \
    -- bash -lc "export SEED_OFFSET=\${HQ_TASK_ID:-0}
export RGPOT_XTB_ENGINE=${ROOT}/engines/libxtb_engine.so
export TARGET_ENERGY=${target}
export TARGET_TOL=1e-3
export LD_LIBRARY_PATH=${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:\${LD_LIBRARY_PATH:-}
${rpc}
exec ${MOL} ${m} ${budget} 1" \
    | tee -a "$out/hq_submit.log"
}

submit_slab() {
  local arm=$1 budget=$2 seeds=$3 port=$4
  local out=${BASE}-cuh2-${arm}
  local last=$((seeds - 1))
  local rpc=""
  if [[ $arm == bank ]]; then
    rpc="export BANK_RPC=${HOST}:${port} BANK_SLICE=${SLICE}"
  fi
  mkdir -p "$out"
  : >"$out/hq_submit.log"
  echo "cuh2 arm=${arm} budget=${budget} seeds=${seeds} target=-415.971529 rpc=${HOST}:${port}" | tee -a "$out/hq_submit.log"
  hq submit \
    --name "cuh2-${arm}" \
    --array="0-${last}" \
    --cpus 1 \
    --time-limit=2h \
    --cwd "$out" \
    --stdout "$out/cuh2_${arm}_%{TASK_ID}.out" \
    --stderr "$out/cuh2_${arm}_%{TASK_ID}.err" \
    -- bash -lc "export SEED_OFFSET=\${HQ_TASK_ID:-0}
export RGPOT_CUH2_LIBRARY=${ROOT}/engines/librgpot_cuh2.so
export TARGET_ENERGY=-415.971529
export TARGET_TOL=1e-3
export LD_LIBRARY_PATH=${XTBLIB}:${IRA_LIB_DIR}:${GCCLIB}:\${LD_LIBRARY_PATH:-}
${rpc}
exec ${SLAB} ${CON} ${budget} 1" \
    | tee -a "$out/hq_submit.log"
}

start_all() {
  start_bank 7402 "${BASE}-h2o2-bank"
  start_bank 7404 "${BASE}-h2o4-bank"
  start_bank 7406 "${BASE}-h2o6-bank"
  start_bank 7412 "${BASE}-cuh2-bank"
  echo "banks 7402/7404/7406/7412 (water 2/4/6, cuh2). LJ ports untouched."
}

submit_bank() {
  submit_mol bank 2 2500 8 7402 -276.168547
  submit_mol bank 4 2500 8 7404 -553.064301
  submit_mol bank 6 4000 8 7406 -829.846965
  submit_slab bank 2500 8 7412
  echo "submitted bank arms"
}

submit_ctrl() {
  submit_mol ctrl 2 2500 8 7402 -276.168547
  submit_mol ctrl 4 2500 8 7404 -553.064301
  submit_mol ctrl 6 4000 8 7406 -829.846965
  submit_slab ctrl 2500 8 7412
  echo "submitted no-bank control"
}

cmd=${1:-start}
case $cmd in
  start) start_all ;;
  submit-bank) submit_bank; hq job list ;;
  submit-ctrl) submit_ctrl; hq job list ;;
  submit) submit_bank; submit_ctrl; hq job list ;;
  all) start_all; submit_bank; submit_ctrl; hq job list ;;
  *)
    echo "usage: $0 start|submit-bank|submit-ctrl|submit|all" >&2
    exit 2
    ;;
esac
