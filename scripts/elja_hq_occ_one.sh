#!/usr/bin/env bash
# One HyperQueue task: one occupancy ensemble, every replica live.
set -euo pipefail
N=${1:?n}
BUDGET=${2:?budget}
ENSEMBLE=${3:?ensemble index}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
export LJ_BIN=${LJ_BIN:-$LJ_ROOT/target/release/examples/lj_cluster_search}
export CATALOG_SERVER_BIN=${CATALOG_SERVER_BIN:-$LJ_ROOT/target/release/examples/catalog_server}
export JCC_SOURCE_COMMIT_FILE=${JCC_SOURCE_COMMIT_FILE:-$LJ_ROOT/SOURCE_COMMIT}
export ANNEAL_REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
export CATALOG_REPLICAS=${CATALOG_REPLICAS:-48}
export CATALOG_WAVE=${CATALOG_WAVE:-48}
export CATALOG_SLICE=${CATALOG_SLICE:-500}
export CATALOG_POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
export CATALOG_MAX_HOPS=${CATALOG_MAX_HOPS:-60000}
export CATALOG_MIN_FAMILIES=${CATALOG_MIN_FAMILIES:-2}
export CATALOG_SHARED_BIAS=${CATALOG_SHARED_BIAS:-1}
export CATALOG_CAMPAIGN=${CATALOG_CAMPAIGN:-lj${N}-occ-hq}
export LJ_OUT=${LJ_OUT:-$HOME/ljwork/hq-occ}
export SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-$((2000000 + N * 10000))}
CALIBRATION=$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj${N}.json
RADIUS=$(python3 -c "import json; print(json.load(open(\"$CALIBRATION\"))[\"census_radius\"])")
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
exec "$LJ_ROOT/scripts/elja_jcc_lj_many_chains.sh" "$N" "$BUDGET" "$ENSEMBLE" "$RADIUS"
