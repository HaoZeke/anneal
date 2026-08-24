#!/usr/bin/env bash
#SBATCH --job-name=occ-lj75-c48f
#SBATCH --partition=64cpu_256mem
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm.%j.out
# LJ75, 48 chains, the full ladder arm: recognition refund plus the
# ensemble frontier exchange (campaign lj75-occ-c48f). The campaign
# TOML is the contract; the binary refuses to start if these exports
# drift from it.
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=${LJ_ROOT:?set LJ_ROOT to the staged tree}
export LJ_BIN=$LJ_ROOT/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$LJ_ROOT/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-c48f
export JCC_SOURCE_COMMIT_FILE=$LJ_ROOT/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$HOME/ira/lib:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=48
export CATALOG_SLICE=500
export CATALOG_SHARED_SCREEN=1
export CATALOG_SEAM_LADDER=1
export CATALOG_FRONTIER_EXCHANGE=1
export CATALOG_CONFIG=$LJ_ROOT/scripts/campaigns/lj75-occ-c48f.toml
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export SEED_OFFSET_BASE=9950000
export LJ_OUT=$HOME/ljwork/c48f
mkdir -p "$LJ_OUT/$CATALOG_CAMPAIGN"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 20 "$RADIUS"
