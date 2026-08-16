#!/usr/bin/env bash
#SBATCH --job-name=lj75-occ-hops
#SBATCH --partition=long
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=32
#SBATCH --mem=44G
#SBATCH --time=4-00:00:00
#SBATCH --output=/users/home/rog32/ljwork/jcc/lj75-occ-rung/slurm.%j.out
# Occupancy catalog with a hop cap that can reach Marks.
# Serial rec Marks first hits are hops 33847-58779 (sci_rec_n75).
# Ensemble 0002 used CATALOG_MAX_HOPS=1000 and finished on the ico shelf.
# 4527057 pruned extras before the first Hyperband rung.
# Binary: anneal-rung 8ded9aa (verdict waits for a rung; default R=60000).
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-rung
export LJ_BIN=$HOME/anneal-rung/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-rung/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-rung
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-rung/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export GCCLIB=/opt/ohpc/pub/compiler/gcc/12.4.0/lib64
export LD_LIBRARY_PATH=$HOME/ira/lib:$GCCLIB:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export SEED_OFFSET_BASE=700000
mkdir -p "$HOME/ljwork/jcc/lj75-occ-rung"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 0 "$RADIUS"
