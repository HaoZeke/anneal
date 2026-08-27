#!/usr/bin/env bash
#SBATCH --job-name=lj75-paper-cat
#SBATCH --partition=long
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=32
#SBATCH --mem=44G
#SBATCH --time=4-00:00:00
#SBATCH --output=/users/home/rog32/ljwork/jcc/lj75-paper-cat/slurm.%j.out
# Paper-budget cooperative catalog: 48 x 4e6 recommended, no hop cap.
# Binary: anneal-occ @ 0883b4a (occupancy Hyperband reseed + isomer keep).
# Hypothesis vs lj75-shared-0002 (1000 hops, 0 Marks, 14/48 on ico shelf):
# the miss is the hop cap, not the leftover-SOAP catalog. Serial rec at
# this budget is 4/48 Marks (sci_rec_prod).
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-occ
export LJ_BIN=$HOME/anneal-occ/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-occ/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-paper-cat
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-occ/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export GCCLIB=/opt/ohpc/pub/compiler/gcc/12.4.0/lib64
export LD_LIBRARY_PATH=$HOME/ira/lib:$GCCLIB:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=48
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
unset CATALOG_MAX_HOPS
mkdir -p "$HOME/ljwork/jcc/lj75-paper-cat"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 0 "$RADIUS"
