#!/usr/bin/env bash
#SBATCH --job-name=lj38-occ-hops
#SBATCH --partition=long
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=32
#SBATCH --mem=44G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/home/rog32/ljwork/jcc/lj38-occ-sb/slurm.%j.out
# LJ38 occupancy on the unified leave stack. Paper budget 4e5.
# Binary: anneal-accel 16e71a9.
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-accel
export LJ_BIN=$HOME/anneal-accel/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-accel/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj38-occ-sb
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-accel/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export GCCLIB=/opt/ohpc/pub/compiler/gcc/12.4.0/lib64
export LD_LIBRARY_PATH=$HOME/ira/lib:$GCCLIB:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export SEED_OFFSET_BASE=800000
mkdir -p "$HOME/ljwork/jcc/lj38-occ-sb"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj38.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 38 400000 0 "$RADIUS"
