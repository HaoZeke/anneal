#!/usr/bin/env bash
#SBATCH --job-name=terra-lj38-occ
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm.%j.out
# LJ38 occupancy on Terra. Paper budget 4e5, hop 60000, 48 replicas.
# WAVE 24 matches the Elja occupancy allocations.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-occ-verify
export LJ_BIN=$HOME/anneal-occ-verify/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-occ-verify/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj38-occ-terra
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-occ-verify/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$HOME/ira/lib:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export SEED_OFFSET_BASE=800000
export LJ_OUT=$HOME/ljwork/terra
mkdir -p "$LJ_OUT/lj38-occ-sb"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
if ! grep -a -F -q different_decaf_family "$LJ_BIN"; then
  echo "refusing: $LJ_BIN missing different_decaf_family" >&2
  exit 2
fi
if ! grep -a -F -q "occupancy leave archive hole" "$LJ_BIN"; then
  echo "refusing: $LJ_BIN missing occupancy leave archive hole" >&2
  exit 2
fi
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj38.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 38 400000 5 "$RADIUS"
