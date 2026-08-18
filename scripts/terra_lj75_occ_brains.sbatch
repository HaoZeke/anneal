#!/usr/bin/env bash
#SBATCH --job-name=terra-lj75-brains
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=44G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm.%j.out
# LJ75 occupancy with one talking brain per replica. Paper budget 4e6,
# hop 60000, 48 replicas. Same 5535490 binary as lj38-shared-0012.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-occ-brains
export LJ_BIN=$HOME/anneal-occ-brains/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-occ-brains/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-terra
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-occ-brains/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$HOME/ira/lib:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
# Two-funnel paper hurdle (ico vs Marks). Unset and the floor is still 2;
# set 1 for Good-Turing alone.
export CATALOG_MIN_FAMILIES=2
export SEED_OFFSET_BASE=1600000
export LJ_OUT=$HOME/ljwork/terra
mkdir -p "$LJ_OUT/lj75-occ-brains"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
for symbol in different_decaf_family step_toward_catalog_hole CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats"; do
  if ! grep -a -F -q "$symbol" "$LJ_BIN"; then
    echo "refusing: $LJ_BIN missing $symbol" >&2
    exit 2
  fi
done
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 0 "$RADIUS"
