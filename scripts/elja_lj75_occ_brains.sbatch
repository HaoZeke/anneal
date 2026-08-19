#!/usr/bin/env bash
#SBATCH --job-name=occ-lj75-brains
#SBATCH --partition=long
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=44G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm.%j.out
# LJ75 occupancy. Paper budget 4e6, hop 60000, 48 replicas.
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=$HOME/anneal-occ-brains
export LJ_BIN=$HOME/anneal-occ-brains/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$HOME/anneal-occ-brains/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-elja
export JCC_SOURCE_COMMIT_FILE=$HOME/anneal-occ-brains/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$HOME/ira/lib:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export CATALOG_MIN_FAMILIES=2
export CATALOG_SHARED_BIAS=1
export SEED_OFFSET_BASE=2100000
export LJ_OUT=$HOME/ljwork/jcc
mkdir -p "$LJ_OUT/lj75-occ-elja"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
for symbol in different_decaf_family "occupancy leave archive hole" CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats" "occupancy min families"; do
  if ! grep -a -F -q "$symbol" "$LJ_BIN"; then
    echo "refusing: $LJ_BIN missing $symbol" >&2
    exit 2
  fi
done
if ! grep -a -F -q "Refuse" "$LJ_BIN" && ! grep -a -F -q "gt stop packing" "$LJ_BIN"; then
  echo "refusing: $LJ_BIN missing Refuse/gt stop packing" >&2
  exit 2
fi
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 0 "$RADIUS"
