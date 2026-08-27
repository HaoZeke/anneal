#!/usr/bin/env bash
#SBATCH --job-name=lj75-occ-marks
#SBATCH --partition=long
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=/users/home/rog32/ljwork/jcc/lj75-occ-dwell/slurm.%j.out
# LJ75 occupancy on packing-gt-stop 9dde60d. Paper budget 4e6,
# 48 replicas, WAVE 24 on one 32-core node. Extras Walk while
# leftover SOAP is unsaturated. Isolated from anneal-stop.
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-9dde60d}
export LJ_BIN=$LJ_ROOT/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$LJ_ROOT/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-dwell
export JCC_SOURCE_COMMIT_FILE=$LJ_ROOT/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export GCCLIB=/opt/ohpc/pub/compiler/gcc/12.4.0/lib64
export LD_LIBRARY_PATH=$HOME/ira/lib:$GCCLIB:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=48
export CATALOG_WAVE=24
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
export CATALOG_MAX_HOPS=60000
export SEED_OFFSET_BASE=9300000
export LJ_OUT=$HOME/ljwork/jcc
mkdir -p "$HOME/ljwork/jcc/lj75-occ-dwell"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
for symbol in different_packing_family "occupancy leave archive hole" "leftover-SOAP TIS seats" "occupancy min families" "gt stop packing"; do
  if ! grep -a -F -q "$symbol" "$LJ_BIN"; then
    echo "refusing: $LJ_BIN missing $symbol" >&2
    exit 2
  fi
done
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 21 "$RADIUS"
