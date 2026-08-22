#!/usr/bin/env bash
#SBATCH --job-name=terra-leave-check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-leave-check.%j.out
# Short occupancy run that reads the Leave contract off a live book.
# LEAVE_CHECK_N picks the cluster size; 75 is the Marks case, 38 the Oh one.
#
# Not the paper budget. What it answers: does a Leave install a packing, does
# the sparsified book count packings rather than cells, and does packing_sat
# stay false while the cells still carry singletons.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
export LJ_BIN=$LJ_ROOT/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$LJ_ROOT/target/release/examples/catalog_server
N=${LEAVE_CHECK_N:-75}
export CATALOG_CAMPAIGN=${CATALOG_CAMPAIGN:-lj${N}-leave-check}
export JCC_SOURCE_COMMIT_FILE=$LJ_ROOT/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}
export CATALOG_REPLICAS=${CATALOG_REPLICAS:-8}
export CATALOG_WAVE=${CATALOG_WAVE:-8}
export CATALOG_SLICE=${CATALOG_SLICE:-500}
export CATALOG_POPULATION_INTERVAL=${CATALOG_POPULATION_INTERVAL:-50000}
export CATALOG_MAX_HOPS=${CATALOG_MAX_HOPS:-4000}
export CATALOG_SHARED_BIAS=1
export SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-1900000}
export LJ_OUT=$HOME/ljwork/terra
BUDGET=${LEAVE_CHECK_BUDGET:-300000}
INDEX=${LEAVE_CHECK_INDEX:-1}
cd "$LJ_ROOT"
git rev-parse HEAD >SOURCE_COMMIT
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server
RADIUS=$(python3 -c "import json; print(json.load(open('$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj$N.json'))['census_radius'])")
scripts/elja_jcc_lj_many_chains.sh "$N" "$BUDGET" "$INDEX" "$RADIUS"
