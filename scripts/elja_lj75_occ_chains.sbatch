#!/usr/bin/env bash
#SBATCH --job-name=occ-lj75-chains
#SBATCH --partition=64cpu_256mem
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm.%j.out
# LJ75 occupancy at a chosen chain count, so escape rate can be read
# against the ensemble size rather than against wall clock.
#
# The claim under test is that chains interact during minimisation: each
# quench is repelled from the packings the others hold, so a wider
# ensemble leaves the icosahedral funnel sooner. That is a statement
# about the reference cloud, which is why the run prints its size.
#
#   CHAINS=48 sbatch scripts/elja_lj75_occ_chains.sbatch
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
CHAINS=${CHAINS:-48}
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-013650d}
export LJ_BIN=$LJ_ROOT/target/release/examples/lj_cluster_search
export CATALOG_SERVER_BIN=$LJ_ROOT/target/release/examples/catalog_server
export CATALOG_CAMPAIGN=lj75-occ-c$CHAINS
export JCC_SOURCE_COMMIT_FILE=$LJ_ROOT/SOURCE_COMMIT
export ANNEAL_REPRO_ROOT=$HOME/anneal_repro
export IRA_LIB_DIR=$HOME/ira/lib
export LD_LIBRARY_PATH=$HOME/ira/lib:${LD_LIBRARY_PATH:-}
# The campaign contract: everything the run's physics needs --
# replicas, wave, slice, hop cap, census interval, every coordination
# channel, the seed base -- lives in the TOML below and nowhere else.
# This script used to also export CATALOG_REPLICAS/WAVE/SLICE/
# SHARED_SCREEN/POPULATION_INTERVAL/MAX_HOPS/SEED_OFFSET_BASE itself,
# duplicating the file; one of those duplicates silently drifted
# (a hardcoded seed-base formula here, a different value in the file)
# and only the campaign's own collision guard catching the mismatch
# stopped it from running dark a second time. The guard should not
# need to fire because a second source of truth should not exist.
export CATALOG_CONFIG=$LJ_ROOT/scripts/campaigns/lj75-occ-c48.toml
export LJ_OUT=$HOME/ljwork/c$CHAINS
mkdir -p "$LJ_OUT/$CATALOG_CAMPAIGN"
cd "$LJ_ROOT"
test -s "$JCC_SOURCE_COMMIT_FILE"
test -x "$LJ_BIN"
test -x "$CATALOG_SERVER_BIN"
RADIUS=$(python3 -c "import json; print(json.load(open(\"$ANNEAL_REPRO_ROOT/results_jcc/calibration/lj75.json\"))[\"census_radius\"])")
exec scripts/elja_jcc_lj_many_chains.sh 75 4000000 20 "$RADIUS"
