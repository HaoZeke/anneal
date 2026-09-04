#!/usr/bin/env bash
# Occupancy recommended hop from the sealed LJ75 ico well until Marks.
# One replica, local catalog control, Walk on a one-community book.
set -euo pipefail
export PATH="/usr/bin:${HOME}/.cargo/bin:${PATH}"
ROOT="${1:-${HOME}/Git/Github/Rust/anneal}"
cd "$ROOT"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/occ_ico_marks.log"
exec > >(tee -a "${LOG}") 2>&1
echo "commit=$(git rev-parse HEAD)"
echo "pwd=$(pwd)"
cargo build --release --example lj_cluster_search --features bank-rpc
export CATALOG_CAMPAIGN=lj75-ico-marks
export CATALOG_ENSEMBLE=solo
export CATALOG_REPLICA=0
export CATALOG_START_FILE="${ROOT}/tests/fixtures/lj75_ico.xyz"
export CATALOG_START_REPLICA=0
export CATALOG_MAX_HOPS=60000
export CATALOG_SLICE=500
export CATALOG_POPULATION_INTERVAL=50000
./target/release/examples/lj_cluster_search 75 4000000 8 rec,catalog
echo "EXIT:$?"
