#!/usr/bin/env bash
#SBATCH --job-name=anneal-occ-marks-build
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/users/home/rog32/anneal-occ-7fe8d5a/slurm-build.%j.out
# Build lj_cluster_search + catalog_server at 7fe8d5a. Isolated tree.
set -euo pipefail
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-7fe8d5a}
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/elja_build_lj.sh
exec scripts/elja_build_lj.sh
