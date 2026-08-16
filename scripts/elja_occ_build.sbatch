#!/usr/bin/env bash
#SBATCH --job-name=anneal-occ-build
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/users/home/rog32/anneal-occ/slurm-build.%j.out
# Compute-node build of lj_cluster_search + catalog_server at 0883b4a.
# Does not touch anneal-fix-epoch or anneal-build (live jobs).
set -euo pipefail
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ}
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/elja_build_lj.sh
exec scripts/elja_build_lj.sh
