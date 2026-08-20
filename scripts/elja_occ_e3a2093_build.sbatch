#!/usr/bin/env bash
#SBATCH --job-name=anneal-occ-e3-build
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/users/home/rog32/anneal-occ-e3a2093/slurm-build.%j.out
# Compute-node build of occupancy e3a2093. Does not touch anneal-stop.
set -euo pipefail
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-e3a2093}
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/elja_build_lj.sh
exec scripts/elja_build_lj.sh
