#!/usr/bin/env bash
#SBATCH --job-name=elja-brains-build
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/users/home/rog32/anneal-occ-brains/slurm-brains-build.%j.out
# Isolated occupancy build. Does not touch anneal-stop or h5/hq.
set -euo pipefail
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
cd "${LJ_ROOT:-$HOME/anneal-occ-brains}"
exec scripts/elja_build_brains.sh
