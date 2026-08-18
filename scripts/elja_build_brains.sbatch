#!/usr/bin/env bash
#SBATCH --job-name=occ-brains-build
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-brains-build.%j.out
set -euo pipefail
export PATH="$HOME/.pixi/bin:/usr/bin:$HOME/.cargo/bin:$PATH"
cd "${LJ_ROOT:-$HOME/anneal-occ-brains}"
exec scripts/elja_build_brains.sh
