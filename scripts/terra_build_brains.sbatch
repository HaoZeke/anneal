#!/usr/bin/env bash
#SBATCH --job-name=terra-brains-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-brains-build.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
cd "${LJ_ROOT:-$HOME/anneal-occ-brains}"
exec scripts/terra_build_brains.sh
