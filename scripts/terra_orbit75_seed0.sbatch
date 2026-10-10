#!/usr/bin/env bash
#SBATCH --job-name=orbit75-seed0
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-orbit75-seed0.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export ORBIT_ROOT=${ORBIT_ROOT:-$HOME/build/anneal-orbit-proof/src}
cd "$ORBIT_ROOT"
exec scripts/terra_orbit75_seed0.sh
