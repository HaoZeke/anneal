#!/usr/bin/env bash
#SBATCH --job-name=box-unify
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-box-unify.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export ANNEAL_ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-box/src}
cd "$ANNEAL_ROOT"
exec scripts/terra_du5b_unify.sh
