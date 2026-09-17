#!/usr/bin/env bash
#SBATCH --job-name=recognition-gate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-recognition-gate.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export ANNEAL_ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-recognition/src}
cd "$ANNEAL_ROOT"
exec scripts/terra_tm8k_gate.sh
