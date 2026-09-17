#!/usr/bin/env bash
#SBATCH --job-name=values-repl-py
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-values-repl-py.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export ANNEAL_ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-values/src}
cd "$ANNEAL_ROOT"
exec scripts/terra_values_replicas_py.sh
