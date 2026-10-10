#!/usr/bin/env bash
#SBATCH --job-name=nng-rpc-gate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-nng-rpc-gate.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export ANNEAL_ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-nng-rpc/src}
cd "$ANNEAL_ROOT"
exec scripts/terra_nng_rpc_gate.sh
