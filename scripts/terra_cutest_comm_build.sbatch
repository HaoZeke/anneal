#!/usr/bin/env bash
#SBATCH --job-name=cutest-comm-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-cutest-comm-build.%j.out
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export CUTEST_COMM_ROOT=${CUTEST_COMM_ROOT:-$HOME/build/cutest-comm-campaign/anneal}
cd "$CUTEST_COMM_ROOT"
exec scripts/terra_cutest_comm_build.sh
