#!/usr/bin/env bash
#SBATCH --job-name=terra-occ-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-build.%j.out
# Rebuild IRA without MKL, then the occupancy LJ binary on Terra.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/anneal-occ-verify}
export IRA_ROOT=${IRA_ROOT:-$HOME/ira}
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/terra_rebuild_ira.sh
test -x scripts/terra_build_lj.sh
scripts/terra_rebuild_ira.sh
exec scripts/terra_build_lj.sh
