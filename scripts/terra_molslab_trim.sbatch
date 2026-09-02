#!/usr/bin/env bash
#SBATCH --job-name=molslab-trim
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-molslab-trim.%j.out
#SBATCH --error=slurm-molslab-trim.%j.err
# Heptamer and octamer, plain vs kappa 0.7, four seeds, 2000 charged.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/terra_measure_molslab_trim.sh
test -x target/release/examples/molecular_cluster
test -e engines/libxtb_engine.so
exec scripts/terra_measure_molslab_trim.sh
