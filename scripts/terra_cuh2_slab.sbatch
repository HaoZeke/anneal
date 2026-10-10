#!/usr/bin/env bash
#SBATCH --job-name=cuh2-slab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-cuh2-slab.%j.out
#SBATCH --error=slurm-cuh2-slab.%j.err
# Four-seed CuH2 fcc slab baseline, plain surface, budget 2500.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
cd "$LJ_ROOT"
test -x scripts/terra_measure_cuh2.sh
test -x target/release/examples/slab_adsorption
test -e engines/librgpot_cuh2.so
test -f examples/fixtures/cuh2_fcc_slab.con
exec scripts/terra_measure_cuh2.sh
