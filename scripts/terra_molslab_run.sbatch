#!/usr/bin/env bash
#SBATCH --job-name=molslab-run
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-molslab-run.%j.out
#SBATCH --error=slurm-molslab-run.%j.err
# Measure only. Binaries, engines, clippy-touched, and rustfmt of the
# touched files are already recorded from the preceding Terra jobs.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/terra_measure_molslab.sh
test -x target/release/examples/molecular_cluster
test -x target/release/examples/slab_adsorption
test -e engines/libxtb_engine.so
test -e engines/librgpot_cuh2.so
rustfmt --edition 2024 --check \
  src/methods/two_phase.rs \
  src/methods/cluster_search.rs \
  examples/common/efficiency.rs \
  examples/molecular_cluster.rs \
  examples/slab_adsorption.rs
echo "=== measure ==="
exec scripts/terra_measure_molslab.sh
