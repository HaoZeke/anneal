#!/usr/bin/env bash
#SBATCH --job-name=molslab-meas
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-molslab-meas.%j.out
#SBATCH --error=slurm-molslab-meas.%j.err
# Clippy the touched files, fmt-check them, then run the molslab measure.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
cd "$LJ_ROOT"
test -x scripts/terra_clippy_touched.sh
test -x scripts/terra_measure_molslab.sh
if [[ ! -x target/release/examples/molecular_cluster || ! -x target/release/examples/slab_adsorption ]]; then
  test -x scripts/terra_build_molslab_dev.sh
  scripts/terra_build_molslab_dev.sh
fi
echo "=== cargo clippy -D warnings (touched lib files) ==="
scripts/terra_clippy_touched.sh --features rgpot-ex,ira --lib -- \
  src/methods/two_phase.rs \
  src/methods/cluster_search.rs
echo "=== cargo clippy -D warnings (touched examples) ==="
scripts/terra_clippy_touched.sh --features rgpot-ex,ira,featomic,bank-rpc \
  --example molecular_cluster \
  --example slab_adsorption \
  --example water_ride_search -- \
  examples/common/efficiency.rs \
  examples/molecular_cluster.rs \
  examples/slab_adsorption.rs
echo "=== cargo fmt --check (touched) ==="
cargo fmt --check -- \
  src/methods/two_phase.rs \
  src/methods/cluster_search.rs \
  examples/common/efficiency.rs \
  examples/molecular_cluster.rs \
  examples/slab_adsorption.rs
echo "=== measure ==="
exec scripts/terra_measure_molslab.sh
