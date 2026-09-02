#!/usr/bin/env bash
#SBATCH --job-name=molslab-dev
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-molslab-dev.%j.out
#SBATCH --error=slurm-molslab-dev.%j.err
# Build, test, clippy, fmt, then measure water and CuH2 on Terra.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export LJ_ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
cd "$LJ_ROOT"
test -x scripts/terra_build_molslab_dev.sh
test -x scripts/terra_measure_molslab.sh
scripts/terra_build_molslab_dev.sh
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "=== cargo test --features rgpot-ex,ira --lib ==="
cargo test --features rgpot-ex,ira --lib
echo "=== cargo clippy -D warnings (lib) ==="
cargo clippy --features rgpot-ex,ira --lib -- -D warnings
echo "=== cargo clippy -D warnings (examples) ==="
cargo clippy --features rgpot-ex,ira,featomic,bank-rpc \
  --example molecular_cluster \
  --example slab_adsorption \
  --example water_ride_search \
  -- -D warnings
echo "=== cargo fmt --check ==="
cargo fmt --check -- \
  src/methods/two_phase.rs \
  src/methods/cluster_search.rs \
  examples/common/efficiency.rs \
  examples/molecular_cluster.rs \
  examples/slab_adsorption.rs
echo "=== measure ==="
exec scripts/terra_measure_molslab.sh
