#!/usr/bin/env bash
#SBATCH --job-name=rgmin-quench
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=probe/rgmin-quench.%j.out
# L-BFGS (current hopping quench) vs FIRE 2.0 on sealed LJ75 starts.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
ROOT=${LJ_ROOT:-$HOME/anneal-leave-ridge}
cd "$ROOT"
mkdir -p probe
echo "{\"kind\":\"build\",\"head\":\"$(git rev-parse HEAD)\",\"host\":\"$(hostname)\"}"
cargo build --release --example rgmin_quench_compare
exec ./target/release/examples/rgmin_quench_compare "${RELAX_STEPS:-200}"
