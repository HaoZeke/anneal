#!/usr/bin/env bash
#SBATCH --job-name=terra-leave-ridge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --output=probe/ridge.%j.out
# Does a packing-map Leave from the sealed LJ75 ico well install Marks?
# Arms: toward Marks mu, fivefold residual, covering ridge.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
export PATH="/usr/bin:$HOME/.cargo/bin:$PATH"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH=$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}
ROOT=${LJ_ROOT:-$HOME/anneal-leave-ridge}
cd "$ROOT"
mkdir -p probe
git rev-parse HEAD >SOURCE_COMMIT
echo "{\"kind\":\"build\",\"head\":\"$(git rev-parse HEAD)\",\"host\":\"$(hostname)\"}"
cargo build --release --features featomic --example leave_packing_probe
exec ./target/release/examples/leave_packing_probe \
  "${LEAVE_PROBE_N:-0}" \
  "${LEAVE_PROBE_STEPS:-400}" \
  ridge
