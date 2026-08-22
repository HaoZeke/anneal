#!/usr/bin/env bash
#SBATCH --job-name=leave-ridge
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=probe/ridge.%j.out
# Directed packing-map Leave from the sealed LJ75 ico well: toward Marks
# mu, fivefold residual, then (if LEAVE_PROBE_N>0) covering ridges.
set -euo pipefail
export PATH="$HOME/.pixi/bin:/usr/bin:$HOME/.cargo/bin:$PATH"
export PIXI_CACHE_DIR=${PIXI_CACHE_DIR:-$HOME/.cache/pixi}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ROOT=${LJ_ROOT:-$HOME/anneal-leave-ridge}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
cd "$ROOT"
mkdir -p probe
git rev-parse HEAD >SOURCE_COMMIT
if [[ ! -x $PIXI ]]; then
  echo "missing pixi at $PIXI" >&2
  exit 1
fi
if [[ ! -d .pixi/envs/cluster ]]; then
  echo "missing .pixi/envs/cluster; pixi install -e cluster on the login image" >&2
  exit 1
fi
echo "{\"kind\":\"build\",\"head\":\"$(git rev-parse HEAD)\",\"host\":\"$(hostname)\",\"job\":\"${SLURM_JOB_ID:-}\"}"
"$PIXI" run -e cluster cargo test --offline --release --test occupancy_leave_contract leave_quench_keeps_the_walk
"$PIXI" run -e cluster cargo build --offline --release --features featomic --example leave_packing_probe
exec "$PIXI" run -e cluster -- ./target/release/examples/leave_packing_probe \
  "${LEAVE_PROBE_N:-0}" \
  "${LEAVE_PROBE_STEPS:-400}" \
  ridge
