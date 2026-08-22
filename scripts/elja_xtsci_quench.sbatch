#!/usr/bin/env bash
#SBATCH --job-name=xtsci-quench
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=probe/xtsci-quench.%j.out
set -euo pipefail
export PATH="$HOME/.pixi/bin:/usr/bin:$HOME/.cargo/bin:$PATH"
export PIXI_CACHE_DIR=${PIXI_CACHE_DIR:-$HOME/.cache/pixi}
ROOT=${LJ_ROOT:-$HOME/anneal-occ-a6fc882}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
cd "$ROOT"
mkdir -p probe
echo "{\"kind\":\"build\",\"head\":\"$(git rev-parse HEAD 2>/dev/null || cat SOURCE_COMMIT)\",\"job\":\"${SLURM_JOB_ID:-}\"}"
"$PIXI" run -e cluster cargo build --offline --release --example xtsci_quench_compare
exec "$PIXI" run -e cluster -- ./target/release/examples/xtsci_quench_compare "${RELAX_STEPS:-200}"
