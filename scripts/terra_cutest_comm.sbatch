#!/usr/bin/env bash
#SBATCH --job-name=cutest-comm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%4
#SBATCH --output=slurm-cutest-comm-%A_%a.out
# Communicating-chain CUTEst census: 209 problems x 3 seeds x 8000 evals.
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
ROOT=${CUTEST_COMM_ROOT:-$HOME/build/cutest-comm-campaign}
ANNEAL=${ROOT}/anneal
VERIFY=${ANNEAL_VERIFY:-$HOME/Git/Github/Rust/anneal/.pixi/envs/verify}
MANIFEST=${ROOT}/toms_cutest_census.csv
METHODS=${METHODS:-comm_hop}
OUTDIR=${ROOT}/shards/${METHODS}
export PATH="${VERIFY}/bin:/usr/bin:${PATH}"
export PYTHONPATH="${ANNEAL}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTDIR"
cd "$ANNEAL"
test -f "$MANIFEST"
test -x "$VERIFY/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} shard=${SLURM_ARRAY_TASK_ID:-} methods=$METHODS"
echo "python=$VERIFY/bin/python"
PY=$VERIFY/bin/python
"$PY" experiments/scripts/sota_cutest.py \
  --out "$OUTDIR/sota_shard_${SLURM_ARRAY_TASK_ID}.csv" \
  --seeds 3 \
  --dim-cap 30 \
  --max-problems 209 \
  --budget 8000 \
  --bench-root "$ANNEAL" \
  --pycutest-cache "$ROOT/cache-${SLURM_ARRAY_TASK_ID}" \
  --problem-manifest "$MANIFEST" \
  --methods "$METHODS" \
  --shard-count 16 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
