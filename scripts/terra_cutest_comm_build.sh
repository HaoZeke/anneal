#!/usr/bin/env bash
# Build the communicating-chain CUTEst driver on terra and bootstrap CUTEst.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_cutest_comm_build.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${CUTEST_COMM_ROOT:-$HOME/build/cutest-comm-campaign/anneal}
VERIFY=${ANNEAL_VERIFY:-$HOME/Git/Github/Rust/anneal/.pixi/envs/verify}
export PATH="${VERIFY}/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${ROOT}/target"
cd "$ROOT"
if [[ -d .git ]]; then
  git rev-parse HEAD >SOURCE_COMMIT
else
  echo "unpinned-rsync" >SOURCE_COMMIT
fi
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
echo "python=$(python --version)"
echo "maturin=$(maturin --version)"
# --locked needs Cargo.lock in the rsync; the filter crate tests stay on terra.
cargo test --lib methods::cutest_ensemble -- --nocapture
if [[ ! -x $ROOT/.bench/SIFDecode/install/bin/sifdecoder ]]; then
  bash experiments/benchmarks/bootstrap_cutest.sh "$ROOT"
fi
export VIRTUAL_ENV="$VERIFY"
mkdir -p "$ROOT/wheels"
maturin build --release --features python --out "$ROOT/wheels"
"$VERIFY/bin/python" -m pip install --force-reinstall --no-deps "$ROOT/wheels"/anneal-*.whl
"$VERIFY/bin/python" - <<'PY'
import anneal
print("anneal", anneal.__version__)
assert hasattr(anneal, "ensemble_optimize"), "ensemble_optimize missing"
print("ensemble_optimize ok")
PY
# Native Sphere does not take the Python GIL. This is the path that hung
# the census: replica threads calling a Python objective.
"$VERIFY/bin/python" -m pytest -q pytest/test_ensemble_optimize.py
echo "BUILD_OK $ROOT"
