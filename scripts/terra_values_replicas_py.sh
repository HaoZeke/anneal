#!/usr/bin/env bash
# Build the python extension on terra and probe ChemFit jac=False replicas.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_values_replicas_py.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-values/src}
VERIFY=${ANNEAL_VERIFY:-$HOME/Git/Github/Rust/anneal/.pixi/envs/verify}
CHEMFIT=${CHEMFIT_ROOT:-$HOME/build/chemfit-demo/src}
DEMO=${DEMO_ROOT:-$HOME/build/chemfit-demo}
export PATH="${VERIFY}/bin:${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-values/target}"
export VIRTUAL_ENV="$VERIFY"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "maturin=$(maturin --version)"
mkdir -p "$ROOT/wheels"
maturin build --release --features python --out "$ROOT/wheels"
WHEEL=$(ls -t "$ROOT/wheels"/anneal-*.whl | head -1)
echo "wheel=$WHEEL"
mkdir -p "$DEMO/anneal_pkg"
"$VERIFY/bin/python" -m pip install --force-reinstall --no-deps --target "$DEMO/anneal_pkg" "$WHEEL"
# Overlay current minimize/store Python on top of the wheel.
cp "$ROOT/python/anneal/__init__.py" "$DEMO/anneal_pkg/anneal/__init__.py"
# shellcheck disable=SC1091
source "$DEMO/venv/bin/activate"
export PYTHONPATH="$DEMO/anneal_pkg${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import anneal
print("anneal", anneal.__file__)
print("minimize", hasattr(anneal, "minimize"))
PY
python "$CHEMFIT/scripts/terra_hop_probe.py"
echo VALUES_REPLICAS_PY_OK
