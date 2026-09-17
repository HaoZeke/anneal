#!/usr/bin/env bash
# Rebuild the python extension and run the ASE / cluster_search tests.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_ase_gate.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/Git/Github/Rust/anneal}
PIXI=${PIXI_BIN:-$HOME/.pixi/bin/pixi}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-ase/target}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm-llnl/slurm.conf}"
# Isolate CARGO_HOME so ~/.cargo/config.toml mold flags do not hit pixi gcc.
CARGO_HOME="${CARGO_HOME:-$HOME/build/anneal-ase/cargo}"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"
if [[ ! -e $CARGO_HOME/registry && -d $HOME/.cargo/registry ]]; then
  ln -sfn "$HOME/.cargo/registry" "$CARGO_HOME/registry"
fi
if [[ ! -e $CARGO_HOME/git && -d $HOME/.cargo/git ]]; then
  ln -sfn "$HOME/.cargo/git" "$CARGO_HOME/git"
fi
cat >"$CARGO_HOME/config.toml" <<'EOF'
[build]
jobs = 4
[net]
git-fetch-with-cli = true
EOF
export CARGO_HOME
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(git rev-parse HEAD 2>/dev/null || echo unpinned)"
VERIFY="${ANNEAL_VERIFY:-$ROOT/.pixi/envs/verify}"
if [[ ! -x $VERIFY/bin/maturin || ! -x $VERIFY/bin/pytest ]]; then
  echo "missing verify env at $VERIFY" >&2
  exit 1
fi
export PATH="$VERIFY/bin:$PATH"
export VIRTUAL_ENV="$VERIFY"
"$VERIFY/bin/maturin" develop --features python
"$VERIFY/bin/pytest" \
  pytest/test_ase_optimizer.py \
  pytest/test_cluster_search.py::test_cluster_search_start_kwarg \
  pytest/test_cluster_search.py::test_cluster_search_rejects_bad_n_and_budget \
  -q --tb=short
echo ASE_GATE_OK
