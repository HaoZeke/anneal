#!/usr/bin/env bash
# Gate and H6 measurement inside one Slurm allocation.
# Invoked as: srun -p cpu -c 16 --mem=24G -t 03:00:00 scripts/terra_slab_h6_srun.sh
set -euo pipefail
export SLURM_CONF=/etc/slurm-llnl/slurm.conf
ROOT=${LJ_ROOT:-$HOME/Git/Github/Rust/anneal-slab}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/anneal-target-slab}
export PATH="$HOME/.cargo/bin:/usr/bin:$PATH"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-16}
WATER=${WATER_ROOT:-$HOME/Git/Github/Rust/anneal-wt-water}
export RGPOT_CUH2_LIBRARY=${RGPOT_CUH2_LIBRARY:-$WATER/engines/librgpot_cuh2.so}
export POTENTIAL_LIBRARY=${POTENTIAL_LIBRARY:-$RGPOT_CUH2_LIBRARY}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_slab_h6_srun.sh requires a Slurm allocation" >&2
  exit 1
fi

cd "$ROOT"
git checkout -f feat/slab-multisite
mkdir -p "$ROOT/engines" "$CARGO_TARGET_DIR" "$ROOT/results/slab-h6"
if [[ -e $RGPOT_CUH2_LIBRARY && ! -e $ROOT/engines/librgpot_cuh2.so ]]; then
  ln -sfn "$RGPOT_CUH2_LIBRARY" "$ROOT/engines/librgpot_cuh2.so"
fi
export LD_LIBRARY_PATH="$IRA_LIB_DIR:${LD_LIBRARY_PATH:-}"

echo "=== cargo test --lib --examples ==="
cargo test --lib --examples 2>&1 | tee "$CARGO_TARGET_DIR/gate-test.log"
echo "=== cargo build --release --features rgpot-ex --examples ==="
cargo build --release --features rgpot-ex --examples 2>&1 | tee "$CARGO_TARGET_DIR/gate-build.log"

echo "=== cargo build --release --features rgpot-ex,featomic,bank-rpc --example slab_adsorption --example slab_random_relax ==="
cargo build --release --features rgpot-ex,featomic,bank-rpc \
  --example slab_adsorption --example slab_random_relax \
  2>&1 | tee "$CARGO_TARGET_DIR/measure-build.log"

export LJ_ROOT=$ROOT
export MOLSLAB_OUT=$ROOT/results/slab-h6
export SLAB_CON=$ROOT/examples/fixtures/cuh2_fcc_slab_h6.con
scripts/terra_measure_slab_h6.sh
echo TERRA_SLAB_H6_SRUN_OK
