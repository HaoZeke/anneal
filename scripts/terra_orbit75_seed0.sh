#!/usr/bin/env bash
# Live proof: one LJ75 seed, paper 4e6, thompson,rscreen,orbit.
# Same argv the o75-orbit dumps used. No invented tolerances.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_orbit75_seed0.sh: run under sbatch" >&2
  exit 1
fi
ROOT=${ORBIT_ROOT:-$HOME/build/anneal-orbit-proof/src}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-orbit-proof/target}"
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so" >&2
  exit 1
fi
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
export HISTORY=${HISTORY:-none}
export HISTORY_REPLICAS=${HISTORY_REPLICAS:-1}
export HISTORY_POLICY=${HISTORY_POLICY:-accepted}
export HISTORY_CHECKPOINT=${HISTORY_CHECKPOINT:-1000}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
cargo build --release --features featomic,ira,bank-rpc --example lj_cluster_search
BIN=$CARGO_TARGET_DIR/release/examples/lj_cluster_search
ldd "$BIN" | head
export SEED_OFFSET=${SEED_OFFSET:-0}
echo "RUN $BIN 75 4000000 1 thompson,rscreen,orbit SEED_OFFSET=$SEED_OFFSET"
"$BIN" 75 4000000 1 thompson,rscreen,orbit | tee "$CARGO_TARGET_DIR/orbit75_seed${SEED_OFFSET}.out"
if grep -q 'verified -397.492331' "$CARGO_TARGET_DIR/orbit75_seed${SEED_OFFSET}.out" \
  && grep -q SOLVED "$CARGO_TARGET_DIR/orbit75_seed${SEED_OFFSET}.out"; then
  echo ORBIT75_LIVE_OK
else
  echo ORBIT75_LIVE_FAIL
  exit 1
fi
