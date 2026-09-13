#!/usr/bin/env bash
# Compute-node build of lj_cluster_search for the communicating paper arm.
# Does not touch ~/anneal-build.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_comm_build.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
CMAKE_BIN=${CMAKE_BIN:-$HOME/rgpot/.pixi/envs/xtbbld/bin/cmake}
if [[ ! -x $CMAKE_BIN ]]; then
  echo "missing compute-node CMake executable: $CMAKE_BIN" >&2
  exit 1
fi
if [[ ! -e $SYS/usr-include/stdint.h ]]; then
  echo "missing $SYS/usr-include" >&2
  exit 1
fi
mkdir -p "$SYS/bin"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/cc"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/gcc"
ln -sfn "$GCC/bin/g++" "$SYS/bin/g++"
ln -sfn "$CMAKE_BIN" "$SYS/bin/cmake"
ln -sfn /usr/bin/ld "$SYS/bin/ld"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export FC="${GCC}/bin/gfortran"
export CFLAGS="${CFLAGS:-} -isystem $SYS/usr-include"
export CXXFLAGS="${CXXFLAGS:-} -isystem $SYS/usr-include"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-B${SYS} -C link-arg=-B${SYS}/bin -L ${SYS}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
export CARGO_NET_OFFLINE=true
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so" >&2
  exit 1
fi
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
echo "source=$(cat SOURCE_COMMIT 2>/dev/null || echo unknown)"
cargo build --offline --locked --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search
BIN=$CARGO_TARGET_DIR/release/examples/lj_cluster_search
ldd "$BIN"
echo "SMOKE"
"$BIN" 13 200 1 comm
echo "BUILD_OK $BIN"
