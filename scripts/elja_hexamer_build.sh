#!/usr/bin/env bash
# Build molecular_cluster with the communicating kernel. Reuses the
# already-staged xtb engine; does not touch ~/anneal-build.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_hexamer_build.sh: run under srun/sbatch, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
CMAKE_BIN=${CMAKE_BIN:-$HOME/rgpot/.pixi/envs/xtbbld/bin/cmake}
XTB_ENGINE=${XTB_ENGINE:-$HOME/anneal-build/engines/libxtb_engine.so}
mkdir -p "$SYS/bin" "$ROOT/engines"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/cc"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/gcc"
ln -sfn "$GCC/bin/g++" "$SYS/bin/g++"
[[ -x $CMAKE_BIN ]] && ln -sfn "$CMAKE_BIN" "$SYS/bin/cmake"
ln -sfn /usr/bin/ld "$SYS/bin/ld"
cp -f "$XTB_ENGINE" "$ROOT/engines/libxtb_engine.so"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export CFLAGS="${CFLAGS:-} -isystem $SYS/usr-include"
export CXXFLAGS="${CXXFLAGS:-} -isystem $SYS/usr-include"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-B${SYS} -C link-arg=-B${SYS}/bin -L ${SYS}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
export CARGO_NET_OFFLINE=true
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
cargo build --offline --locked --release --features rgpot-ex,featomic,bank-rpc \
  --example molecular_cluster
BIN=$CARGO_TARGET_DIR/release/examples/molecular_cluster
export RGPOT_XTB_ENGINE=$ROOT/engines/libxtb_engine.so
export LD_LIBRARY_PATH="${HOME}/rgpot/.pixi/envs/xtbbld/lib:${HOME}/ira/lib:${GCC}/lib64:${LD_LIBRARY_PATH:-}"
"$BIN" 6 80 1 comm
echo "BUILD_OK $BIN"
