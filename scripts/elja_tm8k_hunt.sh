#!/usr/bin/env bash
# Communicating hunt unit tests on an Elja short-partition node.
# Does not touch ~/anneal-build. Staging tree is ~/build/anneal-hunt.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_tm8k_hunt.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${ANNEAL_ROOT:-$HOME/build/anneal-hunt}
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
CMAKE_BIN=${CMAKE_BIN:-$HOME/rgpot/.pixi/envs/xtbbld/bin/cmake}
if [[ ! -e $SYS/usr-include/stdint.h ]]; then
  echo "missing $SYS/usr-include; stage login headers first" >&2
  exit 1
fi
mkdir -p "$SYS/bin"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/cc"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/gcc"
ln -sfn "$GCC/bin/g++" "$SYS/bin/g++"
if [[ -x $CMAKE_BIN ]]; then
  ln -sfn "$CMAKE_BIN" "$SYS/bin/cmake"
fi
ln -sfn /usr/bin/ld "$SYS/bin/ld"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export CFLAGS="${CFLAGS:-} -isystem $SYS/usr-include"
export CXXFLAGS="${CXXFLAGS:-} -isystem $SYS/usr-include"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-B${SYS} -C link-arg=-B${SYS}/bin -L ${SYS}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/anneal-hunt/target}"
export CARGO_NET_OFFLINE=true
mkdir -p "$CARGO_TARGET_DIR"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | awk 'NR == 1 { print }')"
if [[ -f SOURCE_COMMIT ]]; then
  echo "source=$(cat SOURCE_COMMIT)"
fi
if ! grep -n 'fn communicating' src/methods/cluster_hopping/config.rs; then
  echo "COMMUNICATING_PRESET_MISSING" >&2
  exit 1
fi
if ! grep -n 'jump_on_stall = true' src/methods/cluster_hopping/config.rs; then
  echo "JUMP_NOT_ON_COMMUNICATING" >&2
  exit 1
fi
if ! grep -n 'recognition_skip' src/methods/cluster_hopping.rs; then
  echo "RECOGNITION_SKIP_NOT_WIRED" >&2
  exit 1
fi
cargo test --offline --lib -- \
  communicating_is_orbit_without_depth_reward \
  recognition_refunds_a_second_descent_into_the_same_well \
  recognition_does_not_count_a_peer_well_as_known_escape
echo HUNT_OK
