#!/usr/bin/env bash
# Compute-node Elja build of LJ search and coordinator executables.
# rustup lives in ~/.cargo/bin and is off the non-interactive PATH.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_build_lj.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-build}
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
CMAKE_BIN=${CMAKE_BIN:-$HOME/rgpot/.pixi/envs/xtbbld/bin/cmake}
if [[ ! -x $CMAKE_BIN ]]; then
  echo "missing compute-node CMake executable: $CMAKE_BIN" >&2
  exit 1
fi
mkdir -p "$SYS/bin"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/cc"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/gcc"
ln -sfn "$GCC/bin/g++" "$SYS/bin/g++"
ln -sfn "$CMAKE_BIN" "$SYS/bin/cmake"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export FC="${GCC}/bin/gfortran"
# Compute nodes carry no glibc development headers: /usr/include holds
# seventy entries and no stdint.h (measured on compute-20 and
# compute-32), while the login node has the full set for the same glibc
# 2.28. The login node's headers are staged into the sysroot once
# (rsync -a /usr/include/ $SYS/usr-include/) and every C and C++
# compile is pointed at them, which is what lets nng-sys build its C
# library under cmake on a compute node.
if [[ ! -e "$SYS/usr-include/stdint.h" ]]; then
  echo "missing $SYS/usr-include; on the login node run:" >&2
  echo "  rsync -a /usr/include/ $SYS/usr-include/" >&2
  exit 1
fi
export CFLAGS="${CFLAGS:-} -isystem $SYS/usr-include"
export CXXFLAGS="${CXXFLAGS:-} -isystem $SYS/usr-include"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
# Do not pass -fuse-ld=/path: OHPC gcc 12 rejects it. collect2 finds
# ld via -B. rust-lld is avoided by pointing gcc at SYS/bin/ld.
ln -sfn /usr/bin/ld "$SYS/bin/ld"
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-B${SYS} -C link-arg=-B${SYS}/bin -L ${SYS}"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
lscpu | grep -E "Model name|Vendor ID" || true
BIN=target/release/examples/lj_cluster_search
mkdir -p target/release/examples
if [[ -e "$BIN" ]] && ! ldd "$BIN" >/dev/null 2>&1; then
  mv -f "$BIN" "${BIN}.foreign-glibc"
fi
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | awk 'NR == 1 { print }')"
echo "cmake=$($CMAKE_BIN --version | awk 'NR == 1 { print }') path=$(command -v cmake)"
echo "glibc=$(ldd --version | awk 'NR == 1 { print }')"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
if [[ ! -e "$IRA_LIB_DIR/libira.so" ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/elja_rebuild_ira.sh" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
if ldd "$IRA_LIB_DIR/libira.so" | grep -F "not found" >/dev/null; then
  echo "libira unresolved; run scripts/elja_rebuild_ira.sh" >&2
  ldd "$IRA_LIB_DIR/libira.so"
  exit 1
fi
# nng-sys builds the nng C library through cmake at build time, so the
# sysroot cmake above is a build input and not only a link input. The
# crate itself has to be in the NFS registry already: compute has no
# outbound network, so a cache miss here fails as an unresolved
# dependency rather than as a download, which reads like a lockfile
# problem and is not one.
if ! ls "${CARGO_HOME:-$HOME/.cargo}"/registry/cache/*/nng-sys-*.crate >/dev/null 2>&1; then
  echo "nng-sys is not in the offline registry; run on the login node:" >&2
  echo "  cargo fetch --locked" >&2
  exit 1
fi
"$SYS/bin/cmake" --version >/dev/null 2>&1 || {
  echo "nng-sys needs a working cmake at $SYS/bin/cmake" >&2
  exit 1
}
# Registry is on NFS from the login fetch. Compute may have no outbound net.
if ! git diff --quiet HEAD --; then
  echo "refusing build: tracked source differs from HEAD" >&2
  git status --short >&2
  exit 2
fi
cargo build --offline --locked --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example lj_census_calibration \
  --example catalog_server \
  --example bank_server \
  --example leave_packing_probe
ldd "$BIN"
git rev-parse HEAD >SOURCE_COMMIT
sha256sum \
  target/release/examples/lj_cluster_search \
  target/release/examples/catalog_server \
  >BUILD_SHA256SUMS
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
echo "NOTE molecular_cluster and slab_adsorption: scripts/elja_build_rgpot_ex.sh (in-process rgpot, not potserv)"
