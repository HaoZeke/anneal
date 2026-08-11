#!/usr/bin/env bash
# Compute-node Elja build of lj_cluster_search + bank_server.
# rustup lives in ~/.cargo/bin and is off the non-interactive PATH.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_build_lj.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-build}
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
mkdir -p "$SYS/bin"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/cc"
ln -sfn "$GCC/bin/gcc" "$SYS/bin/gcc"
ln -sfn "$GCC/bin/g++" "$SYS/bin/g++"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export FC="${GCC}/bin/gfortran"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
# rustc 1.95 defaults to rust-lld, which cannot consume GNU ld scripts.
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-fuse-ld=/usr/bin/ld -C link-arg=-B${SYS} -L ${SYS}"
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID"
lscpu | grep -E "Model name|Vendor ID" || true
BIN=target/release/examples/lj_cluster_search
mkdir -p target/release/examples
if [[ -e "$BIN" ]] && ! ldd "$BIN" >/dev/null 2>&1; then
  mv -f "$BIN" "${BIN}.foreign-glibc"
fi
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | head -1)"
echo "glibc=$(ldd --version | head -1)"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
if [[ ! -e "$IRA_LIB_DIR/libira.so" ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/elja_rebuild_ira.sh" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
if ldd "$IRA_LIB_DIR/libira.so" | grep -q "not found"; then
  echo "libira unresolved; run scripts/elja_rebuild_ira.sh" >&2
  ldd "$IRA_LIB_DIR/libira.so"
  exit 1
fi
# Registry is on NFS from the login fetch. Compute may have no outbound net.
cargo build --offline --release --features featomic,ira,bank-rpc --example lj_cluster_search --example bank_server
ldd "$BIN"
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
echo "NOTE molecular_cluster and slab_adsorption need --features rgpot-ex and potserv"
