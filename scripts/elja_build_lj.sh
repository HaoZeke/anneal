#!/usr/bin/env bash
# Native Elja build of lj_cluster_search. No rgpot-ex, so no capnp.
# rustup lives in ~/.cargo/bin and is off the non-interactive PATH.
set -euo pipefail
export PATH="${HOME}/.cargo/bin:${PATH}"
ROOT=${LJ_ROOT:-$HOME/anneal-build}
cd "$ROOT"
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
cargo build --release --features featomic,ira,bank-rpc --example lj_cluster_search --example bank_server
ldd "$BIN"
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
echo "NOTE molecular_cluster and slab_adsorption need --features rgpot-ex and potserv"
