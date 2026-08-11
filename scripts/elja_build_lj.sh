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
cargo build --release --features featomic,bank-rpc --example lj_cluster_search --example bank_server
ldd "$BIN"
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
echo "NOTE molecular_cluster and slab_adsorption need --features rgpot-ex and potserv"
