#!/usr/bin/env bash
# Terra compute-node build of lj_cluster_search + catalog_server.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_build_lj.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-occ-verify}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/terra_rebuild_ira.sh" >&2
  exit 1
fi
if ldd "$IRA_LIB_DIR/libira.so" | grep -q "not found"; then
  echo "libira unresolved" >&2
  ldd "$IRA_LIB_DIR/libira.so"
  exit 1
fi
cd "$ROOT"
git rev-parse HEAD >SOURCE_COMMIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | head -1)"
echo "cmake=$(cmake --version | head -1)"
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server \
  --example leave_packing_probe
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
if ! grep -a -F -q different_decaf_family "$BIN"; then
  echo "built binary missing different_decaf_family" >&2
  exit 1
fi
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
