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
if ! git diff --quiet HEAD --; then
  echo "refusing build: tracked source differs from HEAD" >&2
  git status --short >&2
  exit 2
fi
git rev-parse HEAD >SOURCE_COMMIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | head -1)"
echo "cmake=$(cmake --version | head -1)"
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server \
  --example campaign_env \
  --example leave_packing_probe
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
# The Leave accept is the packing community, not the cell grain, so the
# binary must carry leaves_packing and must not carry the two cell-grain
# entry points it replaced: a DECAF L1 of 0.20 is passed by icosahedral
# isomers of the packing being left, which is what made isomer motion
# read as a new packing. The deposit is a free energy, so the arrivals
# and the standing bias have to reach it.
for symbol in leaves_packing different_packing_family arm_leave_free \
  packing_reference_book credit_packing_deposit; do
  if ! grep -a -F -q "$symbol" "$BIN"; then
    echo "built binary missing $symbol" >&2
    exit 1
  fi
done
for symbol in different_decaf_family occupancy_leave_new_class; do
  if grep -a -F -q "$symbol" "$BIN"; then
    echo "built binary still reaches the cell grain through $symbol" >&2
    exit 1
  fi
done
sha256sum \
  target/release/examples/lj_cluster_search \
  target/release/examples/catalog_server \
  target/release/examples/campaign_env \
  >BUILD_SHA256SUMS
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
