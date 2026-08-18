#!/usr/bin/env bash
# Terra compute-node test and occupancy build of the per-replica brain tree.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_build_brains.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/terra_rebuild_ira.sh" >&2
  exit 1
fi
cd "$ROOT"
git rev-parse HEAD >SOURCE_COMMIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
# The whole suite, not a name-filtered list. A filtered gate cannot see
# a test it does not select, which is how a red RETIS test shipped.
cargo test --release --features bank-rpc --no-fail-fast
# What CI enforces, so drift is caught here rather than on push.
cargo fmt --all -- --check
cargo clippy --release --features bank-rpc,featomic,ira --all-targets -- -D warnings
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
for symbol in different_decaf_family step_toward_catalog_hole CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats" "occupancy min families" "gt stop leftover-well"; do
  if ! grep -a -F -q "$symbol" "$BIN"; then
    echo "built binary missing $symbol" >&2
    exit 1
  fi
done
SERVER=target/release/examples/catalog_server
if ! grep -a -F -q occupancy_gt "$SERVER"; then
  echo "built binary missing occupancy_gt" >&2
  exit 1
fi
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
