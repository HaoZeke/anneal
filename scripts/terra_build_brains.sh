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
cargo test --release --features bank-rpc --test elja_submission_contract occupancy_
cargo test --release --features bank-rpc --test cooperative_search visit_merges_the_posted
cargo test --release --features bank-rpc --lib two_brains_exchange
cargo test --release --lib leftover_lambda
cargo test --release --lib leftover_soap_gt_with_two
cargo test --release --lib packing_role_is_per_family
cargo test --release --lib catalog_leave_keeps_the_hole
cargo test --release --lib interface_ranks_follow
cargo test --release --features bank-rpc --test catalog_policy extras_on_a_published
cargo test --release --features bank-rpc --test catalog_policy tis_extras_walk
cargo test --release --features bank-rpc --test catalog_policy explore_collapse_does_not_yank
cargo test --release --features bank-rpc --test catalog_mixing
cargo test --release --features bank-rpc --test catalog_packing leftover_soap_gt_plus
cargo test --release --features bank-rpc --test catalog_packing packing_good_turing
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
for symbol in different_decaf_family step_toward_catalog_hole CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats"; do
  if ! grep -a -F -q "$symbol" "$BIN"; then
    echo "built binary missing $symbol" >&2
    exit 1
  fi
done
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
