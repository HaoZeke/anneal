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
if ! git diff --quiet HEAD --; then
  echo "refusing build: tracked source differs from HEAD" >&2
  git status --short >&2
  exit 2
fi
git rev-parse HEAD >SOURCE_COMMIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
# Occupancy contract, not crate CI. cloud_hop_compare SOAP on/off is
# a separate regression (simmAnnealTeX-qs4o); it does not gate this binary.
cargo fmt --all -- --check
cargo test --release --features bank-rpc --test elja_submission_contract occupancy_
cargo test --release --features bank-rpc --test cooperative_search visit_merges_the_posted
cargo test --release --features bank-rpc --lib two_brains_exchange
cargo test --release --lib leftover_lambda
cargo test --release --lib leftover_soap_gt_with_two
cargo test --release --lib packing_role_is_per_family
cargo test --release --lib a_user_family_floor_of_one
cargo test --release --lib catalog_leave_refuses_a_same_family
cargo test --release --lib occupancy_leave_is_another_family_or_an_archive
cargo test --release --test occupancy_leave_contract leave_quench_keeps_the_walk
cargo test --release --lib interface_ranks_follow
cargo test --release --features bank-rpc --test catalog_policy extras_on_a_published
cargo test --release --features bank-rpc --test catalog_policy tis_extras_walk
cargo test --release --features bank-rpc --test catalog_policy explore_collapse_does_not_yank
cargo test --release --features bank-rpc --test catalog_mixing
cargo test --release --features bank-rpc --test catalog_packing leftover_soap_gt_plus
cargo test --release --features bank-rpc --test catalog_packing leftover_first_wave
cargo test --release --features bank-rpc --test catalog_packing packing_good_turing
cargo test --release --features featomic --lib leave_occupied_packing
cargo test --release --lib packing_householder_flips
cargo test --release --lib rgmin_walks_off_the_known
cargo test --release --lib packing_mode_is_nu3_mean
cargo test --release --lib span_rises_when_the_packing
cargo test --release --lib from_origin_climbs_the_covering
cargo build --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example lj_census_calibration \
  --example catalog_server
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
for symbol in different_packing_family "occupancy leave archive hole" "packing invert nu3 pullback" "leave ridge climb" CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats" "occupancy min families" "gt stop packing"; do
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
CALIBRATOR=target/release/examples/lj_census_calibration
if ! grep -a -F -q seeded-random-cluster-quench-v1 "$CALIBRATOR"; then
  echo "built calibrator missing target-blind source policy" >&2
  exit 1
fi
sha256sum \
  target/release/examples/lj_cluster_search \
  target/release/examples/catalog_server \
  target/release/examples/lj_census_calibration \
  >BUILD_SHA256SUMS
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
