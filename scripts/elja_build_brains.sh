#!/usr/bin/env bash
# Occupancy tests and per-replica brain build via `pixi run -e cluster`.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_build_brains.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
if [[ ! -x $PIXI ]]; then
  echo "missing pixi at $PIXI" >&2
  exit 1
fi
export PIXI_CACHE_DIR=${PIXI_CACHE_DIR:-$HOME/.cache/pixi}
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/elja_rebuild_ira.sh" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
if [[ ! -s SOURCE_COMMIT ]]; then
  echo "missing SOURCE_COMMIT" >&2
  exit 1
fi
if [[ ! -d .pixi/envs/cluster ]]; then
  echo "missing .pixi/envs/cluster; run pixi install -e cluster first" >&2
  exit 1
fi
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "pixi=$($PIXI --version)"
echo "rustc=$($PIXI run -e cluster rustc --version)"
echo "gcc=$($PIXI run -e cluster gcc --version | head -1)"
# Occupancy contract, not crate CI.
if command -v rustfmt >/dev/null 2>&1; then
  "$PIXI" run -e cluster cargo fmt --all -- --check
fi
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test elja_submission_contract occupancy_
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test cooperative_search visit_merges_the_posted
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --lib two_brains_exchange
"$PIXI" run -e cluster cargo test --offline --release --lib leftover_lambda
"$PIXI" run -e cluster cargo test --offline --release --lib leftover_soap_gt_with_two
"$PIXI" run -e cluster cargo test --offline --release --lib packing_role_is_per_family
"$PIXI" run -e cluster cargo test --offline --release --lib a_user_family_floor_of_one
"$PIXI" run -e cluster cargo test --offline --release --lib catalog_leave_refuses_a_same_family
"$PIXI" run -e cluster cargo test --offline --release --lib occupancy_leave_is_another_family_or_an_archive
"$PIXI" run -e cluster cargo test --offline --release --test occupancy_leave_contract leave_quench_keeps_the_walk
"$PIXI" run -e cluster cargo test --offline --release --lib interface_ranks_follow
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_policy extras_on_a_published
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_policy tis_extras_walk
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_policy explore_collapse_does_not_yank
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_mixing
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_packing leftover_soap_gt_plus
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_packing leftover_first_wave
"$PIXI" run -e cluster cargo test --offline --release --features bank-rpc --test catalog_packing packing_good_turing
"$PIXI" run -e cluster cargo test --offline --release --features featomic --lib leave_occupied_packing
"$PIXI" run -e cluster cargo test --offline --release --lib packing_householder_flips
"$PIXI" run -e cluster cargo test --offline --release --lib rgmin_walks_off_the_known
"$PIXI" run -e cluster cargo test --offline --release --lib packing_mode_is_nu3_mean
"$PIXI" run -e cluster cargo test --offline --release --lib span_rises_when_the_packing
"$PIXI" run -e cluster cargo test --offline --release --lib from_origin_climbs_the_covering
"$PIXI" run -e cluster cargo build --offline --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
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
echo "SMOKE"
"$BIN" 13 200 1 rec
echo "BUILD_OK $PWD/$BIN"
