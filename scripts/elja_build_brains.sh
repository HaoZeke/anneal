#!/usr/bin/env bash
# Elja compute-node occupancy tests and per-replica brain build.
# Isolated tree. Does not touch anneal-stop, anneal-accel, or h5/hq.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_build_brains.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
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
ln -sfn /usr/bin/ld "$SYS/bin/ld"
export PATH="${SYS}/bin:${GCC}/bin:${HOME}/.cargo/bin:${PATH}"
export CC="${GCC}/bin/gcc"
export CXX="${GCC}/bin/g++"
export FC="${GCC}/bin/gfortran"
export LIBRARY_PATH="${SYS}:${GCC}/lib64:/usr/lib64:${LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:-} -C linker=${GCC}/bin/gcc -C link-arg=-B${SYS} -C link-arg=-B${SYS}/bin -L ${SYS}"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"
GLIBC_INCLUDE=${ELJA_GLIBC_INCLUDE:-$HOME/elja-glibc-include}
if [[ ! -f $GLIBC_INCLUDE/stdint.h ]]; then
  echo "missing $GLIBC_INCLUDE/stdint.h; run scripts/elja_stage_glibc_include.sh on login" >&2
  exit 1
fi
export CPATH="${GLIBC_INCLUDE}${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="${GLIBC_INCLUDE}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="${GLIBC_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CFLAGS="-idirafter ${GLIBC_INCLUDE} ${CFLAGS:-}"
export CPPFLAGS="-idirafter ${GLIBC_INCLUDE} ${CPPFLAGS:-}"
export CMAKE_C_FLAGS="-idirafter ${GLIBC_INCLUDE} ${CMAKE_C_FLAGS:-}"
export CMAKE_INCLUDE_PATH="${GLIBC_INCLUDE}${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
if [[ ! -e $IRA_LIB_DIR/libira.so ]]; then
  echo "missing $IRA_LIB_DIR/libira.so; run scripts/elja_rebuild_ira.sh" >&2
  exit 1
fi
cd "$ROOT"
if [[ ! -s SOURCE_COMMIT ]]; then
  echo "missing SOURCE_COMMIT; write it on the login node" >&2
  exit 1
fi
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "source=$(cat SOURCE_COMMIT)"
echo "rustc=$(rustc --version)"
echo "gcc=$(gcc --version | head -1)"
# Occupancy contract, not crate CI. Registry from the login fetch.
cargo fmt --all -- --check
cargo test --offline --release --features bank-rpc --test elja_submission_contract occupancy_
cargo test --offline --release --features bank-rpc --test cooperative_search visit_merges_the_posted
cargo test --offline --release --features bank-rpc --lib two_brains_exchange
cargo test --offline --release --lib leftover_lambda
cargo test --offline --release --lib leftover_soap_gt_with_two
cargo test --offline --release --lib packing_role_is_per_family
cargo test --offline --release --lib a_user_family_floor_of_one
cargo test --offline --release --lib catalog_leave_refuses_a_same_family
cargo test --offline --release --lib occupancy_leave_is_another_family_or_an_archive
cargo test --offline --release --lib interface_ranks_follow
cargo test --offline --release --features bank-rpc --test catalog_policy extras_on_a_published
cargo test --offline --release --features bank-rpc --test catalog_policy tis_extras_walk
cargo test --offline --release --features bank-rpc --test catalog_policy explore_collapse_does_not_yank
cargo test --offline --release --features bank-rpc --test catalog_mixing
cargo test --offline --release --features bank-rpc --test catalog_packing leftover_soap_gt_plus
cargo test --offline --release --features bank-rpc --test catalog_packing leftover_first_wave
cargo test --offline --release --features bank-rpc --test catalog_packing packing_good_turing
cargo test --offline --release --features featomic --lib leave_occupied_packing
cargo build --offline --release --features featomic,ira,bank-rpc \
  --example lj_cluster_search \
  --example catalog_server
BIN=target/release/examples/lj_cluster_search
ldd "$BIN"
for symbol in different_decaf_family "occupancy leave archive hole" CATALOG_BRAIN_LISTEN "leftover-SOAP TIS seats" "occupancy min families" "gt stop leftover-well"; do
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
