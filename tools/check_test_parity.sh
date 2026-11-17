#!/usr/bin/env bash
# Check, without building, that meson and CMake register the tests
# CppCore/tests/registry.txt lists.
#
# Each configuration below is set up with both build systems, and each runs
# its test_list_parity test: the tests it registered must be the entries the
# registry enables for the features it found, with the options the registry
# gives them. Each also writes those features and its shared tests, and the
# two lists must be equal, so a test or a feature that one build system has
# and the other lacks fails the check.
#
# tools/check_fp_parity.py then reads both compile_commands.json files: each
# source both build systems compile must get the same contraction and
# fast-math flags from each, or a bit-for-bit test passes in one build and
# fails in the other on a target with FMA, such as arm64.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch:?}"' EXIT
cd "$ROOT"

python3 CppCore/tests/check_test_registry.py lint

# Highway is fetched at configure time and gates no test.
meson_common=(-Dwith_tests=true -Dwith_examples=false -Dwith_xtensor=false
  -Dwith_eigen=false -Dwith_highway=disabled)
cmake_common=(-DCMAKE_BUILD_TYPE=Release -DRGPOT_BUILD_TESTS=ON
  -DRGPOT_BUILD_EXAMPLES=OFF -DRGPOT_WITH_XTENSOR=OFF -DRGPOT_WITH_EIGEN=OFF
  -DRGPOT_WITH_HIGHWAY=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# meson looks for MPI only when asked to; CMake always does.
plain_meson=(-Dwith_rpc=false -Dwith_cache=false -Dwith_mpi=auto)
plain_cmake=(-DRGPOT_WITH_RPC=OFF -DRGPOT_WITH_CACHE=OFF)
rpc_cache_meson=(-Dwith_rpc=true -Dwith_cache=true -Dwith_mpi=auto)
rpc_cache_cmake=(-DRGPOT_WITH_RPC=ON -DRGPOT_WITH_CACHE=ON)
# CMake's client-only bridge builds neither the Fortran potentials nor
# librgpot_mpi.
client_meson=(-Dwith_rpc_client_only=true -Dwith_fortran_pots=disabled
  -Dwith_mpi=disabled)
client_cmake=(-DRGPOT_RPC_CLIENT_ONLY=ON)

quiet() {
  local log=$1
  shift
  if ! "$@" >"$log" 2>&1; then
    cat "$log" >&2
    echo "failed: $*" >&2
    return 1
  fi
}

compare() {
  local config=$1
  local -n meson_args="${config}_meson" cmake_args="${config}_cmake"
  local dir="$scratch/$config"
  mkdir "$dir"
  quiet "$dir/meson-setup.log" \
    meson setup "$dir/meson" "${meson_common[@]}" "${meson_args[@]}"
  quiet "$dir/cmake-setup.log" \
    cmake -S . -B "$dir/cmake" "${cmake_common[@]}" "${cmake_args[@]}"
  RGPOT_TEST_LIST="$dir/meson.txt" quiet "$dir/meson-test.log" \
    meson test -C "$dir/meson" --no-rebuild --print-errorlogs test_list_parity
  RGPOT_TEST_LIST="$dir/cmake.txt" quiet "$dir/ctest.log" \
    ctest --test-dir "$dir/cmake" --output-on-failure --no-tests=error \
    -R '^test_list_parity$'
  if ! diff -u --label "meson ($config)" --label "cmake ($config)" \
    "$dir/meson.txt" "$dir/cmake.txt"; then
    echo "$config: meson and CMake differ in the features or tests above" >&2
    return 1
  fi
  echo "$config: meson and CMake register the same" \
    "$(grep -c '^test ' "$dir/cmake.txt") shared tests"
  if ! python3 tools/check_fp_parity.py "$dir/meson/compile_commands.json" \
    "$dir/cmake/compile_commands.json" >"$dir/fp.txt"; then
    echo "$config: meson and CMake compile the sources above with" \
      "different floating-point flags" >&2
    return 1
  fi
  echo "$config: $(cat "$dir/fp.txt")"
}

for config in plain rpc_cache client; do
  compare "$config"
done
