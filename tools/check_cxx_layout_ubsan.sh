#!/usr/bin/env bash
# Configure, build, and run the ABI layout tests with
# -fsanitize=address,undefined.
#
# A pointer into kCxxSoversionTable compared with nullptr is not a constant
# expression under UBSan, so this build fails until the soversion lookup
# returns an index.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/.build-tmp/cxx-layout-ubsan}"

if [[ -z "$BUILD" || "$BUILD" == "/" ]]; then
  echo "refusing to use BUILD=$BUILD" >&2
  exit 1
fi

cd "$ROOT"
rm -rf "$BUILD"

cxx="${CXX:-c++}"
setup=(
  meson setup "$BUILD"
  -Dwith_tests=true
  -Dwith_examples=false
  -Dwith_xtensor=false
  -Dwith_eigen=false
  -Dwith_rpc=false
  -Dwith_cache=false
  -Dwith_fast_nlist=disabled
  -Dwith_fortran_pots=disabled
  -Dwith_highway=disabled
  -Dwith_ipi=disabled
  -Dpure_lib=true
  -Db_sanitize=address,undefined
)

# Clang leaves the sanitizer runtime unresolved in shared objects, so
# meson's default -Wl,--no-undefined fails the link.
if "$cxx" --version 2>&1 | grep -qi clang; then
  setup+=(-Db_lundef=false)
fi

"${setup[@]}"
meson compile -C "$BUILD" cxx_layout_test print_cxx_layout_stamp
# CxxLayoutTest compiles the soversion static asserts and the index lookup.
# CxxLayoutVirtualCounts checks the virtual-function counters in the same header.
meson test -C "$BUILD" --print-errorlogs \
  CxxLayoutTest CxxLayoutVirtualCounts
