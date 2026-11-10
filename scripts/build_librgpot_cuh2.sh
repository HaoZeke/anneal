#!/usr/bin/env bash
# Build librgpot_cuh2.so from an OmniPotentRPC/rgpot checkout.
# This script pins commit 25c759da82eececa8897e4634046fab27409c81a.
set -euo pipefail

expected=25c759da82eececa8897e4634046fab27409c81a

if (($# != 2)); then
  echo "usage: $0 RGPOT_ROOT OUTPUT" >&2
  exit 2
fi

rgpot=$1
output=$2
head=$(git -C "$rgpot" rev-parse HEAD)
if [[ $head != "$expected" ]]; then
  echo "rgpot HEAD $head is not $expected" >&2
  exit 2
fi
if ! git -C "$rgpot" diff --quiet HEAD --; then
  echo "rgpot tracked source differs from $expected" >&2
  exit 2
fi
if ! command -v gfortran >/dev/null || ! command -v g++ >/dev/null; then
  echo "gfortran and g++ are required" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
src=$rgpot/CppCore/rgpot/fortran
vesin=$rgpot/third_party/vesin

g++ -std=c++17 -fPIC -O2 -c "$vesin/vesin-single-build.cpp" \
  -I "$vesin" -o "$work/vesin.o"
gfortran -std=f2018 -fimplicit-none -frecursive -fPIC -O2 -J "$work" \
  -c "$vesin/fortran/cdef.f90" -o "$work/cdef.o"
gfortran -std=f2018 -fimplicit-none -frecursive -fPIC -O2 -J "$work" \
  -c "$vesin/fortran/vesin.f90" -o "$work/vesin_f.o"
for name in rgpot_kinds rgpot_ferror rgpot_neighbors rgpot_workspace rgpot_cuh2 rgpot_cuh2_capi; do
  gfortran -std=f2018 -fimplicit-none -frecursive -fPIC -Wall \
    -Wno-unused-dummy-argument -O2 -J "$work" \
    -c "$src/$name.f90" -o "$work/$name.o"
done
mkdir -p "$(dirname "$output")"
gfortran -shared -Wl,--no-undefined -o "$output" \
  "$work"/rgpot_kinds.o \
  "$work"/rgpot_ferror.o \
  "$work"/rgpot_neighbors.o \
  "$work"/rgpot_workspace.o \
  "$work"/rgpot_cuh2.o \
  "$work"/rgpot_cuh2_capi.o \
  "$work"/cdef.o \
  "$work"/vesin_f.o \
  "$work"/vesin.o \
  -lstdc++ -lm -pthread
if ! nm -D --defined-only "$output" |
  awk '$NF == "rgpot_cuh2_force" { found = 1 } END { exit found ? 0 : 1 }'; then
  echo "$output does not export rgpot_cuh2_force" >&2
  exit 1
fi
printf '%s\n' "$expected" >"$(dirname "$output")/rgpot.sha"
printf 'RGPOT_CUH2_ENGINE_OK %s\n' "$output"
