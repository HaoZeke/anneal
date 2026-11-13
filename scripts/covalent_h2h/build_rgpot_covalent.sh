#!/usr/bin/env bash
# Build librgpot_covalent.so from an OmniPotentRPC/rgpot checkout.
# This script pins commit 20c58fe5a2859cd0a79a3e06d64d567f96d0ccbe (3.6.0).
# The library is rgpot's StillingerWeberSi and TersoffSiC behind the two
# C entry points in rgpot_covalent.cc. No potential code lives in anneal.
set -euo pipefail

expected=20c58fe5a2859cd0a79a3e06d64d567f96d0ccbe

if (($# != 2)); then
  echo "usage: $0 RGPOT_ROOT OUTPUT" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
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

core=$rgpot/CppCore
mkdir -p "$(dirname "$output")"
# The source list and -ffp-contract=off follow CppCore/meson.build.
(
  cd "$core"
  g++ -std=c++20 -O2 -fPIC -shared -ffp-contract=off -Wl,--no-undefined \
    -I. -I../third_party/vesin -o "$output" \
    "$HERE/rgpot_covalent.cc" \
    rgpot/StillingerWeber/SWSi.cc \
    rgpot/Tersoff/TersoffSiC.cc \
    rgpot/nlist/HalfShell.cc \
    rgpot/nlist/HalfShellWalk.cc \
    rgpot/nlist/NeighbourList.cc \
    rgpot/nlist/Neighbours.cc \
    rgpot/nlist/MinimumImage.cpp \
    rgpot/simd/exact_fma_hw.cc \
    rgpot/simd/exp.cc \
    rgpot/PotHelpers.cc
)
for symbol in rgpot_covalent_energy_gradient rgpot_covalent_sw_units; do
  if ! nm -D --defined-only "$output" |
    awk -v want="$symbol" '$NF == want { found = 1 } END { exit found ? 0 : 1 }'; then
    echo "$output does not export $symbol" >&2
    exit 1
  fi
done
echo "librgpot_covalent=$output rgpot=$expected"
