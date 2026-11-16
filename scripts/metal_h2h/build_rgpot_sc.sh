#!/usr/bin/env bash
# Copyright (c) anneal developers
# rgpot's Sutton-Chen entry at the rgpot revision of Cargo.lock, built from
# cargo's checkout alone with the neighbour list and SIMD sources it needs,
# into /work/metal-sc/rgpot-sc with sc_wall_micro and the revision.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LIBDIR=/work/metal-sc/rgpot-sc

rgpot_rev=$(sed -n 's|^source = "git+https://github.com/OmniPotentRPC/rgpot?rev=\([0-9a-f]*\)#.*|\1|p' \
  "$ROOT/Cargo.lock" | sort -u)
[[ $(wc -w <<< "$rgpot_rev") == 1 ]] || { echo "one rgpot revision expected in Cargo.lock" >&2; exit 1; }
cargo_home=${CARGO_HOME:-$HOME/.cargo}
src=$(ls -d "$cargo_home"/git/checkouts/rgpot-*/"${rgpot_rev:0:7}" | head -1)
cpp=$src/CppCore

mkdir -p "$LIBDIR"
g++ -O3 -std=c++20 -fPIC -shared -DNDEBUG -w -I"$cpp" -I"$src/include" -o "$LIBDIR/librgpot.so" \
  "$cpp/rgpot/SuttonChen/SuttonChenPot.cc" "$cpp"/rgpot/nlist/*.cc "$cpp"/rgpot/nlist/*.cpp \
  "$cpp/rgpot/simd/exact_fma_hw.cc" "$cpp/rgpot/simd/exp.cc" -Wl,--no-undefined
g++ -O3 -std=c++20 -DNDEBUG -w -I"$cpp" -I"$src/include" -o "$LIBDIR/sc_wall_micro" \
  "$HERE/sc_wall_micro.cc" "$LIBDIR/librgpot.so" -ldl -Wl,-rpath,"$LIBDIR"
printf '%s\n' "$rgpot_rev" > "$LIBDIR/rgpot-revision"
echo "rgpot library=$LIBDIR/librgpot.so revision=$rgpot_rev"
