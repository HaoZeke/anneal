#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build librgpot, the shared library the Sutton-Chen driver loads, at the
# revision this repository pins for its pair kernels. Both arms load the same
# file. The build needs meson, ninja and a C++20 compiler.
#
#   build_rgpot.sh
#
#   PANEL_WORK        scratch directory, default .bench/blj-panel in the repo
#   PANEL_RGPOT_REV   revision to build, default the rev in Cargo.toml
#   PANEL_RGPOT_SRC   an existing checkout; cloned into the scratch directory
#                     when unset
#
# The library is left at $PANEL_WORK/rgpot/lib/librgpot.so with a manifest of
# the revision, the compiler and the file hash beside it.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${PANEL_WORK:-$ROOT/.bench/blj-panel}
OUT=$WORK/rgpot
REV=${PANEL_RGPOT_REV:-$(sed -n 's/^rgpot-pair = .*rev = "\([0-9a-f]\{40\}\)".*/\1/p' "$ROOT/Cargo.toml" | head -n 1)}
SRC=${PANEL_RGPOT_SRC:-$OUT/src}
URL=https://github.com/OmniPotentRPC/rgpot

if [ -z "$REV" ]; then
  echo "no rgpot revision: set PANEL_RGPOT_REV" >&2
  exit 1
fi
export CC=${CC:-gcc}
export CXX=${CXX:-g++}

mkdir -p "$OUT/lib"
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet "$URL" "$SRC"
fi
if ! git -C "$SRC" cat-file -e "${REV}^{commit}" 2> /dev/null; then
  git -C "$SRC" fetch --quiet origin "$REV"
fi
git -C "$SRC" checkout --detach -q "$REV"
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
  echo "rgpot checkout differs from $REV" >&2
  exit 1
fi

build=$OUT/build
rm -rf "$build"
meson setup "$build" "$SRC" --buildtype=release \
  -Dwith_tests=false -Dwith_examples=false -Dwith_rpc=false -Dwith_cache=false \
  -Dwith_xtb=false -Dwith_fast_nlist=disabled -Dwith_highway=disabled \
  -Dwith_ipi=disabled -Dwith_lammps=disabled -Dwith_gromacs=disabled \
  -Dwith_openmm=disabled -Dwith_plumed=disabled -Dwith_mpi=disabled \
  -Dwith_fortran_pots=disabled > "$OUT/setup.log"
meson compile -C "$build" rgpot > "$OUT/compile.log"

built=$(meson introspect "$build" --targets | python3 -c '
import json, sys
for target in json.load(sys.stdin):
    if target["name"] == "rgpot" and target["type"] == "shared library":
        print(target["filename"][0])
        break
')
cp -f "$built" "$OUT/lib/librgpot.so"
{
  echo "source=$URL"
  echo "revision=$REV"
  echo "build=meson release"
  echo "options=with_fast_nlist=disabled with_highway=disabled with_rpc=false with_cache=false"
  echo "compiler=$("$CXX" --version | head -n 1)"
  echo "library=librgpot.so ($(basename "$built"))"
  echo "sha256=$(sha256sum "$OUT/lib/librgpot.so" | cut -d' ' -f1)"
} > "$OUT/MANIFEST"
echo "librgpot revision=$REV built"
