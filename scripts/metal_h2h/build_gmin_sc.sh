#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN with the shared call cap and timing inside Sutton-Chen.
# The binary Gupta build is left where it is. This one is separate.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
bash "$ROOT/scripts/metal_h2h/build_gmin_darby.sh"
SRC=/work/metal-darby/src
TIME=$ROOT/bench/metal-cluster-h2h/gmin-sc-time.patch
if ! grep -q H2H_SC_TICK "$SRC/wales/GMIN/source/potential.f90"; then
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$TIME"
fi
WORK=/work/metal-sc
# Basin-hopping is chaotic in the last bit, so the BLAS GMIN links is part of
# the run: reference BLAS and OpenBLAS end the same seed on different minima.
# The sc-n12 GMIN rows reproduce with OpenBLAS.
build=$SRC/wales/GMIN/build/openblas
mkdir -p "$build" "$WORK/bin"
(
  cd "$build"
  if [[ ! -f Makefile ]]; then
    FC=gfortran CC=gcc CXX=g++ cmake ../../source \
      -DBLA_VENDOR=OpenBLAS \
      -DWITH_ALIGN=OFF \
      -DWITH_MPI=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_SPARSE=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MYBLAS=OFF \
      -DWITH_MYLAPACK=OFF
  fi
  make -j"$(nproc)" GMIN
)
cp -a "$build/GMIN" "$WORK/bin/GMIN"
chmod +x "$WORK/bin/GMIN"
blas=$(ldd "$WORK/bin/GMIN" | awk '/libopenblas/ {print $3}')
if [[ -z $blas ]]; then
  echo "GMIN is not linked against OpenBLAS" >&2
  exit 1
fi
cat > "$WORK/gmin-source.txt" << EOF
url=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
revision=9cbec1a446352bc9ef064bd2b1a5767194655919
svnrev=GMIN/source/SVNREV
potential=SC
blas=$(readlink -f "$blas")
EOF
echo "gmin binary=$WORK/bin/GMIN"
