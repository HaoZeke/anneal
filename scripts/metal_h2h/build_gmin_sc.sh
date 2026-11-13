#!/usr/bin/env bash
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
build=$SRC/wales/GMIN/build/gfortran
mkdir -p "$build" "$WORK/bin"
(
  cd "$build"
  if [[ ! -f Makefile ]]; then
    FC=gfortran CC=gcc CXX=g++ cmake ../../source \
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
cat > "$WORK/gmin-source.txt" << EOF
url=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
revision=aac1548a8aec13b53ad12a8d6904625081685d2e
svnrev=GMIN/source/SVNREV
potential=SC
EOF
echo "gmin binary=$WORK/bin/GMIN"
