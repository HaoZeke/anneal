#!/usr/bin/env bash
# A second GMIN binary whose USERPOT is rgpot's Tersoff carbon.
# The public GMIN has no Tersoff keyword. Its USERPOT keyword calls the
# userpot_* routines, which a standard build takes from dummy_userpot.f90.
# This build copies the capped source tree, puts tersoff_userpot.f90 in
# that file's place and links librgpot_covalent.so.
#
#   build_gmin_tersoff.sh <librgpot_covalent.so>
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LIB=$(realpath "${1:?librgpot_covalent.so}")
WORK=/work/gmin-h2h
TREE=$WORK/tersoff

bash "$HERE/build_gmin.sh"
if [[ ! -x $WORK/bin/GMIN-tersoff ]]; then
  rm -rf "$TREE"
  mkdir -p "$TREE"
  (cd "$WORK/src" && tar --exclude=./wales/GMIN/build -cf - ./wales) | tar -xf - -C "$TREE"
  cp "$HERE/tersoff_userpot.f90" "$TREE/wales/GMIN/source/dummy_userpot.f90"
  build=$TREE/wales/GMIN/build/gfortran
  mkdir -p "$build"
  (
    cd "$build"
    FC=gfortran CC=gcc CXX=g++ cmake ../../source \
      -DWITH_ALIGN=OFF \
      -DWITH_MPI=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_SPARSE=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MYBLAS=OFF \
      -DWITH_MYLAPACK=OFF \
      "-DCMAKE_Fortran_STANDARD_LIBRARIES=$LIB" \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$(dirname "$LIB")"
    make -j"$(nproc)" GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN-tersoff"
  chmod +x "$WORK/bin/GMIN-tersoff"
fi
echo "gmin-tersoff binary=$WORK/bin/GMIN-tersoff library=$LIB"
