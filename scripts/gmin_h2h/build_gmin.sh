#!/usr/bin/env bash
# Build Wales's public GMIN and the in-tree evaluator cap.
# The public drop is the group tarball. GMIN/source/SVNREV is the revision.
# ALIGN and OpenMP stay off: the Lennard-Jones comparison is serial, and
# the ALIGN build pulls a separate FFTW tree this run does not use.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/gmin-h2h
SRC=$WORK/src
TARBALL=$WORK/svn.tar.bz2
URL=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
REV=aac1548a8aec13b53ad12a8d6904625081685d2e
PATCH=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch

mkdir -p "$WORK/bin" "$SRC"
if [[ ! -s $TARBALL ]]; then
  if [[ -s /work/svn.tar.bz2 ]]; then
    cp -a /work/svn.tar.bz2 "$TARBALL"
  else
    curl -fL --retry 3 -o "$TARBALL" "$URL"
  fi
fi

if [[ ! -f $SRC/wales/GMIN/source/SVNREV ]]; then
  tar -xjf "$TARBALL" -C "$SRC" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
fi
if [[ ! -f $SRC/wales/HiRE/hire_interface.F90 ]]; then
  tar -xjf "$TARBALL" -C "$SRC" ./wales/HiRE
fi

got=$(tr -d '[:space:]' < "$SRC/wales/GMIN/source/SVNREV")
if [[ $got != "$REV" ]]; then
  echo "GMIN SVNREV is $got, expected $REV" >&2
  exit 1
fi

if [[ ! -f $SRC/wales/GMIN/source/h2h_cap.f90 ]]; then
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$PATCH"
fi

if [[ ! -x $WORK/bin/GMIN ]]; then
  build=$SRC/wales/GMIN/build/gfortran
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
      -DWITH_MYLAPACK=OFF
    make -j"$(nproc)" GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN"
  chmod +x "$WORK/bin/GMIN"
fi

cat > "$WORK/gmin-source.txt" << EOF
url=$URL
revision=$REV
svnrev=GMIN/source/SVNREV
cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF
EOF
echo "gmin url=$URL revision=$REV binary=$WORK/bin/GMIN"
