#!/usr/bin/env bash
# Copyright (c) anneal developers
# One serial GMIN for the binary Lennard-Jones panel: Wales's public tarball
# at the pinned SVNREV, the evaluator cap, and a timer inside BLJ_CLUST.
# ALIGN, MPI and OpenMP stay off. The build lives apart from the one-component
# panel so that binary is left as it was recorded.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${BLJ_WORK:-/work/blj-gmin}
SRC=$WORK/gmin-src
TARBALL=$WORK/svn.tar.bz2
URL=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
REV=aac1548a8aec13b53ad12a8d6904625081685d2e
PATCHES=(
  "$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch"
  "$ROOT/scripts/morse_h2h/morse-timer.patch"
  "$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch"
  "$ROOT/scripts/blj_gmin/blj-timer.patch"
)

mkdir -p "$WORK/bin" "$SRC"
if [[ ! -s $TARBALL ]]; then
  if [[ -s /tmp/svn.tar.bz2 ]]; then
    cp -a /tmp/svn.tar.bz2 "$TARBALL"
  elif [[ -s /work/svn.tar.bz2 ]]; then
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

got=$(tr -d '[:space:]' < "$SRC/wales/GMIN/source/SVNREV")
if [[ $got != "$REV" ]]; then
  echo "GMIN SVNREV is $got, expected $REV" >&2
  exit 1
fi

if [[ ! -f $SRC/wales/GMIN/source/h2h_cap.f90 ]]; then
  for patch in "${PATCHES[@]}"; do
    patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$patch"
  done
fi
if ! grep -q H2H_POT_ENTER "$SRC/wales/GMIN/source/BLJcluster_nocut.f90"; then
  echo "BLJ_CLUST has no timer" >&2
  exit 1
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
    make -j4 GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN"
  chmod +x "$WORK/bin/GMIN"
fi

{
  echo "url=$URL"
  echo "revision=$REV"
  echo "svnrev=$got"
  echo "tarball_sha256=$(sha256sum "$TARBALL" | cut -d' ' -f1)"
  for patch in "${PATCHES[@]}"; do
    echo "patch=${patch#"$ROOT"/} $(sha256sum "$patch" | cut -d' ' -f1)"
  done
  echo "keyword=BLJCLUSTER_NOCUT"
  echo "cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
  echo "gfortran=$(gfortran -dumpfullversion)"
  echo "linked=$(ldd "$WORK/bin/GMIN" | awk '/blas|lapack/ {print $3}' | xargs -r readlink -f | tr '\n' ' ')"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | cut -d' ' -f1)"
} > "$WORK/gmin-source.txt"
echo "gmin url=$URL revision=$REV binary=$WORK/bin/GMIN"
