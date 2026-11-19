#!/usr/bin/env bash
# Copyright (c) anneal developers
# One serial GMIN for the whole panel: Wales's public tarball at the pinned
# SVNREV, the evaluator cap with the Lennard-Jones timer, and the timers
# inside Morse and binary Gupta. ALIGN, MPI and OpenMP stay off.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${GMIN_PANEL_WORK:-/work/gmin-panel}
SRC=$WORK/gmin-src
TARBALL=$WORK/svn.tar.bz2
URL=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
REV=aac1548a8aec13b53ad12a8d6904625081685d2e
# GMIN_EXPECT_REV overrides the pinned panel revision when the public
# image has moved. GMIN_TARBALL_SHA256, when set, is checked before the
# tree is used. GMIN_PANEL_EXTRA_PATCHES is a colon-separated list of
# patch files applied after the panel patches.
EXPECT=${GMIN_EXPECT_REV:-$REV}
PATCHES=(
  "$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch"
  "$ROOT/scripts/morse_h2h/morse-timer.patch"
  "$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch"
)

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

if [[ -n ${GMIN_TARBALL_SHA256:-} ]]; then
  gotsha=$(sha256sum "$TARBALL" | cut -d' ' -f1)
  if [[ $gotsha != "$GMIN_TARBALL_SHA256" ]]; then
    echo "tarball sha256 is $gotsha, expected $GMIN_TARBALL_SHA256" >&2
    exit 1
  fi
fi

got=$(tr -d '[:space:]' < "$SRC/wales/GMIN/source/SVNREV")
if [[ $got != "$EXPECT" ]]; then
  echo "GMIN SVNREV is $got, expected $EXPECT" >&2
  exit 1
fi

if [[ ! -f $SRC/wales/GMIN/source/h2h_cap.f90 ]]; then
  for patch in "${PATCHES[@]}"; do
    patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$patch"
  done
fi

if [[ -n ${GMIN_PANEL_EXTRA_PATCHES:-} ]]; then
  IFS=':' read -r -a extra <<< "$GMIN_PANEL_EXTRA_PATCHES"
  for patch in "${extra[@]}"; do
    if [[ $patch == *newtip* ]] && grep -q H2H_POT_ENTER "$SRC/wales/GMIN/source/newtip.f90"; then
      continue
    fi
    patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$patch"
  done
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

{
  echo "url=$URL"
  echo "revision=$got"
  echo "panel_revision=$REV"
  echo "svnrev=$got"
  echo "tarball_sha256=$(sha256sum "$TARBALL" | cut -d' ' -f1)"
  for patch in "${PATCHES[@]}"; do
    echo "patch=${patch#"$ROOT"/} $(sha256sum "$patch" | cut -d' ' -f1)"
  done
  echo "cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
  echo "gfortran=$(gfortran -dumpfullversion)"
  echo "linked=$(ldd "$WORK/bin/GMIN" | awk '/blas|lapack/ {print $3}' | xargs -r readlink -f | paste -sd' ' -)"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | cut -d' ' -f1)"
} > "$WORK/gmin-source.txt"
echo "gmin url=$URL revision=$got binary=$WORK/bin/GMIN"
