#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN with the shared evaluator cap and timing inside binary Gupta.
# The tarball and revision are the 2026-11-19 snapshot pinned by
# scripts/gmin_panel/build_gmin.sh. A local file is required. This script
# does not fetch.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/metal-darby
SRC=$WORK/src
TARBALL=$WORK/svn.tar.bz2
URL=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
REV=9cbec1a446352bc9ef064bd2b1a5767194655919
TARBALL_SHA256=ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776
CAP=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
TIME=$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch

mkdir -p "$WORK/bin" "$SRC"
if [[ ! -s $TARBALL ]]; then
  if [[ -n ${GMIN_TARBALL:-} && -s $GMIN_TARBALL ]]; then
    cp -a "$GMIN_TARBALL" "$TARBALL"
  elif [[ -s /work/svn.tar.bz2 ]]; then
    cp -a /work/svn.tar.bz2 "$TARBALL"
  else
    echo "pinned GMIN tarball is missing" >&2
    exit 1
  fi
fi
got_sha=$(sha256sum "$TARBALL" | awk '{print $1}')
if [[ $got_sha != "$TARBALL_SHA256" ]]; then
  echo "tarball sha256 mismatch: got ${got_sha}, expected ${TARBALL_SHA256}" >&2
  exit 1
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
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$CAP"
fi
if ! grep -q 'H2H_POT_ENTER' "$SRC/wales/GMIN/source/BGupta.f90"; then
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$TIME"
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
potential=BGUPTAT
EOF
echo "gmin url=$URL revision=$REV binary=$WORK/bin/GMIN"
