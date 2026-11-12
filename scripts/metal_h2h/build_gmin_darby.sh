#!/usr/bin/env bash
# GMIN with the shared evaluator cap and timing inside binary Gupta.
# The tarball and revision are the ones scripts/gmin_h2h/build_gmin.sh pins.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=/work/metal-darby
SRC=$WORK/src
TARBALL=$WORK/svn.tar.bz2
URL=http://www-wales.ch.cam.ac.uk/svn.tar.bz2
REV=aac1548a8aec13b53ad12a8d6904625081685d2e
CAP=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
TIME=$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch

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
