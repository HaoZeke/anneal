#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN with the shared evaluator cap and timing inside binary Gupta.
# The tarball and revision are the ones scripts/gmin_h2h/build_gmin.sh pins.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
WORK=/work/metal-darby
SRC=$WORK/src
TARBALL=$WORK/svn.tar.bz2
URL=$GMIN_PIN_URL
REV=$GMIN_PIN_REVISION
TARBALL_SHA256=$GMIN_PIN_TARBALL_SHA256
CAP=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
TIME=$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch

LOCAL=${GMIN_TARBALL:-}
while [[ $# -gt 0 ]]; do
  case $1 in
    --tarball) LOCAL=${2:?missing path after --tarball}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK/bin" "$SRC"
if [[ -n $LOCAL ]]; then
  python3 "$PIN" verify-tarball "$LOCAL" >/dev/null
  if [[ $LOCAL != "$TARBALL" ]]; then
    cp -a "$LOCAL" "$TARBALL"
  fi
elif python3 "$PIN" verify-tarball "$TARBALL" >/dev/null 2>&1; then
  :
elif python3 "$PIN" verify-tarball /work/svn.tar.bz2 >/dev/null 2>&1; then
  cp -a /work/svn.tar.bz2 "$TARBALL"
else
  tmp=$(mktemp "$WORK/svn.tar.bz2.partial.XXXXXX")
  curl -fL --retry 3 -o "$tmp" "$URL"
  python3 "$PIN" verify-tarball "$tmp" >/dev/null
  mv "$tmp" "$TARBALL"
fi
python3 "$PIN" verify-tarball "$TARBALL" >/dev/null

if [[ -f $SRC/wales/GMIN/source/SVNREV ]] &&
   ! python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV" >/dev/null 2>&1; then
  rm -rf "$SRC/wales/GMIN"
  rm -f "$WORK/bin/GMIN"
fi

if [[ ! -f $SRC/wales/GMIN/source/SVNREV ]]; then
  tar -xjf "$TARBALL" -C "$SRC" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
fi

got=$(python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV")

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
svnrev=$got
tarball_sha256=$(python3 "$PIN" verify-tarball "$TARBALL")
potential=BGUPTAT
EOF
echo "gmin url=$URL revision=$REV binary=$WORK/bin/GMIN"
