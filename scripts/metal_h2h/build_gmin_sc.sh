#!/usr/bin/env bash
# Copyright (c) anneal developers
# GMIN with the shared call cap and timing inside Sutton-Chen.
# The binary Gupta build is left where it is. This one is separate.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
DARBY_WORK=${GMIN_DARBY_WORK:-/work/metal-darby}
WORK=${GMIN_SC_WORK:-/work/metal-sc}
SRC=$WORK/src
BUILD_KEY_FILE=$WORK/gmin-build-key
CAP=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
CAP_SOURCE=$ROOT/bench/lj104-lattice/v2/h2h_cap.f90
DARBY_TIME=$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch
TIME=$ROOT/bench/metal-cluster-h2h/gmin-sc-time.patch
CMAKE_FLAGS=(
  -DBLA_VENDOR=OpenBLAS
  -DWITH_ALIGN=OFF
  -DWITH_MPI=OFF
  -DWITH_OPENMP=OFF
  -DWITH_SPARSE=OFF
  -DWITH_PYTHON=OFF
  -DWITH_MYBLAS=OFF
  -DWITH_MYLAPACK=OFF
)

GMIN_DARBY_WORK=$DARBY_WORK bash "$ROOT/scripts/metal_h2h/build_gmin_darby.sh" "$@"
TARBALL=$DARBY_WORK/svn.tar.bz2
revision=$(python3 "$PIN" read-svnrev "$DARBY_WORK/src/wales/GMIN/source/SVNREV")
tarball_sha256=$(python3 "$PIN" verify-tarball "$TARBALL")

FC_BIN=${FC:-gfortran}
CC_BIN=${CC:-gcc}
CXX_BIN=${CXX:-g++}
GFORTRAN_VER=$("$FC_BIN" -dumpfullversion)
KEY_ARGS=(
  --revision "$revision"
  --tarball-sha256 "$tarball_sha256"
  --input "$CAP_SOURCE"
  --input "$CAP"
  --input "$DARBY_TIME"
  --input "$TIME"
  "--option=FC=$FC_BIN"
  "--option=CC=$CC_BIN"
  "--option=CXX=$CXX_BIN"
  "--option=gfortran=$GFORTRAN_VER"
)
for flag in "${CMAKE_FLAGS[@]}"; do
  KEY_ARGS+=("--option=$flag")
done
BUILD_KEY=$(python3 "$PIN" build-key "${KEY_ARGS[@]}")
OLD_BUILD_KEY=$(
  python3 "$PIN" read-build-key \
    --metadata "$WORK/gmin-source.txt" \
    --key-file "$BUILD_KEY_FILE"
)

# Basin-hopping is chaotic in the last bit, so the BLAS GMIN links is part of
# the run: reference BLAS and OpenBLAS end the same seed on different minima.
# The sc-n12 GMIN rows reproduce with OpenBLAS.
mkdir -p "$WORK/bin"
if [[ $OLD_BUILD_KEY != "$BUILD_KEY" || ! -x $WORK/bin/GMIN ]]; then
  rm -rf "$SRC"
  rm -f "$WORK/bin/GMIN" "$BUILD_KEY_FILE"
  mkdir -p "$SRC"
  (cd "$DARBY_WORK/src" && tar --exclude=./wales/GMIN/build -cf - ./wales) | tar -xf - -C "$SRC"
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$TIME"
  cp "$CAP_SOURCE" "$SRC/wales/GMIN/source/h2h_cap.f90"
  build=$SRC/wales/GMIN/build/openblas
  mkdir -p "$build"
  (
    cd "$build"
    FC=$FC_BIN CC=$CC_BIN CXX=$CXX_BIN cmake ../../source "${CMAKE_FLAGS[@]}"
    make -j"$(nproc)" GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN"
  chmod +x "$WORK/bin/GMIN"
  printf '%s\n' "$BUILD_KEY" > "$BUILD_KEY_FILE"
fi
blas=$(ldd "$WORK/bin/GMIN" | awk '/libopenblas/ {print $3}')
if [[ -z $blas ]]; then
  echo "GMIN is not linked against OpenBLAS" >&2
  exit 1
fi
cat > "$WORK/gmin-source.txt" << EOF
url=$GMIN_PIN_URL
revision=$revision
svnrev=$revision
tarball_sha256=$tarball_sha256
build_key=$BUILD_KEY
potential=SC
cmake=${CMAKE_FLAGS[*]}
gfortran=$GFORTRAN_VER
blas=$(readlink -f "$blas")
EOF
echo "gmin build_key=$BUILD_KEY binary=$WORK/bin/GMIN"
