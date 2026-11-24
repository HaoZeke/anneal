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
FFLAGS_VAL=${FFLAGS:-}
FCFLAGS_VAL=${FCFLAGS:-}
CFLAGS_VAL=${CFLAGS:-}
CXXFLAGS_VAL=${CXXFLAGS:-}
CPPFLAGS_VAL=${CPPFLAGS:-}
LDFLAGS_VAL=${LDFLAGS:-}
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
  "--option=FFLAGS=$FFLAGS_VAL"
  "--option=FCFLAGS=$FCFLAGS_VAL"
  "--option=CFLAGS=$CFLAGS_VAL"
  "--option=CXXFLAGS=$CXXFLAGS_VAL"
  "--option=CPPFLAGS=$CPPFLAGS_VAL"
  "--option=LDFLAGS=$LDFLAGS_VAL"
)
for flag in "${CMAKE_FLAGS[@]}"; do
  KEY_ARGS+=("--option=$flag")
done

# Basin-hopping is chaotic in the last bit, so the BLAS GMIN links is part of
# the run: reference BLAS and OpenBLAS end the same seed on different minima.
# The sc-n12 GMIN rows reproduce with OpenBLAS.
# The BLAS is the one the binary loads, so it is found from the binary and
# folded into the build key before the key is compared or recorded.
discover_blas() {
  local found
  found=$(ldd "$1" 2>/dev/null | awk '/libopenblas/ && !seen++ {print $3}')
  [[ -n $found ]] || return 1
  BLAS=$(readlink -f "$found")
  [[ -f $BLAS ]] || return 1
  BLAS_SHA256=$(sha256sum "$BLAS" | awk '{print $1}')
}
compute_build_key() {
  python3 "$PIN" build-key "${KEY_ARGS[@]}" \
    "--option=blas=$BLAS" \
    "--option=blas_sha256=$BLAS_SHA256"
}

mkdir -p "$WORK/bin"
BLAS=
BLAS_SHA256=
BUILD_KEY=
need_rebuild=1
if [[ -x $WORK/bin/GMIN ]] && discover_blas "$WORK/bin/GMIN"; then
  BUILD_KEY=$(compute_build_key)
  OLD_BUILD_KEY=$(
    python3 "$PIN" read-build-key \
      --metadata "$WORK/gmin-source.txt" \
      --key-file "$BUILD_KEY_FILE"
  )
  if [[ $OLD_BUILD_KEY == "$BUILD_KEY" ]]; then
    need_rebuild=0
  fi
fi
if (( need_rebuild )); then
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
    FC=$FC_BIN CC=$CC_BIN CXX=$CXX_BIN \
      FFLAGS=$FFLAGS_VAL FCFLAGS=$FCFLAGS_VAL \
      CFLAGS=$CFLAGS_VAL CXXFLAGS=$CXXFLAGS_VAL \
      CPPFLAGS=$CPPFLAGS_VAL LDFLAGS=$LDFLAGS_VAL \
      cmake ../../source "${CMAKE_FLAGS[@]}"
    make -j"$(nproc)" GMIN
  )
  cp -a "$build/GMIN" "$WORK/bin/GMIN"
  chmod +x "$WORK/bin/GMIN"
  if ! discover_blas "$WORK/bin/GMIN"; then
    echo "GMIN is not linked against OpenBLAS" >&2
    exit 1
  fi
  BUILD_KEY=$(compute_build_key)
  printf '%s\n' "$BUILD_KEY" > "$BUILD_KEY_FILE"
fi
{
  echo "url=$GMIN_PIN_URL"
  echo "revision=$revision"
  echo "svnrev=$revision"
  echo "tarball_sha256=$tarball_sha256"
  echo "build_key=$BUILD_KEY"
  echo "potential=SC"
  echo "cap=${CAP_SOURCE#"$ROOT"/} $(sha256sum "$CAP_SOURCE" | awk '{print $1}')"
  echo "patch=${CAP#"$ROOT"/} $(sha256sum "$CAP" | awk '{print $1}')"
  echo "patch=${DARBY_TIME#"$ROOT"/} $(sha256sum "$DARBY_TIME" | awk '{print $1}')"
  echo "patch=${TIME#"$ROOT"/} $(sha256sum "$TIME" | awk '{print $1}')"
  echo "cmake=${CMAKE_FLAGS[*]}"
  echo "fc=$FC_BIN"
  echo "cc=$CC_BIN"
  echo "cxx=$CXX_BIN"
  echo "gfortran=$GFORTRAN_VER"
  echo "fflags=$FFLAGS_VAL"
  echo "fcflags=$FCFLAGS_VAL"
  echo "cflags=$CFLAGS_VAL"
  echo "cxxflags=$CXXFLAGS_VAL"
  echo "cppflags=$CPPFLAGS_VAL"
  echo "ldflags=$LDFLAGS_VAL"
  echo "blas=$BLAS"
  echo "blas_sha256=$BLAS_SHA256"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | awk '{print $1}')"
} > "$WORK/gmin-source.txt"
echo "gmin build_key=$BUILD_KEY binary=$WORK/bin/GMIN"
