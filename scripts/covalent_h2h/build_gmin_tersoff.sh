#!/usr/bin/env bash
# Copyright (c) anneal developers
# A second GMIN binary whose USERPOT is rgpot's Tersoff carbon.
# The public GMIN has no Tersoff keyword. Its USERPOT keyword calls the
# userpot_* routines, which a standard build takes from dummy_userpot.f90.
# This build copies the capped source tree, puts tersoff_userpot.f90 in
# that file's place and links librgpot_covalent.so.
#
#   build_gmin_tersoff.sh [--tarball PATH] <librgpot_covalent.so>
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
WORK=${GMIN_H2H_WORK:-/work/gmin-h2h}
TREE=$WORK/tersoff
BUILD_KEY_FILE=$WORK/gmin-tersoff-build-key
SOURCE_NOTE=$WORK/gmin-tersoff-source.txt
CAP_SOURCE=$ROOT/bench/lj104-lattice/v2/h2h_cap.f90
PATCH=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
SW_PATCH=$HERE/sw-timer.patch
USERPOT=$HERE/tersoff_userpot.f90

FORWARD_ARGS=()
while [[ $# -gt 1 ]]; do
  case $1 in
    --tarball) FORWARD_ARGS+=(--tarball "${2:?missing path after --tarball}"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
LIB=$(realpath "${1:?librgpot_covalent.so}")
if [[ ! -f $LIB ]]; then
  echo "missing library: $LIB" >&2
  exit 1
fi

GMIN_H2H_WORK=$WORK bash "$HERE/build_gmin.sh" "${FORWARD_ARGS[@]}"
got=$(python3 "$PIN" read-svnrev "$WORK/src/wales/GMIN/source/SVNREV")
TARBALL_SHA256=$(python3 "$PIN" verify-tarball "$WORK/svn.tar.bz2")

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
CMAKE_FLAGS=(
  -DWITH_ALIGN=OFF
  -DWITH_MPI=OFF
  -DWITH_OPENMP=OFF
  -DWITH_SPARSE=OFF
  -DWITH_PYTHON=OFF
  -DWITH_MYBLAS=OFF
  -DWITH_MYLAPACK=OFF
  "-DCMAKE_Fortran_STANDARD_LIBRARIES=$LIB"
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$(dirname "$LIB")"
)
KEY_ARGS=(
  --revision "$got"
  --tarball-sha256 "$TARBALL_SHA256"
  --input "$CAP_SOURCE"
  --input "$PATCH"
  --input "$SW_PATCH"
  --input "$USERPOT"
  --input "$LIB"
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
BUILD_KEY=$(python3 "$PIN" build-key "${KEY_ARGS[@]}")
OLD_BUILD_KEY=$(
  python3 "$PIN" read-build-key \
    --metadata "$SOURCE_NOTE" \
    --key-file "$BUILD_KEY_FILE"
)

if [[ $OLD_BUILD_KEY != "$BUILD_KEY" || ! -x $WORK/bin/GMIN-tersoff ]]; then
  rm -rf "$TREE"
  rm -f "$WORK/bin/GMIN-tersoff" "$BUILD_KEY_FILE"
  mkdir -p "$TREE"
  (cd "$WORK/src" && tar --exclude=./wales/GMIN/build -cf - ./wales) | tar -xf - -C "$TREE"
  cp "$USERPOT" "$TREE/wales/GMIN/source/dummy_userpot.f90"
  build=$TREE/wales/GMIN/build/gfortran
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
  cp -a "$build/GMIN" "$WORK/bin/GMIN-tersoff"
  chmod +x "$WORK/bin/GMIN-tersoff"
  printf '%s\n' "$BUILD_KEY" > "$BUILD_KEY_FILE"
fi

{
  echo "url=$GMIN_PIN_URL"
  echo "revision=$GMIN_PIN_REVISION"
  echo "svnrev=$got"
  echo "tarball_sha256=$TARBALL_SHA256"
  echo "build_key=$BUILD_KEY"
  echo "cap=${CAP_SOURCE#"$ROOT"/} $(sha256sum "$CAP_SOURCE" | awk '{print $1}')"
  echo "patch=${PATCH#"$ROOT"/} $(sha256sum "$PATCH" | awk '{print $1}')"
  echo "patch=${SW_PATCH#"$ROOT"/} $(sha256sum "$SW_PATCH" | awk '{print $1}')"
  echo "userpot=${USERPOT#"$ROOT"/} $(sha256sum "$USERPOT" | awk '{print $1}')"
  echo "library=$LIB $(sha256sum "$LIB" | awk '{print $1}')"
  echo "cmake=${CMAKE_FLAGS[*]}"
  echo "gfortran=$GFORTRAN_VER"
  echo "fflags=$FFLAGS_VAL"
  echo "cflags=$CFLAGS_VAL"
  echo "cxxflags=$CXXFLAGS_VAL"
  echo "ldflags=$LDFLAGS_VAL"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN-tersoff" | awk '{print $1}')"
} > "$SOURCE_NOTE"
echo "gmin-tersoff build_key=$BUILD_KEY binary=$WORK/bin/GMIN-tersoff library=$LIB"
