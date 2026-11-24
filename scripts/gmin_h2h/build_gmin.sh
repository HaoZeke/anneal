#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build Wales's public GMIN and the in-tree evaluator cap.
# The public drop is the group tarball. GMIN/source/SVNREV is the revision.
# ALIGN and OpenMP stay off: the Lennard-Jones comparison is serial, and
# the ALIGN build pulls a separate FFTW tree this run does not use.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
WORK=${GMIN_H2H_WORK:-/work/gmin-h2h}
SRC=$WORK/src
TARBALL=$WORK/svn.tar.bz2
URL=$GMIN_PIN_URL
REV=$GMIN_PIN_REVISION
PATCH=$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch
CAP_SOURCE=$ROOT/bench/lj104-lattice/v2/h2h_cap.f90
BUILD_KEY_FILE=$WORK/gmin-build-key
CMAKE_FLAGS=(
  -DWITH_ALIGN=OFF
  -DWITH_MPI=OFF
  -DWITH_OPENMP=OFF
  -DWITH_SPARSE=OFF
  -DWITH_PYTHON=OFF
  -DWITH_MYBLAS=OFF
  -DWITH_MYLAPACK=OFF
)

LOCAL=${GMIN_TARBALL:-}
EXTRA_PATCHES=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --tarball) LOCAL=${2:?missing path after --tarball}; shift 2 ;;
    --extra-patch) EXTRA_PATCHES+=("${2:?missing path after --extra-patch}"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n $LOCAL ]]; then
  if [[ ! -s $LOCAL ]]; then
    echo "tarball ${LOCAL} is missing" >&2
    exit 1
  fi
  python3 "$PIN" verify-tarball "$LOCAL" >/dev/null
fi

mkdir -p "$WORK/bin" "$SRC"
if [[ -n $LOCAL ]]; then
  if [[ $LOCAL != "$TARBALL" ]]; then
    cp -a "$LOCAL" "$TARBALL"
  fi
  python3 "$PIN" verify-tarball "$TARBALL" >/dev/null
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
TARBALL_SHA256=$(python3 "$PIN" verify-tarball "$TARBALL")

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
  --revision "$REV"
  --tarball-sha256 "$TARBALL_SHA256"
  --input "$CAP_SOURCE"
  --input "$PATCH"
)
for extra in "${EXTRA_PATCHES[@]}"; do
  KEY_ARGS+=(--input "$extra")
done
KEY_ARGS+=(
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
    --metadata "$WORK/gmin-source.txt" \
    --key-file "$BUILD_KEY_FILE"
)

need_rebuild=0
if [[ $OLD_BUILD_KEY != "$BUILD_KEY" || ! -x $WORK/bin/GMIN || ! -f $SRC/wales/GMIN/source/SVNREV ]]; then
  need_rebuild=1
fi
got=$(python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV" 2>/dev/null || true)
if [[ $got != "$REV" ]]; then
  need_rebuild=1
fi

if (( need_rebuild )); then
  rm -rf "$SRC/wales"
  rm -f "$WORK/bin/GMIN" "$BUILD_KEY_FILE"
  tar -xjf "$TARBALL" -C "$SRC" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
  got=$(python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV")
  patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$PATCH"
  for extra in "${EXTRA_PATCHES[@]}"; do
    patch -p1 --forward --batch -d "$SRC/wales/GMIN" < "$extra"
  done
  cp "$CAP_SOURCE" "$SRC/wales/GMIN/source/h2h_cap.f90"
  build=$SRC/wales/GMIN/build/gfortran
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
  printf '%s\n' "$BUILD_KEY" > "$BUILD_KEY_FILE"
fi

{
  echo "url=$URL"
  echo "revision=$REV"
  echo "svnrev=$got"
  echo "tarball_sha256=$TARBALL_SHA256"
  echo "build_key=$BUILD_KEY"
  echo "cap=${CAP_SOURCE#"$ROOT"/} $(sha256sum "$CAP_SOURCE" | awk '{print $1}')"
  echo "patch=${PATCH#"$ROOT"/} $(sha256sum "$PATCH" | awk '{print $1}')"
  for extra in "${EXTRA_PATCHES[@]}"; do
    echo "patch=${extra#"$ROOT"/} $(sha256sum "$extra" | awk '{print $1}')"
  done
  echo "cmake=${CMAKE_FLAGS[*]}"
  echo "gfortran=$GFORTRAN_VER"
  echo "fflags=$FFLAGS_VAL"
  echo "cflags=$CFLAGS_VAL"
  echo "cxxflags=$CXXFLAGS_VAL"
  echo "ldflags=$LDFLAGS_VAL"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | awk '{print $1}')"
} > "$WORK/gmin-source.txt"
echo "gmin url=$URL revision=$REV build_key=$BUILD_KEY binary=$WORK/bin/GMIN"
