#!/usr/bin/env bash
# Copyright (c) anneal developers
# One serial GMIN for the whole panel: the 2026-11-19 snapshot of the
# public tarball, pinned by SVNREV and sha256, the evaluator cap with the
# Lennard-Jones timer, and the timers inside Morse and binary Gupta.
# ALIGN, MPI and OpenMP stay off.
#
#   build_gmin.sh [--tarball PATH]
#
# PATH or GMIN_TARBALL is a local copy of that snapshot. Its sha256 is
# checked before the file is extracted or built. With neither set, the
# public URL is fetched and refused unless the sha256 matches.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
WORK=${GMIN_PANEL_WORK:-/work/gmin-panel}
SRC=$WORK/gmin-src
TARBALL=$WORK/svn.tar.bz2
URL=$GMIN_PIN_URL
SNAPSHOT=$GMIN_PIN_SNAPSHOT
REV=$GMIN_PIN_REVISION
TARBALL_SHA256=$GMIN_PIN_TARBALL_SHA256
PATCHES=(
  "$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch"
  "$ROOT/scripts/morse_h2h/morse-timer.patch"
  "$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch"
)

LOCAL=${GMIN_TARBALL:-}
while [[ $# -gt 0 ]]; do
  case $1 in
    --tarball)
      if [[ $# -lt 2 ]]; then
        echo "missing path after --tarball" >&2
        exit 2
      fi
      LOCAL=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

tarball_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

refuse_sha_mismatch() {
  local path=$1
  python3 "$PIN" verify-tarball "$path" >/dev/null
}

sha_matches() {
  [[ -s $1 ]] && python3 "$PIN" verify-tarball "$1" >/dev/null 2>&1
}

# A caller-supplied file is checked before any cache, download, or extract.
if [[ -n $LOCAL ]]; then
  if [[ ! -s $LOCAL ]]; then
    echo "tarball ${LOCAL} is missing" >&2
    exit 1
  fi
  refuse_sha_mismatch "$LOCAL"
fi

mkdir -p "$WORK/bin" "$SRC"
if [[ -n $LOCAL ]]; then
  if [[ $LOCAL != "$TARBALL" ]]; then
    cp -a "$LOCAL" "$TARBALL"
  fi
  refuse_sha_mismatch "$TARBALL"
elif sha_matches "$TARBALL"; then
  :
elif sha_matches /work/svn.tar.bz2; then
  cp -a /work/svn.tar.bz2 "$TARBALL"
  refuse_sha_mismatch "$TARBALL"
else
  tmp=$(mktemp "$WORK/svn.tar.bz2.partial.XXXXXX")
  curl -fL --retry 3 -o "$tmp" "$URL"
  if ! sha_matches "$tmp"; then
    got=$(tarball_sha256 "$tmp")
    rm -f "$tmp"
    echo "tarball sha256 mismatch: got ${got}, expected ${TARBALL_SHA256}" >&2
    exit 1
  fi
  mv "$tmp" "$TARBALL"
fi

extract_tree() {
  mkdir -p "$SRC"
  tar -xjf "$TARBALL" -C "$SRC" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
}

if [[ ! -f $SRC/wales/GMIN/source/SVNREV ]]; then
  extract_tree
fi
got=$(python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV" 2>/dev/null || true)
if [[ $got != "$REV" ]]; then
  rm -rf "$SRC"
  rm -f "$WORK/bin/GMIN"
  extract_tree
  got=$(python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV" 2>/dev/null || true)
fi
if [[ $got != "$REV" ]]; then
  python3 "$PIN" read-svnrev "$SRC/wales/GMIN/source/SVNREV" >/dev/null
  exit 1
fi

if [[ ! -f $SRC/wales/GMIN/source/h2h_cap.f90 ]]; then
  for patch in "${PATCHES[@]}"; do
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
  echo "snapshot=$SNAPSHOT"
  echo "url=$URL"
  echo "revision=$REV"
  echo "svnrev=$got"
  echo "tarball_sha256=$(tarball_sha256 "$TARBALL")"
  for patch in "${PATCHES[@]}"; do
    echo "patch=${patch#"$ROOT"/} $(sha256sum "$patch" | awk '{print $1}')"
  done
  echo "cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
  echo "gfortran=$(gfortran -dumpfullversion)"
  echo "linked=$(ldd "$WORK/bin/GMIN" | awk '/blas|lapack/ {print $3}' | xargs -r readlink -f | paste -sd' ' -)"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | awk '{print $1}')"
} > "$WORK/gmin-source.txt"
echo "gmin snapshot=$SNAPSHOT revision=$REV binary=$WORK/bin/GMIN"
