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
# pinned dataset object is fetched. The rolling public URL is only a
# fallback, and either download is refused unless the sha256 matches.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=$ROOT/scripts/gmin_pin.py
eval "$(python3 "$PIN" shell)"
WORK=${GMIN_PANEL_WORK:-/work/gmin-panel}
SRC=$WORK/gmin-src
TARBALL=$WORK/svn.tar.bz2
SNAPSHOT=$GMIN_PIN_SNAPSHOT
REV=$GMIN_PIN_REVISION
TARBALL_SHA256=$GMIN_PIN_TARBALL_SHA256
HF_REVISION=2d639566b6e9a7243bfd485aa3cb9a904c915451
HF_URL=https://huggingface.co/datasets/HaoZeke/anneal-paper-data/resolve/${HF_REVISION}/gmin/svn-20261119.tar.bz2
WALES_URL=$GMIN_PIN_URL
# Newest member mtime in the pinned archive.
SOURCE_DATE_EPOCH=1795057278
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

tmp=
cleanup_partial() {
  if [[ -n $tmp ]]; then
    rm -f "$tmp"
  fi
}
trap cleanup_partial EXIT

# A caller-supplied file is checked before any cache, download, or extract.
if [[ -n $LOCAL ]]; then
  if [[ ! -s $LOCAL ]]; then
    echo "tarball ${LOCAL} is missing" >&2
    exit 1
  fi
  refuse_sha_mismatch "$LOCAL"
fi

mkdir -p "$WORK/bin" "$SRC"
WORK=$(cd "$WORK" && pwd -P)
SRC=$WORK/gmin-src
TARBALL=$WORK/svn.tar.bz2
if [[ -n $LOCAL ]]; then
  if ! test "$LOCAL" -ef "$TARBALL"; then
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
  if ! curl -fL --retry 3 -o "$tmp" "$HF_URL"; then
    rm -f "$tmp"
    tmp=$(mktemp "$WORK/svn.tar.bz2.partial.XXXXXX")
    echo "pinned dataset download failed; trying rolling Wales URL" >&2
    curl -fL --retry 3 -o "$tmp" "$WALES_URL"
  fi
  refuse_sha_mismatch "$tmp"
  mv "$tmp" "$TARBALL"
  tmp=
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
  stamp_bin=$WORK/stamp-tools
  mkdir -p "$stamp_bin"
  cat > "$stamp_bin/date" <<'EOF'
#!/usr/bin/env bash
if [[ $# -eq 0 ]]; then
  LC_ALL=C exec /usr/bin/date -u -d "@${SOURCE_DATE_EPOCH}"
fi
exec /usr/bin/date "$@"
EOF
  cat > "$stamp_bin/uname" <<'EOF'
#!/usr/bin/env bash
case ${1:-} in
  -o) echo linux ;;
  -m) echo x86_64 ;;
  *) exec /usr/bin/uname "$@" ;;
esac
EOF
  cat > "$stamp_bin/gfortran" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=()
for arg in "$@"; do
  case $arg in
    "$GMIN_BUILD_SRC"/*)
      arg=$(realpath --relative-to="$PWD" "$arg")
      ;;
    -I"$GMIN_BUILD_SRC"/*)
      include=${arg#-I}
      arg=-I$(realpath --relative-to="$PWD" "$include")
      ;;
    -D_SVN_ROOT_=*)
      relative_src=$(realpath --relative-to="$PWD" "$GMIN_BUILD_SRC")
      arg=${arg/"$GMIN_BUILD_SRC"/$relative_src}
      ;;
  esac
  args+=("$arg")
done
exec /usr/bin/gfortran "${args[@]}"
EOF
  chmod +x "$stamp_bin/date" "$stamp_bin/uname" "$stamp_bin/gfortran"
  prefix_maps="-ffile-prefix-map=$SRC=/usr/src/gmin -fdebug-prefix-map=$SRC=/usr/src/gmin"
  prefix_maps+=" -ffile-prefix-map=$WORK=/build/gmin -fdebug-prefix-map=$WORK=/build/gmin"
  mkdir -p "$build"
  (
    cd "$build"
    unset CFLAGS CXXFLAGS FFLAGS FCFLAGS LDFLAGS
    export LC_ALL=C
    export SOURCE_DATE_EPOCH
    export USER=builder
    export HOSTNAME=gmin-panel
    export PATH="$stamp_bin:$PATH"
    export GMIN_BUILD_SRC=$SRC
    FC="$stamp_bin/gfortran" CC=gcc CXX=g++ cmake ../../source \
      -DCMAKE_C_FLAGS="$prefix_maps" \
      -DCMAKE_CXX_FLAGS="$prefix_maps" \
      -DCMAKE_Fortran_FLAGS="$prefix_maps" \
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
  echo "url=$HF_URL"
  echo "fallback_url=$WALES_URL"
  echo "revision=$REV"
  echo "svnrev=$got"
  echo "tarball_sha256=$(tarball_sha256 "$TARBALL")"
  echo "source_date_epoch=$SOURCE_DATE_EPOCH"
  for patch in "${PATCHES[@]}"; do
    echo "patch=${patch#"$ROOT"/} $(sha256sum "$patch" | awk '{print $1}')"
  done
  echo "cmake=-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF -DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
  echo "gfortran=$(gfortran -dumpfullversion)"
  echo "linked=$(ldd "$WORK/bin/GMIN" | awk '/blas|lapack/ {print $3}' | xargs -r readlink -f | paste -sd' ' -)"
  echo "binary_sha256=$(sha256sum "$WORK/bin/GMIN" | awk '{print $1}')"
} > "$WORK/gmin-source.txt"
echo "gmin snapshot=$SNAPSHOT revision=$REV binary=$WORK/bin/GMIN"
