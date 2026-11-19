#!/usr/bin/env bash
# Copyright (c) anneal developers
# Rebuild the 2026-11-19 GMIN pin the way scripts/gmin_panel/build_gmin.sh
# does at eeec8e2c, and report why the binary sha256 moves.
#
#   check_gmin_build.sh --tarball PATH
#
# Two absolute work directories, then the first directory again after
# 120 seconds. Section hashes use objcopy -O binary -j. A later pair
# adds SOURCE_DATE_EPOCH, -ffile-prefix-map=$PWD=. and strip -s. The
# last pair also freezes the clock and the account line that
# display_version.sh writes.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REV=9cbec1a446352bc9ef064bd2b1a5767194655919
TARBALL_SHA256=ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776
EPOCH=1763510400
PATCHES=(
  "$ROOT/bench/lj104-lattice/v2/h2h-gmin.patch"
  "$ROOT/scripts/morse_h2h/morse-timer.patch"
  "$ROOT/bench/metal-cluster-h2h/gmin-bgupta.patch"
)
SECTIONS=(.text .rodata .data .data.rel.ro .eh_frame .eh_frame_hdr)

TARBALL=
while [[ $# -gt 0 ]]; do
  case $1 in
    --tarball)
      TARBALL=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -z $TARBALL || ! -s $TARBALL ]]; then
  echo "pass --tarball PATH" >&2
  exit 2
fi

got=$(sha256sum "$TARBALL" | awk '{print $1}')
if [[ $got != "$TARBALL_SHA256" ]]; then
  echo "tarball sha256 mismatch: got ${got}, expected ${TARBALL_SHA256}" >&2
  exit 1
fi

BASE=$(mktemp -d)
trap 'rm -rf "$BASE"' EXIT
A=$BASE/gmin-work-aaa
B=$BASE/gmin-work-bbb
S=$BASE/gmin-work-sss
# Compiler wrappers and the date wrapper stay in different directories.
# The flag pair must see the wall-clock date. GNU date ignores
# SOURCE_DATE_EPOCH, and so does gfortran 13.3.0.
mkdir -p "$BASE/cc" "$BASE/datebin"
cat > "$BASE/datebin/date" << 'EOF'
#!/bin/sh
export LC_ALL=C
if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
  exec /bin/date -u -d "@${SOURCE_DATE_EPOCH}" "$@"
fi
exec /bin/date "$@"
EOF
cat > "$BASE/cc/gfortran" << 'EOF'
#!/bin/sh
exec /usr/bin/gfortran -ffile-prefix-map="$(pwd)=." "$@"
EOF
cat > "$BASE/cc/gcc" << 'EOF'
#!/bin/sh
exec /usr/bin/gcc -ffile-prefix-map="$(pwd)=." "$@"
EOF
cat > "$BASE/cc/g++" << 'EOF'
#!/bin/sh
exec /usr/bin/g++ -ffile-prefix-map="$(pwd)=." "$@"
EOF
chmod +x "$BASE/datebin/date" "$BASE/cc/gfortran" "$BASE/cc/gcc" "$BASE/cc/g++"

extract_tree() {
  local work=$1
  local src=$work/gmin-src
  mkdir -p "$src"
  tar -xjf "$TARBALL" -C "$src" \
    ./wales/GMIN \
    ./wales/CMakeModules \
    ./wales/INTERFACES \
    ./wales/SCRIPTS/all \
    ./wales/HiRE
  local rev
  rev=$(tr -d '[:space:]' < "$src/wales/GMIN/source/SVNREV")
  if [[ $rev != "$REV" ]]; then
    echo "GMIN SVNREV is $rev, expected $REV" >&2
    exit 1
  fi
  if [[ ! -f $src/wales/GMIN/source/h2h_cap.f90 ]]; then
    local patch
    for patch in "${PATCHES[@]}"; do
      patch -p1 --forward --batch -d "$src/wales/GMIN" < "$patch"
    done
  fi
}

compile_gmin() {
  local work=$1
  local mode=$2
  local src=$work/gmin-src
  local build=$src/wales/GMIN/build/gfortran
  if [[ ! -f $src/wales/GMIN/source/SVNREV ]]; then
    extract_tree "$work"
  fi
  rm -rf "$build" "$work/bin/GMIN"
  mkdir -p "$build" "$work/bin"
  (
    cd "$build"
    case $mode in
      plain)
        export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ;;
      flags)
        # Prefix map at compile time ($PWD is the build directory).
        # The real date stays on PATH.
        export SOURCE_DATE_EPOCH=$EPOCH
        export PATH="$BASE/cc:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ;;
      fixed)
        export SOURCE_DATE_EPOCH=$EPOCH
        export PATH="$BASE/datebin:$BASE/cc:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        export USER=repro
        export HOSTNAME=repro
        export LC_ALL=C
        ;;
      *)
        echo "unknown mode $mode" >&2
        exit 2
        ;;
    esac
    hash -r
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
  cp -f "$build/GMIN" "$work/bin/GMIN"
  chmod +x "$work/bin/GMIN"
}

section_file() {
  local bin=$1
  local sec=$2
  local dest=$3
  if ! objcopy -O binary -j "$sec" "$bin" "$dest" 2>/dev/null; then
    return 1
  fi
  [[ -s $dest ]]
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

echo "tarball $TARBALL_SHA256"
echo "svnrev $REV"
echo "gfortran $(gfortran -dumpfullversion)"

echo "plain A"
compile_gmin "$A" plain
cp -f "$A/bin/GMIN" "$BASE/GMIN-a1"
echo "a1 $(sha_file "$BASE/GMIN-a1")"

echo "plain B"
compile_gmin "$B" plain
cp -f "$B/bin/GMIN" "$BASE/GMIN-b1"
echo "b1 $(sha_file "$BASE/GMIN-b1")"

echo "wait 120"
sleep 120

echo "plain A again"
compile_gmin "$A" plain
cp -f "$A/bin/GMIN" "$BASE/GMIN-a2"
echo "a2 $(sha_file "$BASE/GMIN-a2")"

declare -A SEC_SHA
for name in a1 b1 a2; do
  for sec in "${SECTIONS[@]}"; do
    tmp=$BASE/${name}${sec}.bin
    if section_file "$BASE/GMIN-$name" "$sec" "$tmp"; then
      SEC_SHA["$name$sec"]=$(sha_file "$tmp")
    else
      SEC_SHA["$name$sec"]=absent
    fi
  done
done

echo "section hashes"
for sec in "${SECTIONS[@]}"; do
  h1=${SEC_SHA[a1$sec]}
  h2=${SEC_SHA[b1$sec]}
  h3=${SEC_SHA[a2$sec]}
  if [[ $h1 == "$h2" && $h1 == "$h3" ]]; then
    verdict=identical
  else
    verdict=DIFFERS
  fi
  echo "$sec $verdict a1=$h1 b1=$h2 a2=$h3"
done

echo "comment"
for name in a1 b1 a2; do
  echo -n "$name "
  readelf -p .comment "$BASE/GMIN-$name" | awk '/GCC/'
done

debug_n=$(readelf -S "$BASE/GMIN-a1" | grep -c '\.debug' || true)
echo "debug_sections $debug_n"

python3 - "$BASE" << 'PY'
import sys
base = sys.argv[1]
names = ["a1", "b1", "a2"]
text = {}
import subprocess
for name in names:
    lines = subprocess.check_output(["strings", "-a", f"{base}/GMIN-{name}"], text=True, errors="replace").splitlines()
    text[name] = set(lines)

def cls(s):
    if "Compilation time:" in s:
        return "clock"
    if "Compiled by " in s:
        return "account"
    if "/gmin-work-" in s or s.startswith("At line") or s.startswith("In file"):
        return "path"
    return "other"

for left, right in (("a1", "a2"), ("a1", "b1")):
    only_l = text[left] - text[right]
    only_r = text[right] - text[left]
    from collections import Counter
    print(f"strings {left}-vs-{right} only_{left} {dict(Counter(cls(s) for s in only_l))} only_{right} {dict(Counter(cls(s) for s in only_r))}")
    clocks = sorted(s for s in only_l | only_r if cls(s) == "clock")
    for s in clocks:
        print(" ", s)
PY

echo "flags same directory"
compile_gmin "$S" flags
cp -f "$S/bin/GMIN" "$BASE/GMIN-f1"
strip -s -o "$BASE/GMIN-f1.strip" "$BASE/GMIN-f1"
compile_gmin "$S" flags
cp -f "$S/bin/GMIN" "$BASE/GMIN-f2"
strip -s -o "$BASE/GMIN-f2.strip" "$BASE/GMIN-f2"
echo "f1 $(sha_file "$BASE/GMIN-f1")"
echo "f2 $(sha_file "$BASE/GMIN-f2")"
echo "f1.strip $(sha_file "$BASE/GMIN-f1.strip")"
echo "f2.strip $(sha_file "$BASE/GMIN-f2.strip")"
if [[ $(sha_file "$BASE/GMIN-f1.strip") == $(sha_file "$BASE/GMIN-f2.strip") ]]; then
  echo "flags_strip stable"
else
  echo "flags_strip not_stable"
fi

echo "fixed clock and account, same directory"
compile_gmin "$S" fixed
cp -f "$S/bin/GMIN" "$BASE/GMIN-g1"
strip -s -o "$BASE/GMIN-g1.strip" "$BASE/GMIN-g1"
compile_gmin "$S" fixed
cp -f "$S/bin/GMIN" "$BASE/GMIN-g2"
strip -s -o "$BASE/GMIN-g2.strip" "$BASE/GMIN-g2"
echo "g1 $(sha_file "$BASE/GMIN-g1")"
echo "g2 $(sha_file "$BASE/GMIN-g2")"
echo "g1.strip $(sha_file "$BASE/GMIN-g1.strip")"
echo "g2.strip $(sha_file "$BASE/GMIN-g2.strip")"
if [[ $(sha_file "$BASE/GMIN-g1") == $(sha_file "$BASE/GMIN-g2") ]]; then
  echo "fixed_unstripped stable"
else
  echo "fixed_unstripped not_stable"
fi
