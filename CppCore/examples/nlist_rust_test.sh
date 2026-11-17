#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
# Compile the hand-written Rust caller against the neighbour archives.
# Arguments after the build directory link the OpenMP runtime the archives
# were compiled for. Without them -fopenmp is tried, as GCC and Clang take
# it; AppleClang does not, so CMake passes the libraries FindOpenMP chose.
set -eu
root=$1
build=$2
shift 2
rustc_bin=rustc
if ! "$rustc_bin" --version >/dev/null 2>&1; then
  rustc_bin=$(find /usr/local/rustup/toolchains "${RUSTUP_HOME:-/usr/local/rustup}/toolchains" \
    -path '*/bin/rustc' -type f -print -quit 2>/dev/null || true)
fi
if [ -z "$rustc_bin" ]; then
  echo "rustc is not available" >&2
  exit 1
fi
find_archive() {
  find "$build" \( -name "$1" -o -name "$2" -o -name "$3" \) -print -quit
}
half=$(find_archive librgpot_halfshell.a librgpot_halfshell.lib rgpot_halfshell.lib)
mini=$(find_archive librgpot_minimage.a librgpot_minimage.lib rgpot_minimage.lib)
fma=$(find_archive librgpot_exact_fma.a librgpot_exact_fma.lib rgpot_exact_fma.lib)
if [ -z "$half" ] || [ -z "$mini" ] || [ -z "$fma" ]; then
  echo "neighbour archives were not built" >&2
  exit 1
fi
cxx=${CXX:-c++}
os=$(uname -s)
if [ "$#" -eq 0 ] && "$cxx" -fopenmp -x c++ -E - >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
then
  set -- -fopenmp
fi
# Coverage builds instrument the archives. The shared library has to pull
# the gcov runtime or the loader cannot resolve __gcov_init.
gcov_flag=
if nm "$half" 2>/dev/null | grep -q __gcov_init; then
  gcov_flag=--coverage
fi
dir=$(mktemp -d)
case "$os" in
  Darwin)
    "$cxx" -shared -o "$dir/libnlist.dylib" \
      -Wl,-force_load,"$half" -Wl,-force_load,"$mini" -Wl,-force_load,"$fma" \
      "$@" $gcov_flag
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # cl rejects -Wl,--whole-archive. link.exe pulls every object; the
    # import library is nlist.lib, which rustc's -lnlist finds.
    # Path conversion stays on for rustc. Only the link.exe line opts out,
    # or Git bash rewrites /WHOLEARCHIVE. rustc is a Windows binary and
    # creates its metadata dir from TMP; an MSYS /tmp does not exist there.
    whalf=$(cygpath -w "$half")
    wmini=$(cygpath -w "$mini")
    wfma=$(cygpath -w "$fma")
    wdir=$(cygpath -w "$dir")
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
      link.exe /nologo /DLL "/OUT:${wdir}\\nlist.dll" \
      "/WHOLEARCHIVE:${whalf}" "/WHOLEARCHIVE:${wmini}" "/WHOLEARCHIVE:${wfma}" \
      /EXPORT:rgpot_nlist_build \
      /EXPORT:rgpot_nlist_request_init \
      /EXPORT:rgpot_nlist_strerror
    if [ ! -f "$dir/nlist.lib" ]; then
      echo "link.exe did not write nlist.lib" >&2
      ls -la "$dir" >&2 || true
      exit 1
    fi
    TMP="$wdir"
    TEMP="$wdir"
    TMPDIR="$wdir"
    export TMP TEMP TMPDIR
    ;;
  *)
    "$cxx" -shared -o "$dir/libnlist.so" -Wl,--whole-archive "$half" "$mini" "$fma" \
      -Wl,--no-whole-archive "$@" $gcov_flag
    ;;
esac
out=$(mktemp)
case "$os" in
  MINGW*|MSYS*|CYGWIN*)
    # rustc's MSVC linker looks for nlist.lib on a native search path.
    # An MSYS -L directory is not one it opens.
    wout=$(cygpath -w "$out")
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$wout" \
      -L "native=${wdir}" -lnlist
    ;;
  *)
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
      -L "$dir" -lnlist
    ;;
esac
case "$os" in
  Darwin)
    DYLD_LIBRARY_PATH="$dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" "$out"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    PATH="$dir:$PATH" "$out"
    ;;
  *)
    LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out"
    ;;
esac
rm -rf -- "${dir:?}" "${out:?}"
