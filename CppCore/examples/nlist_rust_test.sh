#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
# Compile the hand-written Rust caller against the neighbour archives.
# Arguments after the build directory link the OpenMP, C++ and sanitizer
# runtimes the archives were compiled for. Without them -fopenmp, the C++
# standard library (-lstdc++ or -lc++), and any sanitizer symbols in the
# archives are probed so standalone and cross-compiler invocations link.
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail
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
dir=$(mktemp -d)
out=$(mktemp)
cleanup() {
  rm -f \
    "$dir/libnlist.so" \
    "$dir/libnlist.dylib" \
    "$dir/nlist.dll" \
    "$dir/nlist.lib" \
    "$dir/nlist.exp" \
    "$dir/omp_probe" \
    "$dir/cxx_probe" \
    "$out"
  rmdir "$dir" 2>/dev/null || true
}
trap cleanup EXIT

has_omp=0
cxx_rt=
has_cxx_rt=0
san_flag=
has_san=0
for arg in "$@"; do
  case "$arg" in
    -fopenmp|-fopenmp=*|-lomp|-lgomp|*/libomp.*|*/libgomp.*) has_omp=1 ;;
  esac
  case "$arg" in
    -lstdc++|-lc++)
      cxx_rt=$arg
      has_cxx_rt=1
      ;;
  esac
  case "$arg" in
    -fsanitize=*)
      san_flag=$arg
      has_san=1
      ;;
  esac
done

case "$os" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *)
    if [ -z "$cxx_rt" ]; then
      stdlib_probe=$("$cxx" -x c++ -std=c++20 -E - 2>/dev/null <<'EOF' || true
#include <version>
#if defined(_LIBCPP_VERSION)
RGPOT_LIBCPP
#elif defined(__GLIBCXX__)
RGPOT_GLIBCXX
#endif
EOF
      )
      case "$stdlib_probe" in
        *RGPOT_LIBCPP*) cxx_rt=-lc++ ;;
        *RGPOT_GLIBCXX*) cxx_rt=-lstdc++ ;;
        *)
          if "$cxx" -x c++ - -lstdc++ -o "$dir/cxx_probe" >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
          then
            cxx_rt=-lstdc++
          elif "$cxx" -x c++ - -lc++ -o "$dir/cxx_probe" >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
          then
            cxx_rt=-lc++
          fi
          ;;
      esac
    fi
    if [ -z "$san_flag" ]; then
      half_syms=$(nm "$half" 2>/dev/null || true)
      san_list=
      case "$half_syms" in
        *__tsan_init*) san_list=thread ;;
        *)
          case "$half_syms" in
            *__asan_init*) san_list=address ;;
          esac
          case "$half_syms" in
            *__ubsan_*)
              san_list=${san_list:+$san_list,}undefined
              ;;
          esac
          ;;
      esac
      if [ -n "$san_list" ]; then
        san_flag="-fsanitize=$san_list"
      fi
    fi
    ;;
esac

# Clang takes -fopenmp even when no libomp is on its search path, and then
# fails the link, so the probe links rather than preprocesses.
omp_flag=
if [ "$has_omp" -eq 0 ] && "$cxx" -fopenmp -x c++ - ${cxx_rt:+"$cxx_rt"} -o "$dir/omp_probe" >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
then
  omp_flag=-fopenmp
fi

# Coverage builds instrument the archives. The shared library has to pull
# the gcov runtime or the loader cannot resolve __gcov_init.
gcov_flag=
if nm "$half" 2>/dev/null | grep -q __gcov_init; then
  gcov_flag=--coverage
fi

extra_rt=
if [ "$has_cxx_rt" -eq 0 ] && [ -n "$cxx_rt" ]; then
  extra_rt=$cxx_rt
fi
extra_san=
if [ "$has_san" -eq 0 ] && [ -n "$san_flag" ]; then
  extra_san=$san_flag
fi

case "$os" in
  Darwin)
    "$cxx" -shared -o "$dir/libnlist.dylib" \
      -Wl,-force_load,"$half" -Wl,-force_load,"$mini" -Wl,-force_load,"$fma" \
      "$@" ${omp_flag:+"$omp_flag"} ${extra_rt:+"$extra_rt"} ${extra_san:+"$extra_san"} ${gcov_flag:+"$gcov_flag"}
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
      -Wl,--no-whole-archive "$@" ${omp_flag:+"$omp_flag"} ${extra_rt:+"$extra_rt"} ${extra_san:+"$extra_san"} ${gcov_flag:+"$gcov_flag"}
    ;;
esac

case "$os" in
  MINGW*|MSYS*|CYGWIN*)
    # rustc's MSVC linker looks for nlist.lib on a native search path.
    # An MSYS -L directory is not one it opens.
    wout=$(cygpath -w "$out")
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$wout" \
      -L "native=${wdir}" -lnlist
    ;;
  *)
    set --
    if [ -n "$san_flag" ]; then
      set -- "$@" -C "linker=$cxx" -C "link-arg=$san_flag"
    fi
    if [ -n "$cxx_rt" ]; then
      set -- "$@" -C "link-arg=$cxx_rt"
    fi
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
      "$@" -L "$dir" -lnlist
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
    preload=
    if [ -n "$san_flag" ] && command -v ldd >/dev/null 2>&1; then
      preload=$(ldd "$dir/libnlist.so" 2>/dev/null | awk '/lib(a|t)san\.so/ {print $3; exit}')
    fi
    case "$san_flag" in
      *thread*)
        if command -v setarch >/dev/null 2>&1 && setarch -R true >/dev/null 2>&1; then
          setarch -R env \
            ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
            LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$out"
        else
          env \
            ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
            LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$out"
        fi
        ;;
      *)
        env \
          ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
          LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$out"
        ;;
    esac
    ;;
esac
