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
rustc_bin=${RUSTC:-rustc}
if ! "$rustc_bin" --version >/dev/null 2>&1; then
  rustc_bin=$(find /usr/local/rustup/toolchains "${RUSTUP_HOME:-/usr/local/rustup}/toolchains" \
    -path '*/bin/rustc' -type f -print -quit 2>/dev/null || true)
fi
if [ -z "$rustc_bin" ]; then
  echo "rustc is not available (set RUSTC or add rustc to PATH)" >&2
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
for tool_bin in "$cxx" "${CC:-}"; do
  case "$tool_bin" in
    */*)
      tool_dir=$(dirname "$tool_bin")
      if [ -d "$tool_dir" ]; then
        case ":${PATH:-}:" in
          *":$tool_dir:"*) ;;
          *) PATH="${PATH:+$PATH:}$tool_dir"; export PATH ;;
        esac
      fi
      ;;
  esac
done
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
    rust_lib_dirs=
    add_rust_lib_dir() {
      case "$1" in
        */*)
          cand_dir=$(dirname "$1")
          if [ -d "$cand_dir" ]; then
            cand_dir=$(CDPATH= cd -- "$cand_dir" 2>/dev/null && pwd || printf '%s' "$cand_dir")
            case ":$rust_lib_dirs:" in
              *":$cand_dir:"*) ;;
              *) rust_lib_dirs="${rust_lib_dirs:+$rust_lib_dirs:}$cand_dir" ;;
            esac
          fi
          ;;
      esac
    }
    add_rust_lib_dir "$("$cxx" -print-libgcc-file-name 2>/dev/null || true)"
    add_rust_lib_dir "$("$cxx" -print-file-name=libgcc_s.so 2>/dev/null || true)"
    case "$cxx_rt" in
      -lstdc++)
        add_rust_lib_dir "$("$cxx" -print-file-name=libstdc++.so 2>/dev/null || true)"
        add_rust_lib_dir "$("$cxx" -print-file-name=libstdc++.a 2>/dev/null || true)"
        ;;
      -lc++)
        add_rust_lib_dir "$("$cxx" -print-file-name=libc++.so 2>/dev/null || true)"
        add_rust_lib_dir "$("$cxx" -print-file-name=libc++.dylib 2>/dev/null || true)"
        add_rust_lib_dir "$("$cxx" -print-file-name=libc++.a 2>/dev/null || true)"
        ;;
    esac
    set --
    if [ -n "$san_flag" ]; then
      set -- "$@" -C "linker=$cxx" -C "link-arg=$san_flag"
    fi
    if [ -n "$cxx_rt" ]; then
      set -- "$@" -C "link-arg=$cxx_rt"
    fi
    if [ -n "$rust_lib_dirs" ]; then
      old_ifs=$IFS
      IFS=:
      for lib_dir in $rust_lib_dirs; do
        if [ -n "$lib_dir" ]; then
          set -- "$@" -L "$lib_dir"
        fi
      done
      IFS=$old_ifs
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
    rt_lib_path=$dir
    for rt_soname in libstdc++.so.6 libc++.so.1 libgomp.so.1 libomp.so; do
      rt_file=$("$cxx" -print-file-name="$rt_soname" 2>/dev/null || true)
      case "$rt_file" in
        */*)
          rt_dir=$(dirname "$rt_file")
          if [ -d "$rt_dir" ]; then
            rt_dir=$(CDPATH= cd -- "$rt_dir" 2>/dev/null && pwd || printf '%s' "$rt_dir")
            case ":$rt_lib_path:" in
              *":$rt_dir:"*) ;;
              *) rt_lib_path="$rt_lib_path:$rt_dir" ;;
            esac
          fi
          ;;
      esac
    done
    preload=
    if [ -n "$san_flag" ] && command -v ldd >/dev/null 2>&1; then
      preload=$(ldd "$dir/libnlist.so" 2>/dev/null | awk '/lib(a|t)san\.so/ {print $3; exit}')
    fi
    case "$san_flag" in
      *thread*)
        if command -v setarch >/dev/null 2>&1 && setarch -R true >/dev/null 2>&1; then
          setarch -R env \
            ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
            LD_LIBRARY_PATH="$rt_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$out"
        else
          env \
            ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
            LD_LIBRARY_PATH="$rt_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$out"
        fi
        ;;
      *)
        env \
          ${preload:+LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"} \
          LD_LIBRARY_PATH="$rt_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$out"
        ;;
    esac
    ;;
esac
