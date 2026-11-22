#!/usr/bin/env bash
# Compile an OpenMP program with AddressSanitizer, link it to libomp, and
# require LeakSanitizer to report an injected allocation. The program
# includes lsan_libomp.hpp. On Ubuntu libomp 18 the same program reports
# no other leak when that suppression is empty, so a pass does not show
# that the suppression is required on this runtime.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cxx=${CXX:-clang++}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
runtime_args=()
if command -v g++ >/dev/null; then
  libstdcxx=$(g++ -print-file-name=libstdc++.so)
  if [[ $libstdcxx = /* ]]; then
    runtime_args+=("-L$(dirname "$libstdcxx")")
  fi
fi

# The compiler's own libomp, not whichever libomp.so LD_LIBRARY_PATH finds.
# DT_RPATH is searched before LD_LIBRARY_PATH, so the suppression names the
# ___kmp_allocate_align in that library.
resource=$("$cxx" -print-resource-dir)
omp_libdir=$(cd "$(dirname "$(dirname "$resource")")" && pwd)
if [[ ! -e "$omp_libdir/libomp.so" && ! -e "$omp_libdir/libomp.so.5" ]]; then
  echo "libomp not found in $omp_libdir" >&2
  exit 1
fi
if [[ -n "${ASAN_SYMBOLIZER_PATH:-}" ]]; then
  if [[ ! -x "$ASAN_SYMBOLIZER_PATH" ]]; then
    echo "ASAN_SYMBOLIZER_PATH is not executable: $ASAN_SYMBOLIZER_PATH" >&2
    exit 1
  fi
elif command -v llvm-symbolizer >/dev/null 2>&1; then
  ASAN_SYMBOLIZER_PATH=$(command -v llvm-symbolizer)
else
  echo "llvm-symbolizer was not found" >&2
  exit 1
fi
export ASAN_SYMBOLIZER_PATH

cat >"$work/leak.cc" <<'EOF'
#include <stdlib.h>

#include <omp.h>

#include "lsan_libomp.hpp"

__attribute__((noinline)) void injectLeak() {
  auto *leak = static_cast<unsigned char *>(malloc(137));
  leak[0] = 1;
  asm volatile("" : : "r"(leak) : "memory");
}

int main() {
  int workers = 0;
#pragma omp parallel reduction(+ : workers)
  workers += omp_get_thread_num() >= 0 ? 1 : 0;
  if (workers < 2) {
    return 2;
  }
  injectLeak();
  return 0;
}
EOF

"$cxx" -std=c++20 -O0 -g -fsanitize=address -fopenmp \
  "${runtime_args[@]}" \
  -Wl,--disable-new-dtags \
  -Wl,-rpath,"${omp_libdir}" \
  -I"$root/CppCore/tests" "$work/leak.cc" -o "$work/leak"
ldd "$work/leak" | grep -E 'libomp\.so' >/dev/null
loaded=$(ldd "$work/leak" | awk '/libomp/{print $3; exit}')
case "$loaded" in
  "$omp_libdir"/*) ;;
  *)
    echo "loaded libomp $loaded, expected $omp_libdir" >&2
    exit 1
    ;;
esac
if ! readelf -d "$work/leak" | grep -q 'Library rpath:'; then
  echo "binary has no DT_RPATH" >&2
  exit 1
fi

set +e
OMP_NUM_THREADS=4 ASAN_OPTIONS=detect_leaks=1 \
  "$work/leak" >"$work/output" 2>&1
status=$?
set -e
cat "$work/output"

if [[ $status -eq 0 ]]; then
  echo "injected leak was not detected" >&2
  exit 1
fi
grep -F 'ERROR: LeakSanitizer: detected memory leaks' "$work/output" >/dev/null
grep -F '137 byte(s) leaked in 1 allocation(s)' "$work/output" >/dev/null
