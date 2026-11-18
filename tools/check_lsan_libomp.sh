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
  "${runtime_args[@]}" -I"$root/CppCore/tests" "$work/leak.cc" -o "$work/leak"
ldd "$work/leak" | grep -E 'libomp\.so' >/dev/null

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
