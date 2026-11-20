#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// LLVM libomp (including conda-forge's libgomp.so.1 -> libomp.so symlink)
// retains a process-lifetime affinity map allocation (128 bytes on an 8-CPU
// runner) inside ___kmp_allocate / ___kmp_allocate_align once OpenMP
// initializes. Match ___kmp_allocate so both unstripped libomp.so
// (___kmp_allocate_align) and stripped system libomp.so.5 (which exports
// ___kmp_allocate in .dynsym) are covered when llvm-symbolizer is installed.
// Scope the LeakSanitizer suppression to the OpenMP test binaries so real
// leaks in library and test code still fail.

extern "C" const char *__lsan_default_suppressions() {
  return "leak:___kmp_allocate\n";
}
