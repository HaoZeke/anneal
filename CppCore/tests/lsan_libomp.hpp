#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// LLVM libomp (including conda-forge's libgomp.so.1 -> libomp.so symlink)
// retains a 128-byte process-lifetime allocation in ___kmp_allocate_align
// once an OpenMP team starts. Scope the LeakSanitizer suppression to the
// OpenMP test binaries so real leaks in library and test code still fail.

extern "C" const char *__lsan_default_suppressions() {
  return "leak:___kmp_allocate_align\n";
}
