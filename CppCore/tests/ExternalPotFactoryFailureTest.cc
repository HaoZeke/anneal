// SPDX-License-Identifier: MIT
// Copyright 2023--present rgpot developers
//
// A constructor of the external potentials that fails must release the
// handle it was building. The global allocator is replaced by a counting
// one so the check needs no sanitizer: the handle is a small heap object
// that only the failing path can leave behind.

#include "rgpot/ExternalPot/external_potentials.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {

std::atomic<long> live_allocations{0};

} // namespace

void *operator new(std::size_t size) {
  void *p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  live_allocations.fetch_add(1, std::memory_order_relaxed);
  return p;
}

void operator delete(void *p) noexcept {
  if (p != nullptr) {
    live_allocations.fetch_sub(1, std::memory_order_relaxed);
    std::free(p);
  }
}

void operator delete(void *p, std::size_t) noexcept { operator delete(p); }

namespace {

constexpr double kU0 = 1.0;
constexpr double kCutoff = 3.0;
constexpr double kPsi = 1.0;
constexpr const char *kMissingEngine = "/nonexistent/rgpot-engine-for-the-test";

using Constructor = rgpot_external_potential *(*)(const char *, double, double,
                                                  double, char *, size_t);

int check_failed_constructor(const char *name, Constructor construct) {
  char err[256] = {};
  // The first failure may allocate state that outlives the call, such as a
  // locale or a loader buffer. The measured failure is the second.
  rgpot_external_potential *warm =
      construct(kMissingEngine, kU0, kCutoff, kPsi, err, sizeof(err));
  if (warm != nullptr) {
    std::fprintf(stderr, "%s: a missing engine built a potential\n", name);
    rgpot_external_potential_free(warm);
    return 1;
  }
  const long before = live_allocations.load();
  rgpot_external_potential *pot =
      construct(kMissingEngine, kU0, kCutoff, kPsi, err, sizeof(err));
  const long leaked = live_allocations.load() - before;
  if (pot != nullptr) {
    std::fprintf(stderr, "%s: a missing engine built a potential\n", name);
    rgpot_external_potential_free(pot);
    return 1;
  }
  if (err[0] == '\0') {
    std::fprintf(stderr, "%s: the failure wrote no message\n", name);
    return 1;
  }
  if (leaked != 0) {
    std::fprintf(stderr, "%s: a failed constructor left %ld allocation(s)\n",
                 name, leaked);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  int failures = 0;
  failures += check_failed_constructor("lammps", rgpot_lammps_potential);
  failures += check_failed_constructor("gromacs", rgpot_gromacs_potential);
  failures += check_failed_constructor("openmm", rgpot_openmm_potential);
  return failures == 0 ? 0 : 1;
}
