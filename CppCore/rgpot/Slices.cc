// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Slices.hpp"

#include <cstdlib>

namespace rgpot {
namespace {

thread_local int t_team_cap = 0;

} // namespace

int threadsRequested() {
#if defined(_OPENMP)
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }
  char *end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 1) {
    return 1;
  }
  if (parsed > 0x7fffffffL) {
    return 0x7fffffff;
  }
  return static_cast<int>(parsed);
#else
  return 1;
#endif
}

int poolThreads() noexcept {
#if defined(_OPENMP)
  const int n = omp_get_max_threads();
  return n > 0 ? n : 1;
#else
  return 1;
#endif
}

void resizePool(int threads) noexcept {
#if defined(_OPENMP)
  if (threads < 1) {
    threads = 1;
  }
  omp_set_dynamic(0);
  const int current = poolThreads();
  if (current > threads) {
    // MSVC's OpenMP 2.0 runtime has no pause. GCC reports 4.5 and has one.
#if _OPENMP >= 201511
    omp_pause_resource_all(omp_pause_soft);
#endif
    omp_set_num_threads(threads);
  } else if (current < threads) {
    omp_set_num_threads(threads);
  }
#else
  (void)threads;
#endif
}

TeamCap::TeamCap(int threads) noexcept : previous_(t_team_cap) {
  t_team_cap = threads > 0 ? threads : 0;
}

TeamCap::~TeamCap() { t_team_cap = previous_; }

namespace detail {

int teamFor(int wanted) noexcept {
  const int cap = t_team_cap;
  return cap > 0 && cap < wanted ? cap : wanted;
}

} // namespace detail

} // namespace rgpot
