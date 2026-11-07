// MIT License
// Copyright 2023--present rgpot developers

#pragma once

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {

// Cap the OpenMP team when a call has too little work. n_max is the
// largest atom count that uses the cap. Bigger calls keep every thread
// the caller asked for. Callers leave the extra threads idle instead of
// resizing the pool, which is much slower than the call itself.
inline int work_team(std::size_t n, std::size_t n_max, int cap) {
#ifdef _OPENMP
  int team = omp_get_max_threads();
  if (n <= n_max && cap > 0 && team > cap) {
    team = cap;
  }
  return team;
#else
  (void)n;
  (void)n_max;
  (void)cap;
  return 1;
#endif
}

} // namespace rgpot::lr
