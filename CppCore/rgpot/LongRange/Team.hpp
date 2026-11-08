// MIT License
// Copyright 2023--present rgpot developers

#pragma once

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {

// How many workers a mesh axis can feed.
//
// Measured on this host, one 25-point transform is 55us on one thread,
// 56us on two, 48us on four and 50us on eight. A 32-point complex
// transform is about 50us on four threads and tens of milliseconds on
// eight. So a short axis uses one worker below four threads and four
// workers after that. A longer axis needs at least 8 planes per worker
// (80 points can feed 8, 40 points can feed 5).
// The ladder records the team it asked for. The mesh may run fewer
// workers than that, and this remembers the request.
inline int &caller_team() {
  static int n = 1;
  return n;
}

inline int axis_workers(int npoints, int team) {
  if (team <= 1 || npoints <= 1) {
    return 1;
  }
  int cap = team;
  if (npoints <= 32) {
    cap = team >= 4 ? 4 : 1;
  } else {
    constexpr int kMinPlanes = 8;
    cap = npoints / kMinPlanes;
    if (cap < 1) {
      cap = 1;
    }
  }
  if (cap > team) {
    cap = team;
  }
  return cap < 1 ? 1 : cap;
}

// Spread and gather. One worker needs 32 atoms: 128 atoms feed four
// workers and not eight (eight was slower than four on the short mesh).
inline int atom_workers(int natoms, int team) {
  if (team <= 1 || natoms <= 1) {
    return 1;
  }
  constexpr int kMinAtoms = 32;
  int cap = natoms / kMinAtoms;
  if (cap < 1) {
    cap = 1;
  }
  if (cap > team) {
    cap = team;
  }
  return cap;
}

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
