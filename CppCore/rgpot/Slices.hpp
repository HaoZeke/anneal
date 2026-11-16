#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Every OpenMP region in rgpot opens here. A caller cuts its work into
// slices before the region and keeps one buffer per slice. The runtime may
// start fewer threads than slices: under OMP_THREAD_LIMIT, inside another
// region, with dynamic teams, or below a TeamCap. A thread then runs several
// slices in turn. No other file reads a thread number or a team size, so a
// buffer is indexed by its slice, and a sum taken in slice order has the same
// bits on any team.

#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {

/// Items `[begin, end)`.
struct Range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

/// Contiguous run of `n` items for `slice` in `[0, nslices)`. The runs
/// cover `n` without overlap, in slice order. Any other slice is empty.
[[nodiscard]] inline Range partition(std::int64_t n, int slice, int nslices) noexcept {
  if (nslices < 1) {
    nslices = 1;
  }
  if (slice < 0 || slice >= nslices || n <= 0) {
    return {};
  }
  const auto ns = static_cast<std::int64_t>(nslices);
  const auto s = static_cast<std::int64_t>(slice);
  return {(n * s) / ns, (n * (s + 1)) / ns};
}

/// Slice count from OMP_NUM_THREADS, read on every call. Unset, empty, or
/// a non-positive value is one slice, so a call that does not ask for
/// threads matches the serial order. One without OpenMP.
[[nodiscard]] int threadsRequested();

/// The team a region without a thread count would ask for, from
/// omp_set_num_threads or the first OMP_NUM_THREADS entry. One without
/// OpenMP. The runtime may still start fewer.
[[nodiscard]] int poolThreads() noexcept;

/// Sets the team `poolThreads` reports and fixes it against dynamic
/// adjustment. Shrinking it first lets the idle workers go.
void resizePool(int threads) noexcept;

/// While alive, regions this thread opens here start at most `threads`
/// threads. Zero or less lifts the cap. Slices and results do not change.
class TeamCap {
public:
  explicit TeamCap(int threads) noexcept;
  ~TeamCap();
  TeamCap(const TeamCap &) = delete;
  TeamCap &operator=(const TeamCap &) = delete;

private:
  int previous_;
};

namespace detail {
/// Threads to request for `wanted` slices or ranks under this thread's cap.
[[nodiscard]] int teamFor(int wanted) noexcept;
} // namespace detail

/// Runs `body(slice)` once for each slice in `[0, nslices)`, possibly
/// several on one thread, so slices must not wait on each other.
///
/// MSVC rejects `#pragma omp` inside a lambda (C2760, token PRAGMA_OMP).
/// The regions live here; the lambdas only supply the bodies.
template <typename Body>
void forEachSlice(int nslices, Body &&body) {
#if defined(_OPENMP)
  const int request = nslices > 1 ? detail::teamFor(nslices) : 1;
  if (request > 1) {
#pragma omp parallel num_threads(request)
    {
      const int team = omp_get_num_threads();
      for (int slice = omp_get_thread_num(); slice < nslices; slice += team) {
        body(slice);
      }
    }
    return;
  }
#endif
  for (int slice = 0; slice < nslices; ++slice) {
    body(slice);
  }
}

/// Runs `body(phase, slice)` for each phase in `[0, nphases)` and each
/// slice in `[0, nslices)`, in one region. Every slice of a phase ends
/// before any slice of the next phase starts. Within a phase the slices
/// follow the rule of `forEachSlice`.
template <typename Body>
void forEachSlicePhase(int nphases, int nslices, Body &&body) {
#if defined(_OPENMP)
  const int request = nslices > 1 && nphases > 0 ? detail::teamFor(nslices) : 1;
  if (request > 1) {
#pragma omp parallel num_threads(request)
    {
      const int team = omp_get_num_threads();
      const int first = omp_get_thread_num();
      for (int phase = 0; phase < nphases; ++phase) {
        if (phase > 0) {
#pragma omp barrier
        }
        for (int slice = first; slice < nslices; slice += team) {
          body(phase, slice);
        }
      }
    }
    return;
  }
#endif
  for (int phase = 0; phase < nphases; ++phase) {
    for (int slice = 0; slice < nslices; ++slice) {
      body(phase, slice);
    }
  }
}

/// Runs `body(rank)` for each rank in `[0, nranks)`, all at once on their
/// own threads, so ranks may wait on each other. Returns false and runs
/// nothing when the runtime starts a shorter team, or without OpenMP; the
/// caller then does the same work through `forEachSlice`. One rank runs
/// on the calling thread.
template <typename Body>
[[nodiscard]] bool onFullTeam(int nranks, Body &&body) {
  if (nranks <= 1) {
    if (nranks == 1) {
      body(0);
    }
    return true;
  }
#if defined(_OPENMP)
  if (detail::teamFor(nranks) < nranks) {
    return false;
  }
  bool full = false;
#pragma omp parallel num_threads(nranks)
  {
    if (omp_get_num_threads() == nranks) {
      const int rank = omp_get_thread_num();
      if (rank == 0) {
        full = true;
      }
      body(rank);
    }
  }
  return full;
#else
  (void)body;
  return false;
#endif
}

} // namespace rgpot
