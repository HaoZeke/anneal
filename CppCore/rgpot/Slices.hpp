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
// bits on any team. CppCore/tests/test_openmp_confined.py fails the test
// suite when OpenMP appears outside Slices.hpp and Slices.cc.
//
// A call over atoms takes its slice count from teamSize, so a small system
// opens no region at all.

#include <cstddef>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__SANITIZE_THREAD__)
#define RGPOT_TSAN_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RGPOT_TSAN_ACTIVE 1
#endif
#endif

#if defined(RGPOT_TSAN_ACTIVE)
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#endif

#if defined(RGPOT_TSAN_ACTIVE) && defined(_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#define RGPOT_GOMP_TSAN 1
extern "C" void GOMP_parallel(void (*)(void *), void *, unsigned, unsigned);
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

/// Atoms below which a call runs on the calling thread alone. A region
/// costs more than such a call: on 8 threads a neighbour search of 64
/// atoms, and a Sutton-Chen call on 55 to 80 atoms, took several times
/// their one-thread time.
inline constexpr std::size_t kTeamMinAtoms = 8192;

/// Slices for a call over `n` atoms: one below kTeamMinAtoms or when
/// threadsRequested() is one, else threadsRequested(), at most `n`.
[[nodiscard]] int teamSize(std::size_t n);

/// Regions the calling thread has opened through the helpers below, a
/// region the runtime gave one thread included. A call that leaves the
/// count unchanged ran on the calling thread alone.
[[nodiscard]] std::uint64_t teamsOpened() noexcept;

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
/// Counts a region for teamsOpened.
void countTeam() noexcept;

#if defined(RGPOT_GOMP_TSAN)
/// libgomp's uninstrumented fork does not give TSan a happens-before edge.
/// Passing our own final state to GOMP_parallel lets the caller release after
/// initialization and every worker acquire that exact address before reading.
template <typename Function>
void gompParallel(int threads, Function &&function) {
  using Fn = std::remove_reference_t<Function>;
  struct State {
    Fn *function;
    std::unique_ptr<std::atomic<int>[]> joined;
    std::atomic<int> fork{0};
    std::atomic<int> actual{0};
  };
  State state{
      std::addressof(function),
      std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(threads)),
  };
  state.fork.store(1, std::memory_order_release);
  GOMP_parallel(
      [](void *opaque) {
        auto &shared = *static_cast<State *>(opaque);
        (void)shared.fork.load(std::memory_order_acquire);
        const int thread = omp_get_thread_num();
        if (thread == 0) {
          shared.actual.store(omp_get_num_threads(), std::memory_order_release);
        }
        (*shared.function)();
        shared.joined[static_cast<std::size_t>(thread)].store(
            1, std::memory_order_release);
      },
      &state, static_cast<unsigned>(threads), 0);
  const int actual = state.actual.load(std::memory_order_acquire);
  for (int thread = 0; thread < actual; ++thread) {
    (void)state.joined[static_cast<std::size_t>(thread)].load(
        std::memory_order_acquire);
  }
}
#endif
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
    detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
    detail::gompParallel(request, [&] {
      const int team = omp_get_num_threads();
      for (int slice = omp_get_thread_num(); slice < nslices; slice += team) {
        body(slice);
      }
    });
#else
#if defined(RGPOT_TSAN_ACTIVE)
    // Distro OpenMP runtimes (libomp, libgomp) are built without TSan, so
    // their fork/join barriers are invisible to TSan without an explicit
    // release/acquire pair. Concurrent slices still race if they share writes.
    static std::atomic<std::uint64_t> tsan_fork_epoch{0};
    auto tsan_join =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(request));
#endif
#pragma omp parallel num_threads(request)
    {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
      tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
      (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
      const int team = omp_get_num_threads();
      for (int slice = omp_get_thread_num(); slice < nslices; slice += team) {
        body(slice);
      }
#if defined(RGPOT_TSAN_ACTIVE)
      tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
          1, std::memory_order_release);
#endif
    }
#if defined(RGPOT_TSAN_ACTIVE)
    for (int thread = 0; thread < request; ++thread) {
      (void)tsan_join[static_cast<std::size_t>(thread)].load(
          std::memory_order_acquire);
    }
#endif
#endif
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
    detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
    auto tsan_phase =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nphases));
    detail::gompParallel(request, [&] {
      const int team = omp_get_num_threads();
      const int first = omp_get_thread_num();
      for (int phase = 0; phase < nphases; ++phase) {
        if (phase > 0) {
          tsan_phase[static_cast<std::size_t>(phase)].fetch_add(
              1, std::memory_order_release);
#pragma omp barrier
          (void)tsan_phase[static_cast<std::size_t>(phase)].load(
              std::memory_order_acquire);
        }
        for (int slice = first; slice < nslices; slice += team) {
          body(phase, slice);
        }
      }
    });
#else
#if defined(RGPOT_TSAN_ACTIVE)
    static std::atomic<std::uint64_t> tsan_fork_epoch{0};
    auto tsan_join =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(request));
    auto tsan_phase =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nphases));
#endif
#pragma omp parallel num_threads(request)
    {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
      tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
      (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
      const int team = omp_get_num_threads();
      const int first = omp_get_thread_num();
      for (int phase = 0; phase < nphases; ++phase) {
        if (phase > 0) {
#if defined(RGPOT_TSAN_ACTIVE)
          tsan_phase[static_cast<std::size_t>(phase)].fetch_add(
              1, std::memory_order_release);
#endif
#pragma omp barrier
#if defined(RGPOT_TSAN_ACTIVE)
          (void)tsan_phase[static_cast<std::size_t>(phase)].load(
              std::memory_order_acquire);
#endif
        }
        for (int slice = first; slice < nslices; slice += team) {
          body(phase, slice);
        }
      }
#if defined(RGPOT_TSAN_ACTIVE)
      tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
          1, std::memory_order_release);
#endif
    }
#if defined(RGPOT_TSAN_ACTIVE)
    for (int thread = 0; thread < request; ++thread) {
      (void)tsan_join[static_cast<std::size_t>(thread)].load(
          std::memory_order_acquire);
    }
#endif
#endif
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
  detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
  detail::gompParallel(nranks, [&] {
    if (omp_get_num_threads() == nranks) {
      const int rank = omp_get_thread_num();
      if (rank == 0) {
        full = true;
      }
      body(rank);
    }
  });
#else
#if defined(RGPOT_TSAN_ACTIVE)
  static std::atomic<std::uint64_t> tsan_fork_epoch{0};
  auto tsan_join =
      std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nranks));
#endif
#pragma omp parallel num_threads(nranks)
  {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
    tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
    (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
    if (omp_get_num_threads() == nranks) {
      const int rank = omp_get_thread_num();
      if (rank == 0) {
        full = true;
      }
      body(rank);
    }
#if defined(RGPOT_TSAN_ACTIVE)
    tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
        1, std::memory_order_release);
#endif
  }
#if defined(RGPOT_TSAN_ACTIVE)
  for (int thread = 0; thread < nranks; ++thread) {
    (void)tsan_join[static_cast<std::size_t>(thread)].load(
        std::memory_order_acquire);
  }
#endif
#endif
  return full;
#else
  (void)body;
  return false;
#endif
}

} // namespace rgpot
