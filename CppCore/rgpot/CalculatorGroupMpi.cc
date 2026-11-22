// MIT License
// Copyright 2023--present rgpot developers

// librgpot_mpi: every MPI call of the calculator groups, behind the C ABI
// of calculator_mpi_abi.h. librgpot loads this library with dlopen on the
// first calculator-group call that needs MPI, so a process that never asks
// for groups never maps libmpi.

#include "rgpot/calculator_mpi_abi.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <time.h>
#include <vector>

#include <mpi.h>

#if defined(_WIN32)
#define RGPOT_MPI_EXPORT __declspec(dllexport)
#else
#define RGPOT_MPI_EXPORT __attribute__((visibility("default")))
#endif

namespace {

rgpot_mpi_group_t g_group{0, 1, 0, 1};
bool g_bound = false;
MPI_Comm g_comm = MPI_COMM_NULL;
bool g_mpi_owner = false;
int (*g_abort_requested)(void) = nullptr;

constexpr int kErrorCap = 512;
constexpr double kErrorWaitS = 10.0;

bool waitRequest(MPI_Request *req, double budget_s) {
  const double start = MPI_Wtime();
  for (;;) {
    int flag = 0;
    MPI_Test(req, &flag, MPI_STATUS_IGNORE);
    if (flag)
      return true;
    if (MPI_Wtime() - start >= budget_s)
      return false;
    struct timespec pause = {0, 1000000};
    nanosleep(&pause, nullptr);
  }
}

void flushAndAbortWorld(MPI_Comm sync_comm, bool leader) {
  std::fflush(stderr);
  if (sync_comm != MPI_COMM_NULL) {
    MPI_Request req = MPI_REQUEST_NULL;
    if (MPI_Ibarrier(sync_comm, &req) == MPI_SUCCESS)
      waitRequest(&req, kErrorWaitS);
  }
  // Open MPI's MPI_Abort kills peer processes immediately and drops unread
  // bytes still in their stderr forwarding pipes. Barrier first so every rank
  // has flushed stderr, then let one rank abort after a short drain window.
  const struct timespec drain = {0, leader ? 50000000L : 500000000L};
  nanosleep(&drain, nullptr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int calculatorCount() {
  if (!g_bound || g_group.index < 0 || g_group.ranks <= 0)
    return 1;
  return g_group.world_size / g_group.ranks;
}

int broadcastFromOwner(int owner, void *data, std::size_t bytes) {
  const int root = owner * g_group.ranks;
  auto *p = static_cast<std::uint8_t *>(data);
  while (bytes > 0) {
    const std::size_t chunk = bytes > static_cast<std::size_t>(INT_MAX)
                                  ? static_cast<std::size_t>(INT_MAX)
                                  : bytes;
    MPI_Bcast(p, static_cast<int>(chunk), MPI_BYTE, root, MPI_COMM_WORLD);
    p += chunk;
    bytes -= chunk;
  }
  return 1;
}

void bind(int ranks_per_calculator, rgpot_mpi_group_t *out) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    MPI_Init(nullptr, nullptr);
    g_mpi_owner = true;
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rpc = ranks_per_calculator <= 0 ? size : ranks_per_calculator;
  g_group = rgpot_mpi_group_t{0, 1, 0, size};
  if (rpc > size || size % rpc != 0) {
    g_group.index = -1;
  } else {
    MPI_Comm sub = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank / rpc, rank % rpc, &sub);
    g_comm = sub;
    g_group.index = rank / rpc;
    g_group.ranks = rpc;
    g_group.rank_in_group = rank % rpc;
  }
  g_bound = true;
  if (out)
    *out = g_group;
}

int comm(void *comm_out, std::size_t comm_bytes) {
  if (!comm_out || comm_bytes != sizeof(MPI_Comm) || g_comm == MPI_COMM_NULL)
    return 0;
  *static_cast<MPI_Comm *>(comm_out) = g_comm;
  return 1;
}

int adopt(int (*callback)(const void *, std::size_t, int)) {
  if (!callback || !g_bound || g_group.index < 0 || g_comm == MPI_COMM_NULL)
    return -1;
  return callback(&g_comm, sizeof(g_comm), g_group.ranks);
}

int initialized() {
  int inited = 0;
  MPI_Initialized(&inited);
  return inited ? 1 : 0;
}

int share(int owner, void *data, std::size_t bytes) {
  int inited = 0;
  MPI_Initialized(&inited);
  int size = 1;
  int rank = 0;
  if (inited) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  const bool collective = inited && size > 1 && g_bound && g_group.index >= 0 &&
                          g_group.world_size > 1;
  if (!collective) {
    if (!g_bound || g_group.index < 0 || (bytes > 0 && !data))
      return 0;
    if (owner < 0 || owner >= calculatorCount())
      return 0;
    return broadcastFromOwner(owner, data, bytes);
  }

  int reason = 0;
  if (!g_bound || g_group.index < 0 || (bytes > 0 && !data))
    reason = 1;
  else if (owner < 0 || owner >= calculatorCount())
    reason = 2;
  if (bytes > static_cast<std::size_t>(LLONG_MAX))
    reason = 1;
  const long long local[3] = {
      reason,
      static_cast<long long>(owner),
      static_cast<long long>(
          bytes > static_cast<std::size_t>(LLONG_MAX) ? LLONG_MAX : bytes),
  };
  std::vector<long long> all(static_cast<std::size_t>(size) * 3, 0);
  if (MPI_Allgather(local, 3, MPI_LONG_LONG, all.data(), 3, MPI_LONG_LONG,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    std::fprintf(stderr,
                 "rgpot rank %d: shareFromCalculator allgather failed\n", rank);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0;
  }

  bool bad = false;
  for (int src = 0; src < size; ++src) {
    const std::size_t base = static_cast<std::size_t>(src) * 3;
    if (all[base] != 0 || all[base + 1] != all[1] || all[base + 2] != all[2])
      bad = true;
  }
  if (!bad)
    return broadcastFromOwner(owner, data, bytes);

  for (int src = 0; src < size; ++src) {
    const std::size_t base = static_cast<std::size_t>(src) * 3;
    const long long src_reason = all[base];
    const long long src_owner = all[base + 1];
    const long long src_bytes = all[base + 2];
    if (src_reason == 0 && src_owner == all[1] && src_bytes == all[2])
      continue;
    const char *why = "shareFromCalculator rejected";
    if (src_reason == 2)
      why = "shareFromCalculator owner out of range";
    else if (src_reason == 1)
      why = "shareFromCalculator missing buffer or group";
    else if (src_owner != all[1])
      why = "shareFromCalculator owner disagrees";
    else
      why = "shareFromCalculator byte count disagrees";
    std::fprintf(stderr, "rgpot rank %d: %s\n", src, why);
  }
  std::fflush(stderr);
  // The first MPI_Abort kills every other rank. A peer that has not
  // finished the fprintf above then leaves an empty log. Every rank
  // that observed the mismatch is in this call; wait until those
  // writes have landed.
  MPI_Barrier(MPI_COMM_WORLD);
  flushAndAbortWorld(MPI_COMM_WORLD, rank == 0);
  return 0;
}

void finalizeAtExit(int (*abort_requested)(void)) {
  static std::once_flag once;
  g_abort_requested = abort_requested;
  std::call_once(once, [] {
    std::atexit([] {
      int inited = 0;
      int finalized = 0;
      MPI_Initialized(&inited);
      MPI_Finalized(&finalized);
      if (!inited || finalized)
        return;
      // MPI_Finalize is collective. A rank that leaves after a failed
      // engine call must not wait there for peers blocked in a
      // collective of their own; it takes the whole world down instead,
      // whether or not this library owns the MPI_Init.
      if (g_abort_requested && g_abort_requested() != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
      }
      if (!g_mpi_owner)
        return;
      if (const char *trace = std::getenv("RGPOT_MPI_FINALIZE_TRACE")) {
        if (trace[0] == '1' && trace[1] == '\0') {
          std::fprintf(stderr, "rgpot MPI_Finalize\n");
          std::fflush(stderr);
        }
      }
      MPI_Finalize();
    });
  });
}

int agree(unsigned char *flags, std::size_t n) {
  int inited = 0;
  int finalized = 0;
  MPI_Initialized(&inited);
  MPI_Finalized(&finalized);
  if (!inited || finalized)
    return 0;
  if (!g_bound || g_group.index < 0 || g_group.ranks < 2 ||
      g_comm == MPI_COMM_NULL)
    return 0;
  while (n > 0) {
    const std::size_t chunk = n > static_cast<std::size_t>(INT_MAX)
                                  ? static_cast<std::size_t>(INT_MAX)
                                  : n;
    MPI_Allreduce(MPI_IN_PLACE, flags, static_cast<int>(chunk),
                  MPI_UNSIGNED_CHAR, MPI_LAND, g_comm);
    flags += chunk;
    n -= chunk;
  }
  return 1;
}

void abortWithLocal(int rank, const char *message, std::size_t len) {
  if (len > 0) {
    std::fprintf(stderr, "rgpot rank %d: %.*s\n", rank, static_cast<int>(len),
                 message);
    std::fflush(stderr);
  }
  MPI_Abort(MPI_COMM_WORLD, 1);
}

// Every rank of the calling calculator enters this, and only those ranks:
// a host evaluating an uneven batch calls a force on some calculators and
// not on others, so the exchange stays on the calculator's communicator.
// A non-empty message is printed on every rank of that calculator, then
// MPI_Abort runs on MPI_COMM_WORLD, which takes the idle calculators down
// too.
void publishError(const char *message, std::size_t len) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    return;
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (!g_bound || g_group.index < 0 || g_group.ranks < 2 ||
      g_comm == MPI_COMM_NULL) {
    if (len > 0)
      abortWithLocal(world_rank, message, len);
    return;
  }
  int size = 1;
  MPI_Comm_size(g_comm, &size);
  // Group rank r of calculator g is world rank g * ranks + r (the split is
  // keyed by rank / ranks, ordered by rank % ranks).
  const int world_base = g_group.index * g_group.ranks;

  const int local_n =
      static_cast<int>(std::min(len, static_cast<std::size_t>(kErrorCap)));
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  MPI_Request req = MPI_REQUEST_NULL;
  if (MPI_Iallgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, g_comm,
                     &req) != MPI_SUCCESS) {
    abortWithLocal(world_rank, message, len);
    return;
  }
  if (len == 0) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  } else if (!waitRequest(&req, kErrorWaitS)) {
    abortWithLocal(world_rank, message, len);
    return;
  }

  int max_n = 0;
  bool any = false;
  for (int &count : counts) {
    count = std::clamp(count, 0, kErrorCap);
    if (count > 0)
      any = true;
    max_n = std::max(max_n, count);
  }
  if (!any || max_n <= 0)
    return;

  std::vector<char> send(static_cast<std::size_t>(max_n), '\0');
  std::vector<char> recv(
      static_cast<std::size_t>(max_n) * static_cast<std::size_t>(size), '\0');
  if (local_n > 0)
    std::memcpy(send.data(), message,
                static_cast<std::size_t>(std::min(local_n, max_n)));
  req = MPI_REQUEST_NULL;
  if (MPI_Iallgather(send.data(), max_n, MPI_CHAR, recv.data(), max_n, MPI_CHAR,
                     g_comm, &req) != MPI_SUCCESS) {
    abortWithLocal(world_rank, message, len);
    return;
  }
  if (len == 0) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  } else if (!waitRequest(&req, kErrorWaitS)) {
    abortWithLocal(world_rank, message, len);
    return;
  }

  for (int src = 0; src < size; ++src) {
    const int count = counts[static_cast<std::size_t>(src)];
    if (count <= 0)
      continue;
    const std::size_t offset =
        static_cast<std::size_t>(src) * static_cast<std::size_t>(max_n);
    std::fprintf(stderr, "rgpot rank %d: %.*s\n", world_base + src,
                 std::min(count, max_n), recv.data() + offset);
  }
  flushAndAbortWorld(g_comm, g_group.rank_in_group == 0);
}

const rgpot_mpi_api_t kApi = {
    RGPOT_MPI_ABI_VERSION,
    static_cast<uint32_t>(sizeof(rgpot_mpi_api_t)),
    &bind,
    &comm,
    &initialized,
    &share,
    &finalizeAtExit,
    &agree,
    &publishError,
    &adopt,
};

} // namespace

extern "C" RGPOT_MPI_EXPORT const rgpot_mpi_api_t *rgpot_mpi_api_v1(void) {
  return &kApi;
}
