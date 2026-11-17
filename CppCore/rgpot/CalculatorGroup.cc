// MIT License
// Copyright 2023--present rgpot developers

// The calculator-group API without MPI. Every MPI call lives in
// librgpot_mpi (CalculatorGroupMpi.cc), which bindCalculators and
// calculatorsUseMpi load on first use through the C ABI of
// calculator_mpi_abi.h. Until then, and in a process that never loads it,
// the process is one calculator of one rank.

#include "rgpot/CalculatorGroup.hpp"

#include "rgpot/calculator_mpi_abi.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace rgpot {
namespace {
std::mutex g_mu;
std::vector<CalculatorHook> g_hooks;
std::vector<CalculatorCommAdopter> g_adopters;
CalculatorGroup g_group;
int g_bound_rpc = 0;
bool g_bound = false;
std::atomic<bool> g_abort_at_exit{false};

std::mutex g_load_mu;
std::atomic<const rgpot_mpi_api_t *> g_api{nullptr};
std::string g_load_error;

const rgpot_mpi_api_t *api() { return g_api.load(std::memory_order_acquire); }

int abortRequested() {
  return g_abort_at_exit.load(std::memory_order_acquire) ? 1 : 0;
}

#if !defined(_WIN32)
/// Directory of the object holding this function (librgpot, or the
/// executable that links rgpot statically), with a trailing slash.
std::string ownDirectory() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(&ownDirectory), &info) == 0 ||
      info.dli_fname == nullptr)
    return {};
  std::string path = info.dli_fname;
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string{} : path.substr(0, slash + 1);
}

const rgpot_mpi_api_t *openLibrary(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    const char *why = dlerror();
    g_load_error = why ? why : ("cannot open " + path);
    return nullptr;
  }
  auto fn =
      reinterpret_cast<rgpot_mpi_api_fn>(dlsym(handle, RGPOT_MPI_API_SYMBOL));
  if (fn == nullptr) {
    g_load_error = path + " lacks " RGPOT_MPI_API_SYMBOL;
    return nullptr;
  }
  const rgpot_mpi_api_t *table = fn();
  if (table == nullptr || table->abi_version != RGPOT_MPI_ABI_VERSION ||
      table->struct_size < sizeof(rgpot_mpi_api_t)) {
    g_load_error = path + " has an incompatible rgpot MPI ABI";
    return nullptr;
  }
  return table;
}
#endif
} // namespace

int loadCalculatorMpi(const char *path) {
  if (api() != nullptr)
    return 1;
  std::lock_guard<std::mutex> lock(g_load_mu);
  if (api() != nullptr)
    return 1;
#if defined(_WIN32)
  (void)path;
  g_load_error = "calculator groups need dlopen";
  return 0;
#else
  std::vector<std::string> candidates;
  if (path != nullptr && path[0] != '\0') {
    candidates.emplace_back(path);
  } else {
    if (const char *env = std::getenv("RGPOT_MPI_LIBRARY");
        env != nullptr && env[0] != '\0')
      candidates.emplace_back(env);
    const std::string dir = ownDirectory();
    if (!dir.empty())
      candidates.push_back(dir + RGPOT_MPI_LIBRARY_NAME);
    candidates.emplace_back(RGPOT_MPI_LIBRARY_NAME);
  }
  for (const std::string &c : candidates) {
    if (const rgpot_mpi_api_t *table = openLibrary(c)) {
      g_api.store(table, std::memory_order_release);
      g_load_error.clear();
      return 1;
    }
  }
  return 0;
#endif
}

std::string calculatorMpiLoadError() {
  std::lock_guard<std::mutex> lock(g_load_mu);
  return g_load_error;
}

int calculatorMpiLoaded() { return api() != nullptr ? 1 : 0; }

void addCalculatorHook(CalculatorHook hook) {
  if (!hook)
    return;
  std::lock_guard<std::mutex> lock(g_mu);
  for (CalculatorHook have : g_hooks) {
    if (have == hook)
      return;
  }
  g_hooks.push_back(hook);
}

bool addCalculatorCommAdopter(CalculatorCommAdopter callback) {
  if (!callback)
    return false;
  std::lock_guard<std::mutex> lock(g_mu);
  for (CalculatorCommAdopter have : g_adopters) {
    if (have == callback)
      return !g_bound || g_group.index >= 0;
  }
  if (g_bound && g_group.index < 0)
    return false;
  g_adopters.push_back(callback);
  if (g_bound && adoptCalculatorComm(callback) != g_group.index) {
    g_group.index = -1;
    return false;
  }
  return true;
}

CalculatorGroup bindCalculators(int ranks_per_calculator) {
  loadCalculatorMpi(nullptr);
  std::lock_guard<std::mutex> lock(g_mu);
  // g_bound_rpc holds the resolved size, so resolve the argument the same way.
  const int asked = ranks_per_calculator > 0 ? ranks_per_calculator
                    : api() != nullptr       ? g_group.world_size
                                             : 1;
  if (g_bound && g_bound_rpc == asked)
    return g_group;
  if (g_bound) {
    CalculatorGroup refused;
    refused.index = -1;
    return refused;
  }

  int rpc = ranks_per_calculator;
  g_group = CalculatorGroup{};
  if (const rgpot_mpi_api_t *mpi = api()) {
    rgpot_mpi_group_t got{};
    mpi->bind(rpc, &got);
    g_group.index = got.index;
    g_group.ranks = got.ranks;
    g_group.rank_in_group = got.rank_in_group;
    g_group.world_size = got.world_size;
    if (rpc <= 0)
      rpc = got.world_size;
  } else {
    if (rpc <= 0)
      rpc = 1;
    if (rpc != 1)
      g_group.index = -1;
    else {
      g_group.index = 0;
      g_group.ranks = 1;
      g_group.world_size = 1;
    }
  }
  const int expected_index = g_group.index;
  for (CalculatorHook hook : g_hooks) {
    int idx = hook(rpc);
    // A hook cannot recover a refused split or another hook's refusal.
    if (idx < 0)
      g_group.index = -1;
  }
  // All registered engines receive the same split, including when a
  // hook refuses it. Skipping later callbacks could strand their ranks
  // inside an engine collective. No callback can clear a refusal.
  for (CalculatorCommAdopter callback : g_adopters) {
    const int idx = adoptCalculatorComm(callback);
    if (idx < 0 || idx != expected_index)
      g_group.index = -1;
  }
  g_bound = true;
  g_bound_rpc = rpc;
  return g_group;
}

const CalculatorGroup &thisCalculator() { return g_group; }

int calculatorComm(void *comm_out, std::size_t comm_bytes) {
  const rgpot_mpi_api_t *mpi = api();
  return mpi ? mpi->comm(comm_out, comm_bytes) : 0;
}

int adoptCalculatorComm(CalculatorCommAdopter callback) {
  const rgpot_mpi_api_t *mpi = api();
  return mpi && callback ? mpi->adopt(callback) : -1;
}

int calculatorsUseMpi() {
  if (loadCalculatorMpi(nullptr) == 0)
    return 0;
  return api()->initialized();
}

int calculatorWorldSize() {
  if (!g_bound)
    return 1;
  return g_group.world_size;
}

int calculatorCount() {
  if (!g_bound || g_group.index < 0 || g_group.ranks <= 0)
    return 1;
  return g_group.world_size / g_group.ranks;
}

int shareFromCalculator(int owner, void *data, std::size_t bytes) {
  const rgpot_mpi_api_t *mpi = api();
  return mpi ? mpi->share(owner, data, bytes) : 0;
}

void finalizeMpiAtExit() {
  if (const rgpot_mpi_api_t *mpi = api())
    mpi->finalize_at_exit(&abortRequested);
}

void abortMpiAtExit() {
  g_abort_at_exit.store(true, std::memory_order_release);
}

bool mpiAbortRequested() {
  return g_abort_at_exit.load(std::memory_order_acquire);
}

int calculatorAgree(unsigned char *flags, std::size_t n) {
  const rgpot_mpi_api_t *mpi = api();
  return mpi ? mpi->agree(flags, n) : 0;
}

void publishCalculatorError(const std::string &message) {
  if (const rgpot_mpi_api_t *mpi = api())
    mpi->publish_error(message.data(), message.size());
}

} // namespace rgpot
