// MIT License
// Copyright 2023--present rgpot developers

#ifdef RGPOT_HAS_CP2K
#include "rgpot/CP2KPot/CP2KPot.hpp"
#endif
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/CalculatorGroup.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dlfcn.h>
#include <mpi.h>

namespace {

int split_count(bool *present) {
  void *sym = dlsym(RTLD_DEFAULT, "rgpot_test_mpi_split_count");
  if (!sym) {
    *present = false;
    return -1;
  }
  *present = true;
  return reinterpret_cast<int (*)()>(sym)();
}

int hook_calls = 0;

int refuse_hook(int) {
  ++hook_calls;
  return -1;
}

int accept_hook(int ranks) {
  ++hook_calls;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank / ranks;
}

} // namespace

// mpirun -n 2 missing ENGINE: both ranks return -1 and MPI_Comm_split
// does not run. mpirun -n 4 split ENGINE: two ranks per calculator, indexes
// 0, 0, 1, 1, the adopted communicator is MPI_IDENT with calculatorComm,
// and each rank sees one MPI_Comm_split.
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  const char *mode = argv[1];
  const char *engine = argv[2];
  if (setenv("RGPOT_CPMD_ENGINE", engine, 1) != 0 ||
      setenv("RGPOT_CP2K_ENGINE", engine, 1) != 0)
    return 2;

  if (std::strcmp(mode, "probe") == 0) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool profile = false;
    const int before = split_count(&profile);
    const bool available = rgpot::CPMDPot::probeAvailable();
    const auto group = rgpot::bindCalculators(2);
    const int splits = profile ? split_count(&profile) - before : -1;
    std::printf("adopt-probe rank %d available %d index %d splits %d\n", rank,
                available, group.index, splits);
    const int rc =
        (available && profile && group.index == rank / 2 && splits == 1) ? 0
                                                                         : 7;
    MPI_Finalize();
    return rc;
  }

  if (std::strcmp(mode, "instance-first") == 0 ||
      std::strcmp(mode, "bound-static") == 0 ||
      std::strcmp(mode, "bound-instance") == 0) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool profile = false;
    const int before = split_count(&profile);
    int idx = -1;
    if (std::strcmp(mode, "instance-first") == 0) {
      {
        rgpot::CPMDPot instance;
      }
      idx = rgpot::bindCalculators(2).index;
    } else {
      idx = rgpot::bindCalculators(2).index;
      if (std::strcmp(mode, "bound-static") == 0)
        idx = rgpot::CPMDPot::bindCalculators(2);
      else {
        {
          rgpot::CPMDPot first;
        }
        {
          rgpot::CPMDPot second;
        }
      }
    }
    const int repeated = rgpot::bindCalculators(2).index;
    const int splits = profile ? split_count(&profile) - before : -1;
    void *handle = dlopen(engine, RTLD_NOW | RTLD_NOLOAD);
    using AdoptedFn = int (*)(void *, std::size_t);
    using CountFn = int (*)();
    auto adopted =
        handle
            ? reinterpret_cast<AdoptedFn>(dlsym(handle, "cpmdc_adopted_comm"))
            : nullptr;
    auto count =
        handle
            ? reinterpret_cast<CountFn>(dlsym(handle, "cpmdc_adopt_call_count"))
            : nullptr;
    MPI_Comm mine = MPI_COMM_NULL, theirs = MPI_COMM_NULL;
    const int got = rgpot::calculatorComm(&mine, sizeof(mine));
    const int copied = adopted ? adopted(&theirs, sizeof(theirs)) : -1;
    int cmp = MPI_UNEQUAL;
    if (got == 1 && copied == 0)
      MPI_Comm_compare(mine, theirs, &cmp);
    const int calls = count ? count() : -1;
    std::printf("adopt-order %s rank %d index %d repeated %d ident %d splits "
                "%d calls %d\n",
                mode, rank, idx, repeated, cmp == MPI_IDENT, splits, calls);
    const int rc = (profile && idx == rank / 2 && repeated == idx &&
                    cmp == MPI_IDENT && splits == 1 && calls == 1)
                       ? 0
                       : 8;
    if (handle)
      dlclose(handle);
    MPI_Finalize();
    return rc;
  }

  if (std::strcmp(mode, "refused-force") == 0) {
    MPI_Init(&argc, &argv);
    rgpot::CPMDPot instance;
    rgpot::addCalculatorHook(refuse_hook);
    const auto group = rgpot::bindCalculators(2);
    if (group.index != -1) {
      MPI_Finalize();
      return 9;
    }
    const double positions[3] = {0.0, 0.0, 0.0};
    const int atoms[1] = {1};
    const double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
    double forces[3]{};
    const rgpot::ForceInput input{1, positions, atoms, box};
    rgpot::ForceOut output{};
    output.F = forces;
    instance.forceImpl(input, &output);
    MPI_Finalize();
    return 10;
  }

  // Keep the engine mapped while inspecting its stored communicator.
  void *keep = dlopen(engine, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (!keep) {
    std::fprintf(stderr, "dlopen %s: %s\n", engine, dlerror());
    return 3;
  }

  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool profile = false;
  const int splits_at_init = split_count(&profile);

  if (std::strcmp(mode, "missing") == 0) {
    const int idx = rgpot::CPMDPot::bindCalculators(1);
    const int splits = profile ? split_count(&profile) - splits_at_init : -1;
    std::printf("adopt-missing rank %d %d splits %d\n", rank, idx, splits);
    const int rc = (profile && idx == -1 && splits == 0) ? 0 : 4;
    MPI_Finalize();
    return rc;
  }

  if (std::strcmp(mode, "refused-hook") == 0) {
    rgpot::addCalculatorHook(refuse_hook);
    rgpot::addCalculatorHook(accept_hook);
    const int idx = rgpot::CPMDPot::bindCalculators(2);
    const int repeated = rgpot::CPMDPot::bindCalculators(2);
    const int splits = profile ? split_count(&profile) - splits_at_init : -1;
    std::printf(
        "adopt-refused rank %d index %d repeated %d hooks %d splits %d\n", rank,
        idx, repeated, hook_calls, splits);
    const int rc = (profile && idx == -1 && repeated == -1 && hook_calls == 2 &&
                    splits == 1)
                       ? 0
                       : 6;
    MPI_Finalize();
    return rc;
  }

  // cp2k-split: the same contract through CP2KPot and the cp2kc shim ABI.
  const bool cp2k = std::strcmp(mode, "cp2k-split") == 0;
  if (std::strcmp(mode, "split") == 0 || cp2k) {
#ifdef RGPOT_HAS_CP2K
    const int idx = cp2k ? rgpot::CP2KPot::bindCalculators(2)
                         : rgpot::CPMDPot::bindCalculators(2);
#else
    const int idx = cp2k ? -1 : rgpot::CPMDPot::bindCalculators(2);
#endif
    const int splits = profile ? split_count(&profile) - splits_at_init : -1;
    MPI_Comm mine = MPI_COMM_NULL;
    const int got = rgpot::calculatorComm(&mine, sizeof(mine));
    using AdoptedFn = int (*)(void *, std::size_t);
    auto *adopted = reinterpret_cast<AdoptedFn>(
        dlsym(keep, cp2k ? "cp2kc_adopted_comm" : "cpmdc_adopted_comm"));
    MPI_Comm theirs = MPI_COMM_NULL;
    const int copied =
        adopted != nullptr ? adopted(&theirs, sizeof(theirs)) : -1;
    int cmp = 0;
    if (got == 1 && copied == 0)
      MPI_Comm_compare(mine, theirs, &cmp);
    const int ident = cmp == MPI_IDENT ? 1 : 0;
    const int expect = rank / 2;
    std::printf("adopt-%s rank %d index %d ident %d splits %d\n", mode, rank,
                idx, ident, splits);
    const int rc =
        (profile && idx == expect && ident == 1 && splits == 1) ? 0 : 5;
    MPI_Finalize();
    return rc;
  }

  MPI_Finalize();
  return 2;
}
