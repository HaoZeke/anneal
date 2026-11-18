// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#if !defined(_WIN32)
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#endif

#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/calculator_mpi_abi.h"

// A process started without mpirun is one calculator of one rank, with
// or without MPI in the build. A second bind with another size is
// refused because the world is split once.
// mpirun -n 4 runs this binary with one rank per calculator. Rank 3's
// buffer matches the values rank 0 filled. RGPOT_CALCULATOR_BAD_OWNER=1
// asks every rank for an owner outside the calculator count, and that
// call aborts the world.
TEST_CASE("One process is one calculator", "[CalculatorGroup]") {
  const rgpot::CalculatorGroup g = rgpot::bindCalculators(1);
  if (g.world_size == 1) {
    REQUIRE(g.index == 0);
    REQUIRE(g.ranks == 1);
    REQUIRE(g.world_size == 1);
    REQUIRE(rgpot::calculatorCount() == 1);
    REQUIRE(rgpot::calculatorWorldSize() == 1);
#ifdef RGPOT_HAS_MPI
    REQUIRE(rgpot::calculatorsUseMpi() == 1);
#else
    REQUIRE(rgpot::calculatorsUseMpi() == 0);
#endif
    REQUIRE(rgpot::thisCalculator().index == 0);

    std::array<double, 4> buf{1.0, -2.0, 3.5, 0.25};
    const int shared = rgpot::shareFromCalculator(0, buf.data(), sizeof(buf));
#ifdef RGPOT_HAS_MPI
    REQUIRE(shared == 1);
#else
    REQUIRE(shared == 0);
#endif
    REQUIRE(buf == std::array<double, 4>{1.0, -2.0, 3.5, 0.25});
    REQUIRE(rgpot::shareFromCalculator(1, buf.data(), sizeof(buf)) == 0);

    REQUIRE(rgpot::bindCalculators(1).index == 0);
    // 0 or less is one calculator for the whole world, the split above.
    REQUIRE(rgpot::bindCalculators(0).index == 0);
    REQUIRE(rgpot::bindCalculators(-1).index == 0);
    REQUIRE(rgpot::bindCalculators(2).index == -1);
  } else {
    rgpot::finalizeMpiAtExit();
    REQUIRE(g.world_size == 4);
    REQUIRE(g.ranks == 1);
    REQUIRE(g.index >= 0);
    REQUIRE(g.index < 4);
    REQUIRE(rgpot::calculatorCount() == 4);
    REQUIRE(rgpot::calculatorWorldSize() == 4);
    REQUIRE(rgpot::thisCalculator().index == g.index);
#ifdef RGPOT_HAS_MPI
    REQUIRE(rgpot::calculatorsUseMpi() == 1);
#endif
    REQUIRE(rgpot::bindCalculators(1).index == g.index);
    REQUIRE(rgpot::bindCalculators(0).index == -1);

    const char *bad = std::getenv("RGPOT_CALCULATOR_BAD_OWNER");
    if (bad != nullptr && bad[0] == '1' && bad[1] == '\0') {
      std::array<double, 4> buf{1.0, -2.0, 3.5, 0.25};
      const int shared =
          rgpot::shareFromCalculator(99, buf.data(), sizeof(buf));
      std::fprintf(stderr, "bad owner returned %d on index %d\n", shared,
                   g.index);
      REQUIRE(false);
    }

    std::array<double, 4> buf{0.0, 0.0, 0.0, 0.0};
    if (g.index == 0)
      buf = {1.0, -2.0, 3.5, 0.25};
    const int shared = rgpot::shareFromCalculator(0, buf.data(), sizeof(buf));
    REQUIRE(shared == 1);
    REQUIRE(buf == std::array<double, 4>{1.0, -2.0, 3.5, 0.25});
    if (g.index == 3) {
      std::printf("calculator-share rank 3 %.6f %.6f %.6f %.6f\n", buf[0],
                  buf[1], buf[2], buf[3]);
      std::fflush(stdout);
    }
  }
}

// The calculator-group C ABI must be reachable under unmangled names.
// Where the build produced a shared librgpot (RGPOT_TEST_SHARED_LIBRARY),
// the library next to this binary is opened and must resolve all three
// symbols. A static build has no shared object to open, so the same
// names are looked up in the test binary itself, which the build links
// with exported symbols. If that binary does not export them either, the
// check is skipped with the reason instead of passing.
#if !defined(_WIN32)
namespace {
constexpr std::array<const char *, 3> kCalculatorAbiSymbols = {
    "rgpot_mpi_host_abi_version", "rgpot_mpi_bind_calculators",
    "rgpot_mpi_share_bytes"};

// The first symbol the handle does not resolve, or nullptr if it has all.
const char *firstMissing(void *handle) {
  for (const char *name : kCalculatorAbiSymbols) {
    if (dlsym(handle, name) == nullptr)
      return name;
  }
  return nullptr;
}
} // namespace
#endif

TEST_CASE("calculator-group C ABI exports unmangled symbols",
          "[CalculatorGroup][abi]") {
#if defined(_WIN32)
  SKIP("the symbol probe uses dlsym, which this platform lacks");
#elif defined(RGPOT_TEST_SHARED_LIBRARY)
  char exe[PATH_MAX] = {0};
  const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  std::string path(exe, static_cast<size_t>(n));
  const std::size_t slash = path.find_last_of('/');
  REQUIRE(slash != std::string::npos);
  const std::string libdir = path.substr(0, slash + 1);
  path = libdir + "librgpot.so.3";
  void *lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    path = libdir + "librgpot.so";
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  }
  INFO("shared library searched in " << libdir);
  REQUIRE(lib != nullptr);
  const char *missing = firstMissing(lib);
  dlclose(lib);
  INFO("symbol missing from " << path);
  REQUIRE(missing == nullptr);
#else
  void *self = dlopen(nullptr, RTLD_NOW);
  const char *missing =
      self == nullptr ? "dlopen(nullptr)" : firstMissing(self);
  if (self != nullptr)
    dlclose(self);
  if (missing != nullptr) {
    SKIP("librgpot is static in this build, so there is no shared object "
         "to open, and the test binary does not export "
         << missing << " to dlsym");
  }
  REQUIRE(missing == nullptr);
#endif
}

TEST_CASE("calculator-group C ABI binds and shares through the C symbols",
          "[CalculatorGroup][abi]") {
  rgpot_mpi_group_t group{};
  const int index = rgpot_mpi_bind_calculators(1, &group);
  REQUIRE(index == group.index);
  REQUIRE(rgpot_mpi_host_abi_version() == RGPOT_MPI_ABI_VERSION);
  std::array<double, 2> values{4.0, -1.0};
  const int rc = rgpot_mpi_share_bytes(0, values.data(), sizeof(values));
  if (group.world_size == 1) {
    REQUIRE((rc == 0 || rc == 1));
  } else {
    REQUIRE(rc == 1);
  }
}
