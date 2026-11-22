// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_CACHE
#include "TestTempDir.hpp"
#include "rgpot/PotentialCache.hpp"
#endif
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

// 13 hydrogen atoms from eOn lj38 test (pos.con), positions in Angstrom
// Cell: 101.9424 x 103.1426 x 102.6055 (orthogonal, 90 deg)
// clang-format off
static const double lj13_pos[] = {
    50.594227, 52.017165, 52.277482,
    51.578540, 52.642148, 51.195222,
    51.243758, 52.919086, 52.218383,
    50.490127, 52.736378, 51.417663,
    51.522643, 50.884535, 51.656125,
    50.927263, 51.743087, 51.272643,
    51.240676, 50.960436, 50.573042,
    51.275727, 52.061877, 50.284160,
    50.434110, 50.976423, 51.879062,
    51.695007, 51.919472, 52.038680,
    51.363596, 52.199101, 53.064923,
    52.017483, 51.639074, 51.008389,
    51.187843, 51.165217, 52.678225,
};
// clang-format on
static const int lj13_atmnrs[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static constexpr int N_ATOMS = 13;

TEST_CASE("MetatomicPot LJ energy and forces", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  // Reference values from eOn MetatomicTest (lennard-jones.pt model)
  double expected_energy = 98374.87753058573;
  REQUIRE_THAT(energy, WithinAbs(expected_energy, 1e-2));

  // clang-format off
  double expected_forces_flat[] = {
      -90017.67874663,  14048.63238980,  42667.56998833,
       51715.62564964,  81020.09231365, -33288.72879168,
       13893.56176940,  89793.76258628,  33937.11367155,
      -71755.76235155,  52324.79462503, -32402.45955531,
       46268.61994191, -89837.93451292,  11509.42733453,
      -67770.69814970,  -6678.90441423, -33619.79643378,
      -17329.67691782, -68054.08864133, -59929.39232182,
      -14176.97088206,  30833.36510504, -85864.44933633,
      -75239.50566824, -57104.74927750,  -3715.98865599,
       90169.57785920,   5214.90295971,  30786.57650636,
       20281.84422414,  21840.57389606,  85752.46910389,
      104913.11853287, -11120.68664665, -29664.02315515,
        9047.94473885, -62279.76038293,  73831.68164540,
  };
  // clang-format on

  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(forces(i, j),
                   WithinAbs(expected_forces_flat[i * 3 + j], 1e-2));
    }
  }
}

TEST_CASE("MetatomicPot forces sum to zero", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)energy;
  (void)variance;

  // Newton's third law: force sum should be near zero
  double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
  for (int i = 0; i < N_ATOMS; ++i) {
    fx_sum += forces(i, 0);
    fy_sum += forces(i, 1);
    fz_sum += forces(i, 2);
  }
  REQUIRE_THAT(fx_sum, WithinAbs(0.0, 1.0));
  REQUIRE_THAT(fy_sum, WithinAbs(0.0, 1.0));
  REQUIRE_THAT(fz_sum, WithinAbs(0.0, 1.0));
}

TEST_CASE("MetatomicPot is deterministic across repeated calls",
          "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  // Second call exercises the cached atomic_types tensor path
  auto [e1, f1, v1] = pot(positions, atmtypes, box);
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)v1;
  (void)v2;

  REQUIRE_THAT(e1, WithinAbs(e2, 1e-8));
  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(f1(i, j), WithinAbs(f2(i, j), 1e-6));
    }
  }
}

TEST_CASE("MetatomicPot missing model path throws", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/does-not-exist.pt";
  cfg.device = "cpu";
  REQUIRE_THROWS(rgpot::MetatomicPot(cfg));
}

// Compile-time coverage of vesin_compat traits with synthetic shapes matching
// both historical vesin ABIs. The installed vesin.h is exercised by the
// MetatomicPot LJ energy/forces case (and by parent gpu-scalapack builds).
TEST_CASE("vesin_compat traits distinguish enum vs struct device layouts",
          "[metatomic][vesin_compat]") {
  struct StructDevice {
    int type = 0;
    int device_id = 0;
  };
  enum EnumDevice { EnumCPU = 1 };

  STATIC_REQUIRE(rgpot::vesin_compat::is_device_struct<StructDevice>::value);
  STATIC_REQUIRE_FALSE(rgpot::vesin_compat::is_device_struct<EnumDevice>::value);

  struct OptionsWithAlgorithm {
    int algorithm = 99;
  };
  struct OptionsWithoutAlgorithm {
    double cutoff = 0.0;
  };

  STATIC_REQUIRE(
      rgpot::vesin_compat::has_algorithm_member<OptionsWithAlgorithm>::value);
  STATIC_REQUIRE_FALSE(
      rgpot::vesin_compat::has_algorithm_member<OptionsWithoutAlgorithm>::value);

  OptionsWithAlgorithm with_alg{};
  with_alg.algorithm = 7;
  rgpot::vesin_compat::set_algorithm_default(with_alg);
  REQUIRE(with_alg.algorithm == 0);

  OptionsWithoutAlgorithm without_alg{};
  without_alg.cutoff = 3.5;
  rgpot::vesin_compat::set_algorithm_default(without_alg);
  REQUIRE_THAT(without_alg.cutoff, WithinAbs(3.5, 1e-15));

  // vesin 0.6 skin / n_threads members (and absence on older shapes).
  struct OptionsWithSkinThreads {
    double skin = -1.0;
    int n_threads = -1;
  };
  struct OptionsNoSkin {
    double cutoff = 0.0;
  };
  STATIC_REQUIRE(
      rgpot::vesin_compat::has_skin_member<OptionsWithSkinThreads>::value);
  STATIC_REQUIRE(
      rgpot::vesin_compat::has_n_threads_member<OptionsWithSkinThreads>::value);
  STATIC_REQUIRE_FALSE(
      rgpot::vesin_compat::has_skin_member<OptionsNoSkin>::value);

  OptionsWithSkinThreads with_skin{};
  rgpot::vesin_compat::set_skin(with_skin, 0.25);
  rgpot::vesin_compat::set_n_threads(with_skin, 4);
  REQUIRE_THAT(with_skin.skin, WithinAbs(0.25, 1e-15));
  REQUIRE(with_skin.n_threads == 4);

  // Live VesinOptions from the build's vesin.h: fill_neighbor_options must
  // produce a usable request (and set 0.6 fields when present).
  {
    VesinOptions live{};
    rgpot::vesin_compat::fill_neighbor_options(live, /*cutoff=*/5.0,
                                               /*full_list=*/true);
    REQUIRE_THAT(live.cutoff, WithinAbs(5.0, 1e-15));
    REQUIRE(live.full);
    REQUIRE_FALSE(live.sorted);
    REQUIRE(live.return_shifts);
    REQUIRE_FALSE(live.return_distances);
    REQUIRE(live.return_vectors);
    if constexpr (rgpot::vesin_compat::has_skin_member<VesinOptions>::value) {
      REQUIRE_THAT(live.skin, WithinAbs(0.0, 1e-15));
    }
    if constexpr (rgpot::vesin_compat::has_n_threads_member<
                      VesinOptions>::value) {
      REQUIRE(live.n_threads == 0);
    }
  }

  // Live VesinDevice from the build's vesin.h must construct without error.
  // Branch via a function template so if constexpr can discard the inactive
  // arm (plain if constexpr in this non-template TEST_CASE type-checks both).
  auto check_live_cpu = [](auto device) {
    using Device = decltype(device);
    if constexpr (rgpot::vesin_compat::is_device_struct<Device>::value) {
      REQUIRE(device.type == VesinCPU);
      REQUIRE(device.device_id == 0);
    } else {
      REQUIRE(device == static_cast<Device>(VesinCPU));
    }
  };
  check_live_cpu(rgpot::vesin_compat::make_cpu_device());
}

namespace {

// RAII snapshot of process-global LibTorch flags. Captures on construction
// and restores on destruction so REQUIRE failures, constructor throws, and
// early returns cannot leak deterministic/SDP/TF32/cuDNN settings into later
// cases.
struct TorchContextSnapshot {
  TorchContextSnapshot() {
    auto &ctx = at::globalContext();
    deterministic = ctx.deterministicAlgorithms();
    deterministic_warn_only = ctx.deterministicAlgorithmsWarnOnly();
    fill_uninit = ctx.deterministicFillUninitializedMemory();
    det_cudnn = ctx.deterministicCuDNN();
    benchmark_cudnn = ctx.benchmarkCuDNN();
    flash_sdp = ctx.userEnabledFlashSDP();
    mem_efficient_sdp = ctx.userEnabledMemEfficientSDP();
    math_sdp = ctx.userEnabledMathSDP();
    cudnn_sdp = ctx.userEnabledCuDNNSDP();
    tf32_cublas = ctx.allowTF32CuBLAS();
    tf32_cudnn = ctx.allowTF32CuDNN();
  }

  ~TorchContextSnapshot() { restore(); }

  TorchContextSnapshot(const TorchContextSnapshot &) = delete;
  TorchContextSnapshot &operator=(const TorchContextSnapshot &) = delete;

private:
  void restore() const {
    auto &ctx = at::globalContext();
    ctx.setDeterministicAlgorithms(deterministic, deterministic_warn_only);
    ctx.setDeterministicFillUninitializedMemory(fill_uninit);
    ctx.setDeterministicCuDNN(det_cudnn);
    ctx.setBenchmarkCuDNN(benchmark_cudnn);
    ctx.setSDPUseFlash(flash_sdp);
    ctx.setSDPUseMemEfficient(mem_efficient_sdp);
    ctx.setSDPUseMath(math_sdp);
    ctx.setSDPUseCuDNN(cudnn_sdp);
    ctx.setAllowTF32CuBLAS(tf32_cublas);
    ctx.setAllowTF32CuDNN(tf32_cudnn);
  }

  bool deterministic = false;
  bool deterministic_warn_only = false;
  bool fill_uninit = false;
  bool det_cudnn = false;
  bool benchmark_cudnn = false;
  bool flash_sdp = true;
  bool mem_efficient_sdp = true;
  bool math_sdp = true;
  bool cudnn_sdp = true;
  bool tf32_cublas = true;
  bool tf32_cudnn = true;
};

void seed_nonstrict_torch_context() {
  auto &ctx = at::globalContext();
  ctx.setDeterministicAlgorithms(false, /*warn_only=*/false);
  ctx.setDeterministicFillUninitializedMemory(false);
  ctx.setDeterministicCuDNN(false);
  ctx.setBenchmarkCuDNN(true);
  ctx.setSDPUseFlash(true);
  ctx.setSDPUseMemEfficient(true);
  ctx.setSDPUseCuDNN(true);
  ctx.setSDPUseMath(true);
  // Enable TF32 so Strict must clearly clear both cuBLAS and cuDNN flags.
  ctx.setAllowTF32CuBLAS(true);
  ctx.setAllowTF32CuDNN(true);
}

} // namespace

TEST_CASE("strict torch determinism policy enables det algorithms and math-only SDP",
          "[metatomic][determinism]") {
  TorchContextSnapshot snap;
  seed_nonstrict_torch_context();

  rgpot::apply_torch_determinism_policy(rgpot::TorchDeterminismPolicy::Strict);

  auto &ctx = at::globalContext();
  REQUIRE(ctx.deterministicAlgorithms());
  REQUIRE_FALSE(ctx.deterministicAlgorithmsWarnOnly());
  REQUIRE(ctx.deterministicFillUninitializedMemory());
  REQUIRE(ctx.deterministicCuDNN());
  REQUIRE_FALSE(ctx.benchmarkCuDNN());
  REQUIRE_FALSE(ctx.userEnabledFlashSDP());
  REQUIRE_FALSE(ctx.userEnabledMemEfficientSDP());
  REQUIRE_FALSE(ctx.userEnabledCuDNNSDP());
  REQUIRE(ctx.userEnabledMathSDP());
  REQUIRE_FALSE(ctx.allowTF32CuBLAS());
  REQUIRE_FALSE(ctx.allowTF32CuDNN());
}

TEST_CASE("fast torch determinism policy leaves process-global state alone",
          "[metatomic][determinism]") {
  TorchContextSnapshot snap;
  seed_nonstrict_torch_context();

  // Poison one flag so a no-op Fast path is distinguishable from a restore.
  at::globalContext().setSDPUseFlash(false);
  REQUIRE_FALSE(at::globalContext().userEnabledFlashSDP());
  REQUIRE(at::globalContext().allowTF32CuBLAS());
  REQUIRE(at::globalContext().allowTF32CuDNN());

  rgpot::apply_torch_determinism_policy(rgpot::TorchDeterminismPolicy::Fast);

  auto &ctx = at::globalContext();
  REQUIRE_FALSE(ctx.deterministicAlgorithms());
  REQUIRE_FALSE(ctx.userEnabledFlashSDP());
  REQUIRE(ctx.userEnabledMemEfficientSDP());
  REQUIRE(ctx.userEnabledMathSDP());
  // Fast must not clear TF32 either (still the seeded-true values).
  REQUIRE(ctx.allowTF32CuBLAS());
  REQUIRE(ctx.allowTF32CuDNN());
}

TEST_CASE("MetatomicConfig defaults to Fast torch determinism",
          "[metatomic][determinism]") {
  rgpot::MetatomicConfig cfg;
  REQUIRE(cfg.torch_determinism == rgpot::TorchDeterminismPolicy::Fast);
}

TEST_CASE("MetatomicPot construction applies configured torch determinism policy",
          "[metatomic][determinism]") {
  TorchContextSnapshot snap;
  seed_nonstrict_torch_context();

  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  cfg.torch_determinism = rgpot::TorchDeterminismPolicy::Strict;
  rgpot::MetatomicPot pot(cfg);

  auto &ctx = at::globalContext();
  REQUIRE(ctx.deterministicAlgorithms());
  REQUIRE_FALSE(ctx.deterministicAlgorithmsWarnOnly());
  REQUIRE(ctx.deterministicFillUninitializedMemory());
  REQUIRE(ctx.deterministicCuDNN());
  REQUIRE_FALSE(ctx.benchmarkCuDNN());
  REQUIRE_FALSE(ctx.userEnabledFlashSDP());
  REQUIRE_FALSE(ctx.userEnabledMemEfficientSDP());
  REQUIRE_FALSE(ctx.userEnabledCuDNNSDP());
  REQUIRE(ctx.userEnabledMathSDP());
  REQUIRE_FALSE(ctx.allowTF32CuBLAS());
  REQUIRE_FALSE(ctx.allowTF32CuDNN());
}

// Matched provider-level reproducer: same MetatomicConfig (Strict, fixed SO3
// Monte-Carlo set n=4), same geometry, two successive force evaluations on one
// pot. Seeds are pass-index based (0x50333A5EED + i_rot), so the orientation
// set is identical call-to-call. Any force/energy drift is provider
// nondeterminism, not SO3 RNG. Do NOT cite mismatched CLI (different
// n_symmetry_rotations or torch_determinism) as provider nondeterminism.
//
// The bound is a few ulp rather than equality because that is the guarantee
// the provider actually offers. Instrumenting this path shows rgpot handing
// the model bit-identical inputs -- the positions, cell, and neighbour list
// hash the same on every pass of both calls -- and getting energies back that
// differ by one to two ulp on a call chosen at random. It reproduces with
// OMP_NUM_THREADS=1, with the profiling executor frozen, and with the pair
// vectors copied into torch-owned storage, so it is neither thread-count
// reassociation, nor graph specialization, nor alignment of our buffers; what
// remains is reassociation inside the model evaluation, below
// setDeterministicAlgorithms. Tighten this back to equality if metatensor or
// torch ever grow that guarantee; widening it further would instead be a
// regression worth investigating.
TEST_CASE("Strict MetatomicPot SO3 n=4 holds to a few ulp across matched "
          "force calls",
          "[metatomic][determinism][provider]") {
  TorchContextSnapshot snap;

  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  cfg.torch_determinism = rgpot::TorchDeterminismPolicy::Strict;
  cfg.n_symmetry_rotations = 4;
  cfg.random_rotation = true;
  cfg.so3_probe_scatter = false;
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  auto [e1, f1, v1] = pot(positions, atmtypes, box);
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)v2;

  // Stable to a few ulp on CPU under Strict. The energy is a sum over the
  // whole system, so it is compared in ulp of its own magnitude; the force
  // components straddle zero, where an ulp bound degenerates, so they take an
  // absolute floor scaled to the largest component instead.
  REQUIRE_THAT(e2, Catch::Matchers::WithinULP(e1, 4));

  double f_scale = 0.0;
  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      f_scale = std::max(f_scale, std::abs(f1(i, j)));
    }
  }
  const double f_tol = std::max(1e-12, 1e-12 * f_scale);
  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(f2(i, j), Catch::Matchers::WithinAbs(f1(i, j), f_tol));
    }
  }

  // The variance slot holds the mean-square force deviation over the four
  // orientations. The Lennard-Jones model is rotation invariant, so that
  // deviation is rounding noise: an RMS no larger than f_tol, hence a variance
  // no larger than f_tol squared. Storing the RMS in the slot instead leaves
  // about 1.6e-9, five orders of magnitude above this bound.
  REQUIRE(std::isfinite(v1));
  REQUIRE(v1 >= 0.0);
  REQUIRE(v1 <= f_tol * f_tol);
}

TEST_CASE("Strict MetatomicPot SO3 n=4 CUDA matched forces when available",
          "[metatomic][determinism][provider][cuda]") {
  if (!torch::cuda::is_available()) {
    SKIP("CUDA not available");
  }

  TorchContextSnapshot snap;

  // Matched env requirement for cuBLAS bit-stability (CUDA >= 10.2).
  // Setting here only helps if no prior cuBLAS use in this process.
  // Catch tests that run before other CUDA cases can still validate.
  if (const char *existing = std::getenv("CUBLAS_WORKSPACE_CONFIG");
      existing == nullptr ||
      (std::string(existing) != ":4096:8" &&
       std::string(existing) != ":16:8")) {
    // Prefer the larger workspace; process-local for this test process.
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/1);
  }

  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cuda";
  cfg.length_unit = "angstrom";
  cfg.torch_determinism = rgpot::TorchDeterminismPolicy::Strict;
  cfg.n_symmetry_rotations = 4;
  cfg.random_rotation = true;
  cfg.so3_probe_scatter = false;

  // LJ TorchScript may be CPU-only; skip cleanly if device pick fails.
  try {
    rgpot::MetatomicPot pot(cfg);

    AtomMatrix positions(N_ATOMS, 3);
    for (int i = 0; i < N_ATOMS; ++i)
      for (int j = 0; j < 3; ++j)
        positions(i, j) = lj13_pos[i * 3 + j];

    std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
    std::array<std::array<double, 3>, 3> box = {
        {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

    auto [e1, f1, v1] = pot(positions, atmtypes, box);
    auto [e2, f2, v2] = pot(positions, atmtypes, box);
    (void)v1;
    (void)v2;

    REQUIRE(e1 == e2);
    for (int i = 0; i < N_ATOMS; ++i) {
      for (int j = 0; j < 3; ++j) {
        REQUIRE(f1(i, j) == f2(i, j));
      }
    }
  } catch (const c10::Error &) {
    SKIP("CUDA Metatomic load/eval not supported for this model/build");
  } catch (const std::exception &) {
    SKIP("CUDA Metatomic load/eval not supported for this model/build");
  }
}

#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#include "rgpot/MetatomicPot/metatomic_c_abi.h"
#include <dlfcn.h>

TEST_CASE("MetatomicDlopen loads engine and matches linked pot energy",
          "[metatomic][dlopen]") {
  const char *eng = std::getenv("RGPOT_METATOMIC_ENGINE");
  if (!eng || !*eng) {
    WARN("RGPOT_METATOMIC_ENGINE unset; skip dlopen parity");
    return;
  }
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "lennard-jones.pt";
  cfg.device = "cpu";
  cfg.engine_path = eng;

  rgpot::MetatomicPot linked(cfg);
  rgpot::MetatomicDlopen plugin(cfg);

  // 2-atom toy positions
  double pos[6] = {0, 0, 0, 1.5, 0, 0};
  int z[2] = {1, 1};
  double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  double F1[6]{}, F2[6]{};
  rgpot::ForceOut o1{F1, 0, 0}, o2{F2, 0, 0};
  rgpot::ForceInput in{2, pos, z, box};
  linked.forceImpl(in, &o1);
  plugin.forceImpl(in, &o2);
  REQUIRE(std::isfinite(o1.energy));
  REQUIRE(std::isfinite(o2.energy));
  REQUIRE(o1.energy == Catch::Approx(o2.energy).margin(1e-5));
}

#ifdef RGPOT_HAS_CACHE
TEST_CASE("MetatomicPot cache key follows charge and spin",
          "[metatomic][cache]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];
  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};
  const rgpot::ForceInput input{static_cast<size_t>(N_ATOMS), positions.data(),
                                atmtypes.data(), box.front().data()};
  const auto key_for = [&] {
    return rgpot::cache::result_key(input, rgpot::PotType::Metatomic,
                                    pot.paramsKey())
        .hash;
  };

  const uint64_t first_params = pot.paramsKey();
  const uint64_t first_key = key_for();
  REQUIRE(first_params != 0);

  rgpot::tests::TestTempDir directory("rgpot-metatomic-key");
  rgpot::cache::PotentialCache cache((directory.path() / "results").string());
  pot.setCache(&cache);

  (void)pot(positions, atmtypes, box);
  REQUIRE(cache.counts().computed == 1);
  REQUIRE(cache.counts().served == 0);
  (void)pot(positions, atmtypes, box);
  REQUIRE(cache.counts().computed == 1);
  REQUIRE(cache.counts().served == 1);

  pot.setChargeSpin(-1, 3);
  REQUIRE(pot.paramsKey() != first_params);
  REQUIRE(key_for() != first_key);
  (void)pot(positions, atmtypes, box);
  REQUIRE(cache.counts().computed == 2);
  REQUIRE(cache.counts().served == 1);
  (void)pot(positions, atmtypes, box);
  REQUIRE(cache.counts().computed == 2);
  REQUIRE(cache.counts().served == 2);

  pot.setChargeSpin(-1, 1);
  const uint64_t charge_only = key_for();
  pot.setChargeSpin(0, 3);
  const uint64_t spin_only = key_for();
  REQUIRE(charge_only != first_key);
  REQUIRE(spin_only != first_key);
  REQUIRE(charge_only != spin_only);

  pot.setChargeSpin(0, 1);
  REQUIRE(pot.paramsKey() == first_params);
  REQUIRE(key_for() == first_key);
  (void)pot(positions, atmtypes, box);
  REQUIRE(cache.counts().computed == 2);
  REQUIRE(cache.counts().served == 3);
  pot.setCache(nullptr);
}
#endif
