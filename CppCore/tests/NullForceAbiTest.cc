// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#ifdef RGPOT_HAS_RUST_CORE
#include "rgpot/potential.hpp"
#endif

using rgpot::types::AtomMatrix;

namespace {

void requireRejected(auto &&call, const char *message) {
  try {
    call();
    FAIL("the call was accepted");
  } catch (const std::invalid_argument &ex) {
    REQUIRE(std::string(ex.what()) == message);
  }
}

std::array<std::array<double, 3>, 3> cell() {
  return {{{{20.0, 0.0, 0.0}}, {{0.0, 20.0, 0.0}}, {{0.0, 0.0, 20.0}}}};
}

void fillCell(double *box) {
  const double values[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  for (int i = 0; i < 9; ++i) {
    box[i] = values[i];
  }
}

template <bool Periodic> struct AnalyticProbe {
  mutable int hessianCalls = 0;

  [[nodiscard]] rgpot::PotCaps caps() const {
    rgpot::PotCaps c;
    c.periodic = Periodic;
    c.analyticHvp = true;
    return c;
  }

  void forceImpl(const rgpot::ForceInput &in, rgpot::ForceOut *out) const {
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      out->F[i] = 0.0;
    }
    out->energy = 0.0;
  }

  void hessianVectorImpl(const rgpot::ForceInput &in, const double *v,
                         double *hv) const {
    ++hessianCalls;
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      hv[i] = v[i];
    }
  }
};

} // namespace

TEST_CASE("zero atoms are not reported as null forces", "[null-force]") {
  double pos[3] = {0.0, 0.0, 0.0};
  int species = 1;
  double box[9];
  fillCell(box);
  rgpot::ForceInput in{0, pos, &species, box};
  rgpot::ForceOut out{};
  REQUIRE(rgpot::rejectForceCall(in, &out, true) == rgpot::kInvalidParameter);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: nAtoms must be positive");

  double forces[3] = {0.0, 0.0, 0.0};
  out.F = forces;
  REQUIRE(rgpot::rejectForceCall(in, &out, true) == rgpot::kInvalidParameter);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: nAtoms must be positive");
}

TEST_CASE("a successful check keeps the previous failure", "[null-force]") {
  rgpot::ForceInput empty{0, nullptr, nullptr, nullptr};
  REQUIRE(rgpot::rejectForceCall(empty, nullptr, true) ==
          rgpot::kInvalidParameter);

  double pos[3] = {0.0, 0.0, 0.0};
  int species = 1;
  double box[9];
  fillCell(box);
  double forces[3] = {0.0, 0.0, 0.0};
  rgpot::ForceInput ok{1, pos, &species, box};
  rgpot::ForceOut out{};
  out.F = forces;
  REQUIRE(rgpot::rejectForceCall(ok, &out, false) == 0);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: nAtoms must be positive");

  rgpot::ForceInput noBox{1, pos, &species, nullptr};
  REQUIRE(rgpot::rejectForceCall(noBox, &out, false) == 0);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: nAtoms must be positive");
}

TEST_CASE("LJ operator rejects a missing species buffer", "[null-force]") {
  rgpot::LJPot pot;
  AtomMatrix pos = AtomMatrix::Zero(2, 3);
  pos(1, 0) = 1.5;
  std::vector<int> types;
  requireRejected(
      [&] {
        auto ignored = pot(pos, types, cell());
        (void)ignored;
      },
      "invalid parameter: atomic numbers are null");
}

TEST_CASE("LJ operator rejects a system with no atoms", "[null-force]") {
  rgpot::LJPot pot;
  AtomMatrix pos(0, 3);
  std::vector<int> types;
  requireRejected(
      [&] {
        auto ignored = pot(pos, types, cell());
        (void)ignored;
      },
      "invalid parameter: nAtoms must be positive");
}

TEST_CASE("LJ forceBatch rejects a null system array", "[null-force]") {
  rgpot::LJPot pot;
  rgpot::ForceBatch none{0, nullptr, nullptr};
  REQUIRE_NOTHROW(pot.forceBatch(none));

  rgpot::ForceBatch batch{1, nullptr, nullptr};
  requireRejected([&] { pot.forceBatch(batch); },
                  "invalid parameter: batch input or output is null");
}

TEST_CASE("LJ forceBatch rejects null positions", "[null-force]") {
  rgpot::LJPot pot;
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  double forces[6] = {};
  rgpot::ForceInput in{2, nullptr, species, box};
  rgpot::ForceOut out{};
  out.F = forces;
  rgpot::ForceBatch batch{1, &in, &out};
  requireRejected([&] { pot.forceBatch(batch); },
                  "invalid parameter: positions is null");
}

TEST_CASE("LJ forceBatch names zero atoms before a null force buffer",
          "[null-force]") {
  rgpot::LJPot pot;
  double box[9];
  fillCell(box);
  rgpot::ForceInput in{0, nullptr, nullptr, box};
  rgpot::ForceOut out{};
  rgpot::ForceBatch batch{1, &in, &out};
  requireRejected([&] { pot.forceBatch(batch); },
                  "invalid parameter: nAtoms must be positive");
}

TEST_CASE("a free cluster accepts a null cell", "[null-force]") {
  rgpot::LJClusterPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  int species[2] = {1, 1};
  double forces[6] = {};
  rgpot::ForceInput in{2, pos, species, nullptr};
  rgpot::ForceOut out{};
  out.F = forces;
  rgpot::ForceBatch batch{1, &in, &out};
  REQUIRE_NOTHROW(pot.forceBatch(batch));
}

TEST_CASE("a molecular dispersion kernel accepts a null cell", "[null-force]") {
  double pos[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  int species[2] = {1, 1};
  double forces[6] = {};
  const rgpot::ForceInput in{2, pos, species, nullptr};
  rgpot::ForceOut out{};
  out.F = forces;
  rgpot::ForceBatch batch{1, &in, &out};

  rgpot::D2Pot d2;
  REQUIRE_NOTHROW(d2.forceBatch(batch));
  rgpot::D2CpmdPot d2Cpmd;
  REQUIRE_NOTHROW(d2Cpmd.forceBatch(batch));

  REQUIRE(rgpot::kernelCellOptional<rgpot::D2Pot>());
  REQUIRE(rgpot::kernelCellOptional<rgpot::D2CpmdPot>());
  REQUIRE_FALSE(rgpot::kernelCellOptional<rgpot::LJPot>());
  REQUIRE_FALSE(rgpot::kernelNeedsCell(d2));
  REQUIRE(rgpot::kernelNeedsCell(rgpot::LJPot{}));
  REQUIRE_FALSE(rgpot::kernelNeedsCell(rgpot::LJClusterPot{}));
}

TEST_CASE("a molecular dispersion kernel still rejects a missing species buffer",
          "[null-force]") {
  double pos[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  double forces[6] = {};
  const rgpot::ForceInput in{2, pos, nullptr, nullptr};
  rgpot::ForceOut out{};
  out.F = forces;
  rgpot::ForceBatch batch{1, &in, &out};
  rgpot::D2Pot d2;
  requireRejected([&] { d2.forceBatch(batch); },
                  "invalid parameter: atomic numbers are null");
}

TEST_CASE("per-atom evaluation rejects a missing species buffer",
          "[null-force]") {
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  double box[9];
  fillCell(box);
  double forces[6] = {};
  double energy[2] = {};
  rgpot::ForceInput in{2, pos, nullptr, box};
  rgpot::ForceOut out{};
  out.F = forces;
  requireRejected(
      [&] {
        rgpot::invokeForcePerAtom(pot, in, &out,
                                  rgpot::PerAtomOut{energy, nullptr});
      },
      "invalid parameter: atomic numbers are null");
}

namespace {

/// A direct forcePerAtom call with one unusable buffer throws and leaves
/// the force and per-atom arrays as they were.
template <typename Pot>
void requirePerAtomGuard(const Pot &pot, bool readsSpecies) {
  constexpr double kSentinel = 7.0;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);

  auto attempt = [&](const rgpot::ForceInput &in, bool withForces,
                     const char *message) {
    double forces[6];
    double energy[2];
    double virial[12];
    std::fill(forces, forces + 6, kSentinel);
    std::fill(energy, energy + 2, kSentinel);
    std::fill(virial, virial + 12, kSentinel);
    rgpot::ForceOut out{};
    out.F = withForces ? forces : nullptr;
    out.energy = kSentinel;
    requireRejected(
        [&] { pot.forcePerAtom(in, &out, rgpot::PerAtomOut{energy, virial}); },
        message);
    REQUIRE(out.energy == kSentinel);
    for (const double f : forces) {
      REQUIRE(f == kSentinel);
    }
    for (const double e : energy) {
      REQUIRE(e == kSentinel);
    }
    for (const double w : virial) {
      REQUIRE(w == kSentinel);
    }
  };

  attempt({2, nullptr, species, box}, true,
          "invalid parameter: positions is null");
  attempt({2, pos, species, box}, false, "invalid parameter: forces are null");
  attempt({0, pos, species, box}, true,
          "invalid parameter: nAtoms must be positive");

  if (readsSpecies) {
    attempt({2, pos, nullptr, box}, true,
            "invalid parameter: atomic numbers are null");
  } else {
    double forces[6] = {};
    double energy[2] = {};
    rgpot::ForceOut out{};
    out.F = forces;
    REQUIRE_NOTHROW(pot.forcePerAtom({2, pos, nullptr, box}, &out,
                                     rgpot::PerAtomOut{energy, nullptr}));
  }

  rgpot::ForceOut none{};
  REQUIRE_THROWS_AS(pot.forcePerAtom({2, pos, species, box}, nullptr,
                                     rgpot::PerAtomOut{}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pot.forcePerAtom({2, pos, species, box}, &none, rgpot::PerAtomOut{}),
      std::invalid_argument);
}

} // namespace

TEST_CASE("direct forcePerAtom members validate before writing",
          "[null-force]") {
  requirePerAtomGuard(rgpot::LJPot{}, false);
  requirePerAtomGuard(rgpot::LJClusterPot{}, false);
  requirePerAtomGuard(rgpot::MorsePot{}, false);
  requirePerAtomGuard(rgpot::ZBLPot{}, true);
  requirePerAtomGuard(rgpot::SuttonChenPot{}, false);
}

TEST_CASE("cellVolume gives zero for a cell without a volume", "[null-force]") {
  double box[9];
  fillCell(box);
  REQUIRE(rgpot::cellVolume(box) == 8000.0);

  REQUIRE(rgpot::cellVolume(nullptr) == 0.0);

  double empty[9] = {};
  REQUIRE(rgpot::cellVolume(empty) == 0.0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    for (int k : {0, 4, 8, 1, 5}) {
      double broken[9];
      fillCell(broken);
      broken[k] = bad;
      const double volume = rgpot::cellVolume(broken);
      REQUIRE(volume == 0.0);
      REQUIRE_FALSE(std::isnan(volume));
    }
  }

  double huge[9] = {1.0e200, 0.0, 0.0, 0.0, 1.0e200, 0.0, 0.0, 0.0, 1.0e200};
  REQUIRE(rgpot::cellVolume(huge) == 0.0);
}

TEST_CASE("a cell without a volume publishes no stress", "[null-force]") {
  const double inf = std::numeric_limits<double>::infinity();
  double box[9];
  fillCell(box);
  box[0] = inf;
  const double strain[6] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  rgpot::ForceOut out{};
  out.has_stress = 1;
  rgpot::publishCauchyStress(&out, strain, rgpot::cellVolume(box));
  REQUIRE(out.has_stress == 0);
  for (const double s : out.stress) {
    REQUIRE(s == 0.0);
  }
}

TEST_CASE("a Hessian product rejects a missing species buffer",
          "[null-force]") {
  rgpot::HessianVectorCache cache;
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  double box[9];
  fillCell(box);
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};
  rgpot::ForceInput in{2, pos, nullptr, box};
  requireRejected([&] { cache.apply(pot, in, direction, product); },
                  "invalid parameter: atomic numbers are null");
}

TEST_CASE("a Hessian product rejects a missing cell on a periodic kernel",
          "[null-force]") {
  rgpot::HessianVectorCache cache;
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};
  rgpot::ForceInput in{2, pos, species, nullptr};
  requireRejected([&] { cache.apply(pot, in, direction, product); },
                  "invalid parameter: box is null");
}

TEST_CASE("an analytic Hessian product rejects a missing species buffer",
          "[null-force]") {
  rgpot::HessianVectorCache cache;
  AnalyticProbe<true> pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  double box[9];
  fillCell(box);
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};
  rgpot::ForceInput in{2, pos, nullptr, box};
  requireRejected(
      [&] {
        cache.apply(pot, in, direction, product, 0.0,
                    rgpot::HvpScheme::Analytic);
      },
      "invalid parameter: atomic numbers are null");
  REQUIRE(pot.hessianCalls == 0);

  int species[2] = {1, 1};
  in.atmnrs = species;
  cache.apply(pot, in, direction, product, 0.0, rgpot::HvpScheme::Analytic);
  REQUIRE(pot.hessianCalls == 1);
  REQUIRE(product[0] == 1.0);
}

TEST_CASE("an analytic Hessian product rejects a null cell only when periodic",
          "[null-force]") {
  rgpot::HessianVectorCache cache;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};
  rgpot::ForceInput in{2, pos, species, nullptr};

  AnalyticProbe<true> periodic;
  requireRejected([&] { cache.apply(periodic, in, direction, product); },
                  "invalid parameter: box is null");
  REQUIRE(periodic.hessianCalls == 0);

  AnalyticProbe<false> cluster;
  REQUIRE_NOTHROW(cache.apply(cluster, in, direction, product));
  REQUIRE(cluster.hessianCalls == 1);
}

TEST_CASE("the built-in LJ plugin rejects null positions", "[null-force]") {
  auto &registry = rgpot::plugin::PluginRegistry::instance();
  const auto *lj = registry.find("lennard-jones");
  REQUIRE(lj != nullptr);
  REQUIRE(lj->desc->last_error != nullptr);
  rgpot_plugin_instance *instance = nullptr;
  REQUIRE(lj->desc->create("{}", &instance) == RGPOT_PLUGIN_OK);

  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  double forces[6] = {};
  rgpot_plugin_input_t input{};
  input.n_atoms = 2;
  input.atomic_numbers = species;
  input.cell = box;
  rgpot_plugin_output_t output{};
  output.forces = forces;
  REQUIRE(lj->desc->calculate(instance, &input, &output) ==
          RGPOT_PLUGIN_INVALID_PARAM);
  REQUIRE(std::string(lj->desc->last_error(instance)) ==
          "invalid parameter: positions is null");

  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  input.positions = pos;
  REQUIRE(lj->desc->calculate(instance, &input, &output) == RGPOT_PLUGIN_OK);
  REQUIRE(std::string(lj->desc->last_error(instance)) ==
          "invalid parameter: positions is null");

  input.n_atoms = 0;
  input.positions = nullptr;
  output.forces = nullptr;
  REQUIRE(lj->desc->calculate(instance, &input, &output) ==
          RGPOT_PLUGIN_INVALID_PARAM);
  REQUIRE(std::string(lj->desc->last_error(instance)) ==
          "invalid parameter: nAtoms must be positive");
  lj->desc->destroy(instance);
}

#ifdef RGPOT_HAS_RUST_CORE

namespace {

template <class Pot>
rgpot_status_t callTrampoline(Pot &pot, rgpot_force_input_t *input) {
  auto handle = rgpot::PotentialHandle::from_impl(pot);
  auto *raw = handle.raw();
  rgpot_force_out_t output = rgpot_force_out_create();
  const rgpot_status_t status =
      raw->callback(raw->pot_user_data, input, &output);
  rgpot_tensor_free(output.forces);
  return status;
}

} // namespace

namespace rgpot {
struct TrampolineAccess {
  template <typename Impl>
  static HessianVectorCallback hessianVector() {
    return &PotentialHandle::hvp_trampoline<Impl>;
  }
};
} // namespace rgpot

TEST_CASE("the Hessian trampoline rejects null pointer arguments",
          "[null-force]") {
  using Probe = AnalyticProbe<true>;
  const auto callback = rgpot::TrampolineAccess::hessianVector<Probe>();
  Probe pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  rgpot_force_input_t input = rgpot_force_input_create(2, pos, species, box);
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};

  REQUIRE(callback(nullptr, &input, direction, product) ==
          RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: potential is null");

  REQUIRE(callback(&pot, &input, nullptr, product) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: direction or output is null");

  REQUIRE(callback(&pot, &input, direction, nullptr) ==
          RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: direction or output is null");
  REQUIRE(pot.hessianCalls == 0);

  REQUIRE(callback(&pot, &input, direction, product) == RGPOT_SUCCESS);
  REQUIRE(pot.hessianCalls == 1);
  rgpot_force_input_free(&input);
}

TEST_CASE("the LJ trampoline rejects null species data", "[null-force]") {
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  rgpot_force_input_t input = rgpot_force_input_create(2, pos, species, box);
  input.atomic_numbers->dl_tensor.data = nullptr;
  REQUIRE(callTrampoline(pot, &input) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: atomic numbers are null");
  rgpot_force_input_free(&input);
}

TEST_CASE("the LJ trampoline rejects null position data", "[null-force]") {
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  rgpot_force_input_t input = rgpot_force_input_create(2, pos, species, box);
  input.positions->dl_tensor.data = nullptr;
  REQUIRE(callTrampoline(pot, &input) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: positions is null");
  rgpot_force_input_free(&input);
}

TEST_CASE("the LJ trampoline rejects a null cell", "[null-force]") {
  rgpot::LJPot pot;
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  rgpot_force_input_t input = rgpot_force_input_create(2, pos, species, box);
  input.box_matrix->dl_tensor.data = nullptr;
  REQUIRE(callTrampoline(pot, &input) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) == "invalid parameter: box is null");
  rgpot_force_input_free(&input);
}

TEST_CASE("the LJ HVP trampoline rejects a null cell", "[null-force]") {
  rgpot::LJPot pot;
  auto handle = rgpot::PotentialHandle::from_impl(pot);
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double direction[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double product[6] = {};
  rgpot_force_input_t input =
      rgpot_force_input_create(2, pos, species, nullptr);
  rgpot_tensor_free(input.box_matrix);
  input.box_matrix = nullptr;
  REQUIRE(rgpot_potential_hvp(handle.raw(), &input, direction, 0.0,
                              RGPOT_HVP_ANALYTIC,
                              product) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) == "invalid parameter: box is null");
  rgpot_force_input_free(&input);
}

TEST_CASE("the D2 trampoline accepts a null cell", "[null-force]") {
  rgpot::D2Pot pot;
  double pos[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  int species[2] = {1, 1};
  rgpot_force_input_t input =
      rgpot_force_input_create(2, pos, species, nullptr);
  REQUIRE(callTrampoline(pot, &input) == RGPOT_SUCCESS);
  rgpot_tensor_free(input.box_matrix);
  input.box_matrix = nullptr;
  REQUIRE(callTrampoline(pot, &input) == RGPOT_SUCCESS);
  rgpot_force_input_free(&input);
}

TEST_CASE("the LJ trampoline names zero atoms before forces", "[null-force]") {
  rgpot::LJPot pot;
  double box[9];
  fillCell(box);
  rgpot_force_input_t input =
      rgpot_force_input_create(0, nullptr, nullptr, box);
  REQUIRE(callTrampoline(pot, &input) == RGPOT_INVALID_PARAMETER);
  REQUIRE(std::string(rgpot_last_error()) ==
          "invalid parameter: nAtoms must be positive");
  rgpot_force_input_free(&input);
}

#endif
