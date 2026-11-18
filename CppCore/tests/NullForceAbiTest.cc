// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
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

rgpot_status_t callTrampoline(rgpot::LJPot &pot, rgpot_force_input_t *input) {
  auto handle = rgpot::PotentialHandle::from_impl(pot);
  auto *raw = handle.raw();
  rgpot_force_out_t output = rgpot_force_out_create();
  const rgpot_status_t status =
      raw->callback(raw->pot_user_data, input, &output);
  rgpot_tensor_free(output.forces);
  return status;
}

} // namespace

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
