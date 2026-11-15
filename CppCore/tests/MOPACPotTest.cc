// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cstdlib>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using rgpot::types::AtomMatrix;

namespace {

AtomMatrix threeAtoms() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {1.15, 0.0, 0.0}, {-1.06, 0.0, 0.0}};
}

std::array<std::array<double, 3>, 3> vacuumBox() {
  return {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

} // namespace

TEST_CASE("MOPACPot probeAvailable with fake engine", "[mopac]") {
  REQUIRE(rgpot::MOPACPot::probeAvailable());
}

TEST_CASE("MOPACPot converts Hartree and Hartree/Bohr", "[mopac]") {
  rgpot::MOPACPot::Config cfg;
  cfg.charge = 0;
  cfg.model = 4;
  rgpot::MOPACPot pot(cfg);
  REQUIRE(pot.available());

  const std::vector<int> z{1, 6, 7};
  auto [energy, forces, variance] = pot(threeAtoms(), z, vacuumBox());
  (void)variance;
  REQUIRE(energy == Approx(0.25 * rgpot::units::HARTREE_TO_EV));
  REQUIRE(forces(0, 0) == Approx(0.001 * rgpot::units::NEG_GRAD_TO_FORCE));
}

TEST_CASE("MOPACPot packed charge reaches the engine", "[mopac]") {
  rgpot::MOPACPot::Config cfg;
  cfg.charge = 2;
  cfg.model = 4;
  rgpot::MOPACPot pot(cfg);
  REQUIRE(pot.available());
  const std::vector<int> z{1, 1, 8};
  auto [energy, forces, variance] = pot(threeAtoms(), z, vacuumBox());
  (void)forces;
  (void)variance;
  REQUIRE(energy == Approx((0.25 + 0.02) * rgpot::units::HARTREE_TO_EV));
}

// Hidden: meson runs it in a process of its own as MOPACPotEmptyVariable.
// An empty path opens the running program, which resolves the engine
// symbols once another case has opened the fake engine RTLD_GLOBAL.
TEST_CASE("MOPACPot skips an empty MOPACC_LIBRARY",
          "[.][mopac][empty-variable]") {
  setenv("MOPACC_LIBRARY", "", 1);
  rgpot::MOPACPot pot(rgpot::MOPACPot::Config{});
  const bool available = pot.available();
  unsetenv("MOPACC_LIBRARY");
  REQUIRE(available);
}
