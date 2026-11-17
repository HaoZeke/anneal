// MIT License
// Copyright 2023--present rgpot developers
//
// Morse pot checks: the analytic two-atom limit, a 13-atom regression pin,
// and the parameter fingerprint.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

/// eOn's neb_morse reactant geometry (13 hydrogens, client/unit_tests/data/
/// systems/neb_morse/reactant.con), inlined because rgpot reads no .con.
AtomMatrix nebMorseCluster() {
  return AtomMatrix{
      {50.594227, 52.017165, 52.277482}, {51.578540, 52.642148, 51.195222},
      {51.243758, 52.919086, 52.218383}, {50.490127, 52.736378, 51.417663},
      {51.522643, 50.884535, 51.656125}, {50.927263, 51.743087, 51.272643},
      {51.240676, 50.960436, 50.573042}, {51.275727, 52.061877, 50.284160},
      {50.434110, 50.976423, 51.879062}, {51.695007, 51.919472, 52.038680},
      {51.363596, 52.199101, 53.064923}, {52.017483, 51.639074, 51.008389},
      {51.187843, 51.165217, 52.678225}};
}

/// The cell that ships with the geometry above.
std::array<std::array<double, 3>, 3> nebMorseCell() {
  return {
      {{101.942400, 0.0, 0.0}, {0.0, 103.142600, 0.0}, {0.0, 0.0, 102.605500}}};
}

double maxForceNorm(const AtomMatrix &forces) {
  double worst = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

double columnSum(const AtomMatrix &forces, size_t col) {
  double acc = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    acc += forces(i, col);
  }
  return acc;
}

} // namespace

TEST_CASE("MorsePot two-atom well depth", "[MorsePot]") {
  rgpot::MorsePot pot;
  const auto cfg = pot.config();

  // Cutoff shift: the pair term at the minimum is -De, shifted by the
  // unshifted energy at the cutoff. Two terms near De cancel to 3.6e-5, so
  // the comparison is absolute at the operands' rounding scale: a relative
  // pin on the cancelled result asks for agreement below one ulp of De, and
  // exp() and multiply-add contraction differ by that much between an x86
  // and an aarch64 build of the same expression.
  const double d = 1.0 - std::exp(-cfg.a * (cfg.cutoff - cfg.re));
  const double shift = cfg.De * d * d - cfg.De;
  REQUIRE_THAT(pot.energyShift(),
               WithinAbs(shift, 8 * std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(pot.energyShift(), WithinAbs(-3.5537997279178057e-05, 1e-14));

  const AtomMatrix positions{{10.0, 10.0, 10.0}, {10.0 + cfg.re, 10.0, 10.0}};
  const std::vector<int> atmtypes{78, 78};
  // Cutoff 9.5 A needs a cell wider than 19 A for the minimum image
  // convention to leave a single pair.
  const std::array<std::array<double, 3>, 3> box{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-cfg.De - shift, 1e-12));
  REQUIRE_THAT(energy, WithinAbs(-0.71016446200272088, 1e-12));
  // r == re is the minimum of the pair term: the force vanishes there.
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(0.0, 1e-12));
    }
  }
}

TEST_CASE("MorsePot energy and forces on the neb_morse cluster", "[MorsePot]") {
  // Reference produced by an independent double-precision evaluation of
  // this kernel (eOn pins only finiteness for Morse on this geometry).
  // Platinum parameters on hydrogens: deterministic, not physical.
  rgpot::MorsePot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);

  auto [energy, forces, variance] = pot(positions, atmtypes, nebMorseCell());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(7483.1166203812, 1e-9));

  const std::array<double, 3> expected_row0{-2056.95286055812, 333.678513150438,
                                            1022.83140169994};
  for (size_t j = 0; j < 3; ++j) {
    REQUIRE_THAT(forces(0, j), WithinRel(expected_row0[j], 1e-9));
  }
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(2480.28362300107, 1e-9));

  // Pair forces are equal and opposite, so the net force vanishes.
  for (size_t j = 0; j < 3; ++j) {
    REQUIRE_THAT(columnSum(forces, j), WithinAbs(0.0, 1e-9));
  }
}

TEST_CASE("MorsePot parameter fingerprint tracks the config", "[MorsePot]") {
  const rgpot::MorsePot defaults;
  const rgpot::MorsePot deeper{rgpot::MorseConfig{.De = 0.8}};
  const rgpot::MorsePot sameAsDefaults{rgpot::MorseConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != deeper.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::Morse);
}
