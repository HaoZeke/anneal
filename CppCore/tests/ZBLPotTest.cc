// MIT License
// Copyright 2023--present rgpot developers
//
// ZBL checks against LAMMPS pair_style zbl, plus the switching, cutoff and
// species-table behaviour of the rgpot port.

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::test_fixtures::cubicCell;
using rgpot::types::AtomMatrix;

namespace {

/// Si-Au dimer separated by (1.2, 1.3, 1.4) Angstrom, centred in a 20 A
/// cube: eOn's ZBL fixture (client/potentials/ZBL/wip_explorations
/// generates it, client/unit_tests/ZBLPotTest.cpp consumes it).
AtomMatrix siAuDimer() {
  return AtomMatrix{{9.40, 9.35, 9.30}, {10.60, 10.65, 10.70}};
}

} // namespace

TEST_CASE("ZBLPot against LAMMPS on a Si-Au dimer", "[ZBLPot]") {
  // LAMMPS pair_style zbl 2.0 2.5 in metal units; the same numbers eOn
  // pins in client/unit_tests/ZBLPotTest.cpp. Tolerances follow the
  // printed precision of the LAMMPS log and force dump.
  rgpot::ZBLPot pot;
  const auto positions = siAuDimer();
  const std::vector<int> atmtypes{14, 79};

  auto [energy, forces, variance] = pot(positions, atmtypes, cubicCell(20.0));
  (void)variance;

  REQUIRE_THAT(energy, WithinAbs(0.38537731, 1e-8));

  const std::array<std::array<double, 3>, 2> expected{
      {{-2.37926, -2.57753, -2.7758}, {2.37926, 2.57753, 2.7758}}};
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(expected[i][j], 1e-5));
    }
  }
}

TEST_CASE("ZBLPot switching zeroes energy and force at the cutoff",
          "[ZBLPot]") {
  rgpot::ZBLPot pot;
  const std::vector<int> atmtypes{14, 79};
  const auto cell = cubicCell(20.0);
  const double cut = pot.config().cut_global;

  const AtomMatrix atCut{{5.0, 5.0, 5.0}, {5.0 + cut, 5.0, 5.0}};
  auto [e_cut, f_cut, v_cut] = pot(atCut, atmtypes, cell);
  (void)v_cut;
  REQUIRE_THAT(e_cut, WithinAbs(0.0, 1e-12));
  for (size_t i = 0; i < f_cut.rows(); ++i) {
    for (size_t j = 0; j < f_cut.cols(); ++j) {
      REQUIRE_THAT(f_cut(i, j), WithinAbs(0.0, 1e-12));
    }
  }

  const AtomMatrix beyond{{5.0, 5.0, 5.0}, {5.0 + cut + 0.5, 5.0, 5.0}};
  auto [e_far, f_far, v_far] = pot(beyond, atmtypes, cell);
  (void)v_far;
  REQUIRE_THAT(e_far, WithinAbs(0.0, 1e-15));
  for (size_t i = 0; i < f_far.rows(); ++i) {
    for (size_t j = 0; j < f_far.cols(); ++j) {
      REQUIRE_THAT(f_far(i, j), WithinAbs(0.0, 1e-15));
    }
  }
}

TEST_CASE("ZBLPot rebuilds its tables when the species change", "[ZBLPot]") {
  // The coefficient tables are keyed on the set of atomic numbers, so one
  // instance reused across systems never carries stale screening lengths.
  rgpot::ZBLPot pot;
  const auto positions = siAuDimer();
  const auto cell = cubicCell(20.0);

  auto [e_siau, f_siau, v_siau] =
      pot(positions, std::vector<int>{14, 79}, cell);
  auto [e_auau, f_auau, v_auau] =
      pot(positions, std::vector<int>{79, 79}, cell);
  auto [e_sisi, f_sisi, v_sisi] =
      pot(positions, std::vector<int>{14, 14}, cell);
  auto [e_again, f_again, v_again] =
      pot(positions, std::vector<int>{14, 79}, cell);
  (void)v_siau;
  (void)v_auau;
  (void)v_sisi;
  (void)v_again;

  // References from an independent double-precision evaluation of this
  // kernel, cross-checked against LAMMPS for the Si-Au pair above.
  REQUIRE_THAT(e_siau, WithinRel(0.38537730883, 1e-9));
  REQUIRE_THAT(e_auau, WithinRel(0.961609764914, 1e-9));
  REQUIRE_THAT(e_sisi, WithinRel(0.167604056792, 1e-9));
  REQUIRE_THAT(e_again, WithinRel(e_siau, 1e-15));
}

TEST_CASE("ZBLPot ignores the cell", "[ZBLPot]") {
  // Free boundaries: a cell narrower than the cutoff must not fold the pair
  // vector, so the energy is unchanged.
  rgpot::ZBLPot pot;
  const auto positions = siAuDimer();
  const std::vector<int> atmtypes{14, 79};

  auto [e_wide, f_wide, v_wide] = pot(positions, atmtypes, cubicCell(20.0));
  auto [e_tight, f_tight, v_tight] = pot(positions, atmtypes, cubicCell(3.0));
  (void)v_wide;
  (void)v_tight;

  REQUIRE_THAT(e_tight, WithinRel(e_wide, 1e-15));
  REQUIRE_FALSE(pot.caps().periodic);
}

TEST_CASE("ZBLPot rejects an invalid cutoff pair", "[ZBLPot]") {
  // Named configs: a brace list with a comma cannot travel through a Catch
  // macro argument.
  const rgpot::ZBLConfig inverted{.cut_inner = 2.5, .cut_global = 2.0};
  const rgpot::ZBLConfig coincident{.cut_inner = 2.0, .cut_global = 2.0};
  const rgpot::ZBLConfig zeroInner{.cut_inner = 0.0, .cut_global = 2.5};
  const rgpot::ZBLConfig wide{.cut_inner = 2.0, .cut_global = 4.5};

  REQUIRE_THROWS_AS(rgpot::ZBLPot{inverted}, std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::ZBLPot{coincident}, std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::ZBLPot{zeroInner}, std::invalid_argument);
  REQUIRE_NOTHROW(rgpot::ZBLPot{wide});
}

TEST_CASE("ZBLPot parameter fingerprint tracks the config", "[ZBLPot]") {
  const rgpot::ZBLPot defaults;
  const rgpot::ZBLPot wider{rgpot::ZBLConfig{.cut_global = 4.5}};
  const rgpot::ZBLPot sameAsDefaults{rgpot::ZBLConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != wider.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::ZBL);
  REQUIRE(defaults.config().cut_inner == 2.0);
  REQUIRE(defaults.config().cut_global == 2.5);
}
