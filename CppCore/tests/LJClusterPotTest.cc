// MIT License
// Copyright 2023--present rgpot developers
//
// Free-boundary Lennard-Jones cluster checks: the eOn energy pin, agreement
// with the periodic LJPot on an isolated cluster, and cell independence.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
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

} // namespace

TEST_CASE("LJClusterPot energy on the neb_morse cluster", "[LJClusterPot]") {
  rgpot::LJClusterPot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);

  auto [energy, forces, variance] = pot(positions, atmtypes, nebMorseCell());
  (void)variance;

  // eOn reference (client/unit_tests/PotTest.cpp, "LJCluster energy matches
  // SVN"), carried over with its epsilon.
  REQUIRE_THAT(energy, WithinRel(-39.965379, 1e-4));
  // Tighter pin from an independent double-precision evaluation of this
  // kernel; eOn's LJ case pins the same maximum force to 1e-2.
  REQUIRE_THAT(energy, WithinRel(-39.9653513626179, 1e-9));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.00470419517301762, 1e-9));
}

TEST_CASE("LJClusterPot matches LJPot on an isolated cluster",
          "[LJClusterPot]") {
  // Same 12-6 kernel and cutoff shift; the cell is wide enough that the
  // minimum image convention leaves LJPot with the identical pair set.
  rgpot::LJClusterPot cluster;
  rgpot::LJPot periodic;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);
  const auto cell = nebMorseCell();

  auto [e_cluster, f_cluster, v_cluster] = cluster(positions, atmtypes, cell);
  auto [e_periodic, f_periodic, v_periodic] =
      periodic(positions, atmtypes, cell);
  (void)v_cluster;
  (void)v_periodic;

  REQUIRE_THAT(e_cluster, WithinRel(e_periodic, 1e-12));
  for (size_t i = 0; i < f_cluster.rows(); ++i) {
    for (size_t j = 0; j < f_cluster.cols(); ++j) {
      REQUIRE_THAT(f_cluster(i, j), WithinAbs(f_periodic(i, j), 1e-12));
    }
  }
}

TEST_CASE("LJClusterPot ignores the cell", "[LJClusterPot]") {
  // Free boundaries: a cell narrower than the cutoff must not fold any pair
  // vector, so the energy is unchanged.
  rgpot::LJClusterPot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);
  const std::array<std::array<double, 3>, 3> tight{
      {{5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0}}};

  auto [e_wide, f_wide, v_wide] = pot(positions, atmtypes, nebMorseCell());
  auto [e_tight, f_tight, v_tight] = pot(positions, atmtypes, tight);
  (void)v_wide;
  (void)v_tight;

  REQUIRE_THAT(e_tight, WithinRel(e_wide, 1e-12));
  REQUIRE_FALSE(pot.caps().periodic);
}

TEST_CASE("LJClusterPot parameter fingerprint tracks the config",
          "[LJClusterPot]") {
  const rgpot::LJClusterPot defaults;
  const rgpot::LJClusterPot wider{rgpot::LJClusterConfig{.cutoff = 20.0}};
  const rgpot::LJClusterPot sameAsDefaults{rgpot::LJClusterConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != wider.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::LJCluster);
  // Identical numbers hash identically; the result cache keeps the two
  // kernels apart through PotType, not through paramsKey.
  REQUIRE(defaults.paramsKey() == rgpot::LJPot{}.paramsKey());
}

TEST_CASE("LJPot stress matches a coordinate difference", "[LJPot]") {
  rgpot::LJPot pot;
  REQUIRE(pot.caps().stress);

  AtomMatrix pos{{0.0, 0.0, 0.0}, {1.3, 0.4, -0.25}};
  const std::vector<int> types{1, 1};
  // Row-major cell, large enough that the pair is the minimum image.
  const double box[9] = {12.0, 0.0, 0.0, 0.4, 11.0, 0.0, -0.2, 0.3, 10.0};
  double forces[6] = {};
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{2, pos.data(), types.data(), box}, &out);
  REQUIRE(out.has_stress == 1);

  const double volume = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                                 box[1] * (box[3] * box[8] - box[5] * box[6]) +
                                 box[2] * (box[3] * box[7] - box[4] * box[6]));
  constexpr double step = 1e-6;
  const int rows[6] = {0, 1, 2, 1, 0, 0};
  const int cols[6] = {0, 1, 2, 2, 2, 1};
  for (int comp = 0; comp < 6; ++comp) {
    const int row = rows[comp];
    const int col = cols[comp];
    auto energyAt = [&](double eps) {
      AtomMatrix shifted = pos;
      double strained[9];
      std::copy(box, box + 9, strained);
      for (int atom = 0; atom < 2; ++atom) {
        shifted(atom, col) += pos(atom, row) * eps;
      }
      for (int k = 0; k < 3; ++k) {
        strained[k * 3 + col] += box[k * 3 + row] * eps;
      }
      double f2[6] = {};
      rgpot::ForceOut sample{f2, 0.0, 0.0, {}, 0};
      pot.forceImpl(
          rgpot::ForceInput{2, shifted.data(), types.data(), strained},
          &sample);
      return sample.energy;
    };
    const double derivative = (energyAt(step) - energyAt(-step)) / (2.0 * step);
    const double expected = derivative / volume;
    const double got = out.stress[row * 3 + col];
    REQUIRE_THAT(got, WithinAbs(expected, 1e-6));
  }
}

namespace {
template <class Potential>
void checkPairStressByStrain(const Potential &pot, const AtomMatrix &positions,
                             const std::vector<int> &types) {
  REQUIRE(pot.caps().stress);
  const std::array<double, 9> box{20.0, 0.0, 0.0, 0.0, 21.0,
                                  0.0,  0.0, 0.0, 22.0};
  constexpr double volume = 20.0 * 21.0 * 22.0;
  constexpr double step = 1e-6;
  AtomMatrix forces(positions.rows(), 3);
  rgpot::ForceOut result{forces.data(), 0.0, 0.0, {}, 0};
  const rgpot::ForceInput input{static_cast<size_t>(positions.rows()),
                                positions.data(), types.data(), box.data()};
  pot.forceImpl(input, &result);
  REQUIRE(result.has_stress == 1);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const auto energyAt = [&](double strain) {
        AtomMatrix moved = positions;
        auto cell = box;
        for (int atom = 0; atom < positions.rows(); ++atom) {
          moved(atom, col) += strain * positions(atom, row);
        }
        for (int axis = 0; axis < 3; ++axis) {
          cell[3 * axis + col] += strain * box[3 * axis + row];
        }
        AtomMatrix sample_forces(positions.rows(), 3);
        rgpot::ForceOut sample{sample_forces.data(), 0.0, 0.0, {}, 0};
        pot.forceImpl({static_cast<size_t>(positions.rows()), moved.data(),
                       types.data(), cell.data()},
                      &sample);
        return sample.energy;
      };
      const double expected =
          (energyAt(step) - energyAt(-step)) / (2.0 * step * volume);
      CAPTURE(row, col, expected);
      REQUIRE_THAT(result.stress[3 * row + col], WithinAbs(expected, 1e-8));
    }
  }
  rgpot::ForceOut repeated{forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(input, &repeated);
  REQUIRE(repeated.energy == result.energy);
  REQUIRE(repeated.has_stress == 1);
  for (size_t i = 0; i < 9; ++i) {
    REQUIRE(repeated.stress[i] == result.stress[i]);
  }
}
} // namespace

TEST_CASE(
    "Pair stresses differentiate the actual truncated and switched energy",
    "[stress][LJPot][MorsePot][ZBLPot]") {
  const std::vector<int> types{6, 8};
  SECTION("periodic Lennard-Jones pair crosses the cell boundary") {
    rgpot::LJPot pot;
    AtomMatrix positions{{19.1, 0.1, -0.2}, {0.5, 0.4, 0.3}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Lennard-Jones switch contributes to stress") {
    rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 3.0, .switch_width = 0.8});
    AtomMatrix positions{{0.0, 0.0, 0.0}, {2.6, 0.2, -0.1}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Morse shifted energy") {
    rgpot::MorsePot pot;
    AtomMatrix positions{{0.0, 0.0, 0.0}, {3.1, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Morse switch contributes to stress") {
    rgpot::MorsePot pot(rgpot::MorseConfig{.cutoff = 5.0, .switch_width = 1.0});
    AtomMatrix positions{{0.0, 0.0, 0.0}, {4.3, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("ZBL screened pair") {
    rgpot::ZBLPot pot;
    AtomMatrix positions{{0.0, 0.0, 0.0}, {1.3, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
}

TEST_CASE("Pair stress is cleared when the configuration has no pair",
          "[stress][LJPot][MorsePot][ZBLPot]") {
  const int type = 6;
  const double position[3] = {0.0, 0.0, 0.0};
  const double box[9] = {20.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 22.0};
  double forces[3] = {};
  const auto check = [&](const auto &pot) {
    rgpot::ForceOut out{
        forces, 1.0, 0.0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 0};
    pot.forceImpl({1, position, &type, box}, &out);
    REQUIRE(out.has_stress == 1);
    REQUIRE(out.energy == 0.0);
    for (double component : out.stress) {
      REQUIRE(component == 0.0);
    }
  };
  check(rgpot::LJPot{});
  check(rgpot::MorsePot{});
  check(rgpot::ZBLPot{});
}
