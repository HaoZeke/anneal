// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/Water/WaterPots.hpp"
#include "rgpot/Water/potential_base.hpp"

using Catch::Matchers::WithinAbs;

namespace {

// One flexible water, off the TIP4P equilibrium geometry, in a cell
// larger than the 8.5 angstrom cutoff. Order is H, H, O.
rgpot::types::AtomMatrix waterPositions() {
  rgpot::types::AtomMatrix positions(3, 3);
  const double xyz[3][3] = {
      {0.80, 0.10, 0.50},
      {-0.70, 0.05, 0.55},
      {0.0, 0.0, 0.0},
  };
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[i][k];
    }
  }
  return positions;
}

const std::array<std::array<double, 3>, 3> kBox = {{
    {40.0, 0.0, 0.0},
    {0.0, 40.0, 0.0},
    {0.0, 0.0, 40.0},
}};

double maxForce(const rgpot::types::AtomMatrix &forces) {
  double worst = 0.0;
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) + forces(i, 1) * forces(i, 1) +
                      forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

void requireNetForce(const rgpot::types::AtomMatrix &forces) {
  double sum[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    sum[0] += forces(i, 0);
    sum[1] += forces(i, 1);
    sum[2] += forces(i, 2);
  }
  REQUIRE_THAT(sum[0], WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(sum[1], WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(sum[2], WithinAbs(0.0, 1e-8));
}

class PtLjProbe : public forcefields::PotentialBase {
public:
  void eval(int nAtoms, double positions[], double forces[], double &energy,
            double const periods[], bool const fixed[]) {
    computePt(nAtoms, positions, forces, energy, periods, fixed);
  }
};

} // namespace

TEST_CASE("TIP4P energy and force on one distorted water", "[water]") {
  rgpot::TIP4PPot pot(rgpot::WaterConfig{});
  const std::vector<int> types = {1, 1, 8};
  const auto [energy, forces, variance] = pot(waterPositions(), types, kBox);
  std::printf("TIP4P_ENERGY %.12f TIP4P_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(0.141344038570, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(4.479603395144, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.config().cutoff == 8.5);
  REQUIRE(pot.config().switching_width == 1.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.getType() == rgpot::PotType::TIP4P);
  requireNetForce(forces);
  const auto [again, forces2, variance2] = pot(waterPositions(), types, kBox);
  REQUIRE_THAT(again, WithinAbs(energy, 0.0));
  (void)forces2;
  (void)variance2;
}

TEST_CASE("SPC/E energy and force on one distorted water", "[water]") {
  rgpot::SPCEPot pot(rgpot::WaterConfig{});
  const std::vector<int> types = {1, 1, 8};
  const auto [energy, forces, variance] = pot(waterPositions(), types, kBox);
  std::printf("SPCE_ENERGY %.12f SPCE_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(0.532963694950, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(8.458875831491, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.getType() == rgpot::PotType::SPCE);
  requireNetForce(forces);
  rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  const auto [tipEnergy, tipForces, tipVar] = tip(waterPositions(), types, kBox);
  REQUIRE(energy != tipEnergy);
  (void)tipForces;
  (void)tipVar;
}

TEST_CASE("TIP4P-Pt energy on one water and one platinum", "[water]") {
  // The mirror plane is z = 0, the plane of the top platinum layer.
  // The water sits above that plane.
  rgpot::types::AtomMatrix positions(4, 3);
  positions(0, 0) = 0.80;
  positions(0, 1) = 0.10;
  positions(0, 2) = 2.50;
  positions(1, 0) = -0.70;
  positions(1, 1) = 0.05;
  positions(1, 2) = 2.55;
  positions(2, 0) = 0.0;
  positions(2, 1) = 0.0;
  positions(2, 2) = 2.0;
  positions(3, 0) = 0.0;
  positions(3, 1) = 0.0;
  positions(3, 2) = 0.0;
  const std::vector<int> types = {1, 1, 8, 78};
  rgpot::TIP4PPtPot pot(rgpot::WaterConfig{});
  const auto [energy, forces, variance] = pot(positions, types, kBox);
  std::printf("TIP4P_PT_ENERGY %.12f TIP4P_PT_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(1.642427227258, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(15.396689248363, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.getType() == rgpot::PotType::TIP4PPt);
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    REQUIRE(std::isfinite(forces(i, 0)));
    REQUIRE(std::isfinite(forces(i, 1)));
    REQUIRE(std::isfinite(forces(i, 2)));
  }
}

TEST_CASE("computePt keeps an LJ pair when one atom is fixed", "[water][pt]") {
  auto evalPair = [](bool fixed0, bool fixed1) {
    PtLjProbe pot;
    double positions[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    double periods[3] = {40.0, 40.0, 40.0};
    bool fixed[2] = {fixed0, fixed1};
    std::array<double, 6> forces{};
    double energy = 0.0;
    pot.eval(2, positions, forces.data(), energy, periods, fixed);
    return std::pair<double, std::array<double, 6>>{energy, forces};
  };
  const auto free = evalPair(false, false);
  REQUIRE(free.first != 0.0);
  REQUIRE(free.second[0] != 0.0);
  REQUIRE(free.second[0] == -free.second[3]);
  REQUIRE(free.second[1] == 0.0);
  REQUIRE(free.second[2] == 0.0);

  const auto fix0 = evalPair(true, false);
  const auto fix1 = evalPair(false, true);
  REQUIRE(fix0.first == free.first);
  REQUIRE(fix1.first == free.first);
  REQUIRE(fix0.second == free.second);
  REQUIRE(fix1.second == free.second);

  const auto both = evalPair(true, true);
  REQUIRE(both.first == 0.0);
  for (double component : both.second) {
    REQUIRE(component == 0.0);
  }
}
