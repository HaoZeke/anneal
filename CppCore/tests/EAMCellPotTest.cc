// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"

TEST_CASE("Cell-list EAM returns a finite aluminium pair", "[eam]") {
  auto positions = rgpot::types::AtomMatrix::Zero(2, 3);
  positions(1, 0) = 2.5;
  const std::vector<int> types = {13, 13};
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EAMPot pot;
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE(std::isfinite(energy));
  REQUIRE(std::isfinite(forces(0, 0)));
  REQUIRE(std::abs(forces(0, 0) + forces(1, 0)) < 1e-8);
  REQUIRE(pot.paramsKey() != 0);
  (void)variance;
}

TEST_CASE("Cell-list EAM rejects an element it has no parameters for",
          "[eam]") {
  auto positions = rgpot::types::AtomMatrix::Zero(2, 3);
  positions(1, 0) = 2.5;
  const std::vector<int> types = {13, 1};
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EAMPot pot;
  REQUIRE_THROWS_AS(pot(positions, types, box), std::invalid_argument);
}
