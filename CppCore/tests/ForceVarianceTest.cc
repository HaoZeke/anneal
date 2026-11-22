// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "rgpot/MetatomicPot/ForceVariance.hpp"

// An SO(3) force-component scatter of 0.005 eV/A is a variance of 2.5e-5.
// Storing the RMS makes a later square root a fourth root (0.0707 eV/A).
//
// These cases pin the arithmetic of the two helpers only; they pass whatever
// MetatomicPot::forceImpl stores. The stored value itself is asserted in
// MetatomicPotTest.cc, on the variance that the SO(3) force call returns.

TEST_CASE("SO(3) force scatter is stored as variance, not RMS", "[variance]") {
  constexpr double rms = 0.005;
  constexpr double n_components = 12.0;
  const double sum_of_squares = rms * rms * n_components;
  const double stored =
      rgpot::forceScatterVariance(sum_of_squares, n_components);

  REQUIRE(stored == Catch::Approx(rms * rms).epsilon(1e-12));
  REQUIRE(std::sqrt(stored) == Catch::Approx(rms));
  REQUIRE(std::sqrt(stored) != Catch::Approx(std::sqrt(rms)).margin(1e-6));
}

TEST_CASE("model energy uncertainty is stored as its square", "[variance]") {
  constexpr double rms = 0.005;
  const double stored = rgpot::modelUncertaintyVariance(rms);

  REQUIRE(stored == Catch::Approx(rms * rms).epsilon(1e-12));
  REQUIRE(std::sqrt(stored) == Catch::Approx(rms));
}

TEST_CASE("an empty force sample stores zero variance", "[variance]") {
  REQUIRE(rgpot::forceScatterVariance(1.0, 0.0) == 0.0);
}
