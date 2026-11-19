// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Gupta/gupta.h"

#include <cmath>
#include <limits>
#include <vector>

TEST_CASE("Gupta dimer at r0 is 2A - 2 xi", "[gupta]") {
  const double a = 0.1477;
  const double xi = 0.8900;
  const double p = 9.689;
  const double q = 4.602;
  const double r0 = 1.0;
  const unsigned char slot[2] = {0, 0};
  const double x[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6];
  rgpot_gupta(x, 2, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  REQUIRE(energy == Catch::Approx(2.0 * a - 2.0 * xi).margin(1e-12));
}

TEST_CASE("Gupta keeps a distant pair", "[gupta]") {
  const double a = 0.2061;
  const double xi = 1.790;
  const double p = 10.229;
  const double q = 4.036;
  const double r0 = 2.884;
  const unsigned char slot[3] = {0, 0, 0};
  const double x[9] = {0.0, 0.0, 0.0, r0, 0.0, 0.0, 4.0 * r0, 0.0, 0.0};
  double full = 0.0;
  double gradient[9];
  rgpot_gupta(x, 3, slot, 1, &a, &xi, &p, &q, &r0, &full, gradient);
  const unsigned char two[2] = {0, 0};
  const double near[6] = {0.0, 0.0, 0.0, r0, 0.0, 0.0};
  double contact = 0.0;
  double unused[6];
  rgpot_gupta(near, 2, two, 1, &a, &xi, &p, &q, &r0, &contact, unused);
  REQUIRE(full != contact);
}

TEST_CASE("Gupta returns on the first nonpositive pair", "[gupta]") {
  // Atoms 0 and 1 coincide, so the first pair has r2 == 0 and the walk
  // returns before the NaN coordinate of atom 2 is read.
  const double a = 0.2061;
  const double xi = 1.790;
  const double p = 10.229;
  const double q = 4.036;
  const double r0 = 2.884;
  const unsigned char slot[3] = {0, 0, 0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double x[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, 0.0, 0.0};
  double energy = -1.0;
  double gradient[9];
  for (double &component : gradient) {
    component = 7.0;
  }
  rgpot_gupta(x, 3, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  REQUIRE(std::isinf(energy));
  REQUIRE(energy > 0.0);
  for (double component : gradient) {
    REQUIRE(component == 0.0);
  }
}

TEST_CASE("Gupta with fewer than two atoms is zero", "[gupta]") {
  // The pair loop does not run, so a NaN coordinate is never read.
  const double a = 0.1477;
  const double xi = 0.8900;
  const double p = 9.689;
  const double q = 4.602;
  const double r0 = 1.0;
  const unsigned char slot[1] = {0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double one[3] = {nan, nan, nan};
  double energy = 5.0;
  double gradient[3] = {7.0, 7.0, 7.0};
  rgpot_gupta(one, 1, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  REQUIRE(energy == 0.0);
  for (double component : gradient) {
    REQUIRE(component == 0.0);
  }
  energy = 5.0;
  rgpot_gupta(one, 0, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  REQUIRE(energy == 0.0);
}

TEST_CASE("Gupta evaluates a reached NaN pair", "[gupta]") {
  // NaN is not <= 0, so the only pair is evaluated and both results are NaN.
  const double a = 0.1477;
  const double xi = 0.8900;
  const double p = 9.689;
  const double q = 4.602;
  const double r0 = 1.0;
  const unsigned char slot[2] = {0, 0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double x[6] = {0.0, 0.0, 0.0, nan, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
  rgpot_gupta(x, 2, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  REQUIRE(std::isnan(energy));
  for (double component : gradient) {
    REQUIRE(std::isnan(component));
  }
}
