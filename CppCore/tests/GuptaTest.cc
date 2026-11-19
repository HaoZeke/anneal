// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Gupta/gupta.h"

#include <algorithm>
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

TEST_CASE("Gupta stretched dimer gradient matches a central difference",
          "[gupta]") {
  const double a = 0.2061;
  const double xi = 1.790;
  const double p = 10.229;
  const double q = 4.036;
  const double r0 = 2.884;
  const unsigned char slot[2] = {0, 0};
  const double x[6] = {0.0, 0.0, 0.0, 1.2 * r0, -0.3 * r0, 0.4 * r0};
  double energy = 0.0;
  double gradient[6] = {};
  rgpot_gupta(x, 2, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);
  const double h = 1.0e-6;
  for (std::size_t coord = 0; coord < 6; ++coord) {
    double up[6];
    double down[6];
    for (std::size_t k = 0; k < 6; ++k) {
      up[k] = x[k];
      down[k] = x[k];
    }
    up[coord] += h;
    down[coord] -= h;
    double eu = 0.0;
    double ed = 0.0;
    double unused[6];
    rgpot_gupta(up, 2, slot, 1, &a, &xi, &p, &q, &r0, &eu, unused);
    rgpot_gupta(down, 2, slot, 1, &a, &xi, &p, &q, &r0, &ed, unused);
    const double numeric = (eu - ed) / (2.0 * h);
    REQUIRE(std::abs(numeric) > 1.0e-2);
    REQUIRE(gradient[coord] == Catch::Approx(numeric).margin(1.0e-7));
  }
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

TEST_CASE("Gupta gradient matches the energy derivative", "[gupta][force]") {
  // No existing Gupta test checks the gradient against the energy it came
  // from; only the energy itself (dimer closed form, distant-pair
  // sensitivity) is pinned above. A sign or scale error in the pairwise
  // derivative accumulation would pass both of those but fail here.
  const double a = 0.2061;
  const double xi = 1.790;
  const double p = 10.229;
  const double q = 4.036;
  const double r0 = 2.884;
  const unsigned char slot[3] = {0, 0, 0};
  double x[9] = {0.0, 0.0, 0.0, 1.05 * r0, 0.0, 0.0, 0.3 * r0, 1.4 * r0, -0.2 * r0};
  double gradient[9];
  double energy = 0.0;
  rgpot_gupta(x, 3, slot, 1, &a, &xi, &p, &q, &r0, &energy, gradient);

  constexpr double h = 1e-6;
  for (int coord = 0; coord < 9; ++coord) {
    const double base = x[coord];
    double unused_gradient[9];
    x[coord] = base + h;
    double ep = 0.0;
    rgpot_gupta(x, 3, slot, 1, &a, &xi, &p, &q, &r0, &ep, unused_gradient);
    x[coord] = base - h;
    double em = 0.0;
    rgpot_gupta(x, 3, slot, 1, &a, &xi, &p, &q, &r0, &em, unused_gradient);
    x[coord] = base;
    const double numeric = (ep - em) / (2.0 * h);
    const double scale = std::max(1.0, std::max(std::abs(numeric), std::abs(gradient[coord])));
    REQUIRE(std::abs(numeric - gradient[coord]) <= 1e-4 * scale);
  }
}
