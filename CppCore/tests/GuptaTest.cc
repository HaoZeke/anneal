// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Gupta/gupta.h"

#include <cmath>
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
