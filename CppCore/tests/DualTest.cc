// MIT License
// Copyright 2023--present rgpot developers
//
// Forward-mode dual numbers (Dual.hpp): first and second derivatives of the
// operations and functions a kernel uses, against their closed forms, and
// the piecewise derivative of a branch taken on the value.

#include <algorithm>
#include <cmath>

#include <catch2/catch_all.hpp>

#include "rgpot/Dual.hpp"

using rgpot::ad::Dual;
using rgpot::ad::Dual2;
using rgpot::ad::secondDerivative;
using rgpot::ad::value;

namespace {

bool close(double got, double want, double rel = 1.0e-15) {
  return std::abs(got - want) <= rel * std::max(1.0, std::abs(want));
}

template <typename S> S lennardJonesInSquare(const S &s) {
  const S inv3 = 1.0 / (s * s * s);
  return 4.0 * inv3 * (inv3 - 1.0);
}

template <typename S> S clampedCube(const S &t) {
  if (value(t) <= 0.0) {
    return S(0.0);
  }
  if (value(t) >= 1.0) {
    return S(1.0);
  }
  return t * t * t;
}

} // namespace

TEST_CASE("a seeded dual carries the derivative of each operation", "[dual]") {
  const Dual<double> x{1.7, 1.0};
  const Dual<double> y{-0.4, 0.0};
  REQUIRE((x + y).d == 1.0);
  REQUIRE((x - y).d == 1.0);
  REQUIRE((x * y).d == -0.4);
  REQUIRE(close((y / x).d, 0.4 / (1.7 * 1.7)));
  REQUIRE(close((2.0 / x).d, -2.0 / (1.7 * 1.7)));
  REQUIRE((3.0 * x - x / 2.0 + 1.0).d == 2.5);
  REQUIRE((-x).d == -1.0);
  REQUIRE(close(sqrt(x).d, 0.5 / std::sqrt(1.7)));
  REQUIRE(close(exp(x).d, std::exp(1.7)));
  REQUIRE(close(log(x).d, 1.0 / 1.7));
}

TEST_CASE("mixed scalar operations match the dual-dual ones", "[dual]") {
  const Dual<double> x{0.9, 1.3};
  const Dual<double> c{2.5};
  REQUIRE((x * 2.5).d == (x * c).d);
  REQUIRE((2.5 * x).d == (c * x).d);
  REQUIRE((x + 2.5).v == (x + c).v);
  REQUIRE(close((2.5 / x).d, (c / x).d));
  REQUIRE(close((x / 2.5).d, (x / c).d));
}

TEST_CASE("a nested dual gives the second derivative", "[dual]") {
  const double x = 1.3;
  const auto cube = secondDerivative([](auto t) { return t * t * t; }, x);
  REQUIRE(close(cube.value, x * x * x));
  REQUIRE(close(cube.first, 3.0 * x * x));
  REQUIRE(close(cube.second, 6.0 * x));

  const auto inverse = secondDerivative([](auto t) { return 1.0 / t; }, x);
  REQUIRE(close(inverse.first, -1.0 / (x * x)));
  REQUIRE(close(inverse.second, 2.0 / (x * x * x)));

  const auto root = secondDerivative([](auto t) { return sqrt(t); }, x);
  REQUIRE(close(root.first, 0.5 / std::sqrt(x)));
  REQUIRE(close(root.second, -0.25 / (x * std::sqrt(x))));

  const auto decay = secondDerivative([](auto t) { return exp(-2.0 * t); }, x);
  REQUIRE(close(decay.first, -2.0 * std::exp(-2.0 * x)));
  REQUIRE(close(decay.second, 4.0 * std::exp(-2.0 * x)));

  const auto logarithm = secondDerivative([](auto t) { return log(t); }, x);
  REQUIRE(close(logarithm.second, -1.0 / (x * x)));
}

TEST_CASE("the nested derivative of a pair term in the squared distance",
          "[dual]") {
  for (const double s : {0.81, 1.0, 1.2599, 2.3, 6.0}) {
    const auto d =
        secondDerivative([](auto t) { return lennardJonesInSquare(t); }, s);
    const double inv3 = 1.0 / (s * s * s);
    REQUIRE(close(d.value, 4.0 * inv3 * (inv3 - 1.0), 1.0e-14));
    REQUIRE(
        close(d.first, 4.0 * (-6.0 * inv3 * inv3 + 3.0 * inv3) / s, 1.0e-14));
    REQUIRE(close(d.second, 4.0 * (42.0 * inv3 * inv3 - 12.0 * inv3) / (s * s),
                  1.0e-14));
  }
}

TEST_CASE("the inner and outer first derivatives of a nested dual agree",
          "[dual]") {
  const Dual2 y = lennardJonesInSquare(rgpot::ad::seed2(1.1));
  REQUIRE(y.v.d == y.d.v);
}

TEST_CASE("a branch on the value takes the derivative of the branch",
          "[dual]") {
  const auto inside =
      secondDerivative([](auto t) { return clampedCube(t); }, 0.5);
  REQUIRE(inside.first == 0.75);
  REQUIRE(inside.second == 3.0);
  for (const double t : {-0.25, 0.0, 1.0, 1.5}) {
    const auto outside =
        secondDerivative([](auto u) { return clampedCube(u); }, t);
    REQUIRE(outside.first == 0.0);
    REQUIRE(outside.second == 0.0);
  }
}
