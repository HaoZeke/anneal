// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Models/ModelPots.hpp"
#ifdef RGPOT_HAS_CBE
#include "rgpot/CBEPot/CBEPot.hpp"
#endif

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace {
void close(double got, double expected, double scale) {
  const double bar = 256.0 * std::numeric_limits<double>::epsilon() * scale;
  REQUIRE(std::abs(got - expected) <= bar);
}
} // namespace

TEST_CASE("Eckart analytic product has the barrier curvature") {
  rgpot::EckartPot pot(2.0, 1.5);
  const std::array<double, 1> x{0.0};
  const std::array<double, 1> v{0.7};
  const auto got = pot.evaluateCoordinates(x, v);
  close(got.energy, 2.0, 2.0);
  close(got.gradient[0], 0.0, 2.0);
  close(got.product[0], -4.0 * v[0] / (1.5 * 1.5), 2.0);
}

TEST_CASE("ND double well product matches a tiny dense oracle") {
  rgpot::DoubleWellPot pot(3, 2.0, 3.0, 1.5, 0.4);
  const std::array<double, 3> x{0.7, -0.2, 0.8};
  const std::array<double, 3> v{0.3, -0.5, 0.2};
  const auto got = pot.evaluateCoordinates(x, v);
  const double u = x[0] / 3.0;
  const double k = 1.5 * 1.5;
  const double h00 =
      8.0 * (3.0 * u * u - 1.0) / 9.0 + 2.0 * k * 0.4 * 0.4;
  const double h01 = -k * 0.4;
  const std::array<double, 3> oracle{
      h00 * v[0] + h01 * v[1] + h01 * v[2],
      h01 * v[0] + k * v[1], h01 * v[0] + k * v[2]};
  for (std::size_t i = 0; i < oracle.size(); ++i)
    close(got.product[i], oracle[i], 4.0);
}

TEST_CASE("LEPS analytic product is symmetric and translation invariant") {
  const std::array<double, 9> x{0.1, 0.2, -0.1, 1.8, -0.3,
                                 0.2, 3.2, 0.4, -0.2};
  const std::array<double, 9> a{0.2, -0.1, 0.3, -0.4, 0.5,
                                 0.1, 0.3, -0.2, -0.5};
  const std::array<double, 9> b{-0.3, 0.4, 0.2, 0.1, -0.5,
                                 0.3, 0.2, 0.1, -0.4};
  for (const auto system : {rgpot::LepsSystem::HH2, rgpot::LepsSystem::HCl2,
                            rgpot::LepsSystem::FH2}) {
    rgpot::LepsPot pot(system);
    const auto ha = pot.evaluateCoordinates(x, a).product;
    const auto hb = pot.evaluateCoordinates(x, b).product;
    double ahb = 0.0, bha = 0.0, scale = 1.0;
    for (std::size_t i = 0; i < 9; ++i) {
      ahb += a[i] * hb[i];
      bha += b[i] * ha[i];
      scale += std::abs(ha[i]) + std::abs(hb[i]);
    }
    close(ahb, bha, scale);
    for (std::size_t axis = 0; axis < 3; ++axis)
      close(ha[axis] + ha[3 + axis] + ha[6 + axis], 0.0, scale);
  }
}

TEST_CASE("Muller variants match analytic dense products") {
  const std::array<double, 2> x{0.23, -0.17};
  const std::array<double, 2> v{0.31, -0.27};
  rgpot::MullerBrownPot published;
  rgpot::ModifiedMullerBrownPot modified(9.0, 5.0);
  const auto base = published.evaluateCoordinates(x, v);
  const auto got = modified.evaluateCoordinates(x, v);
  const double f = 10.0 * std::numbers::pi;
  const double sx = std::sin(f * x[0]);
  const double sy = std::sin(f * x[1]);
  const double cx = std::cos(f * x[0]);
  const double cy = std::cos(f * x[1]);
  const double diagonal = -9.0 * f * f * sx * sy;
  const double mixed = 9.0 * f * f * cx * cy;
  close(got.product[0] - base.product[0],
        diagonal * v[0] + mixed * v[1], 4000.0);
  close(got.product[1] - base.product[1],
        mixed * v[0] + diagonal * v[1], 4000.0);
}

TEST_CASE("coupled proton well matches its two-by-two Hessian") {
  rgpot::CoupledProtonWellPot pot(2.0, 3.0, 1.7, 0.4);
  const std::array<double, 2> x{0.7, -0.2};
  const std::array<double, 2> v{0.3, -0.5};
  const double u = x[0] / 3.0;
  const double hxx =
      8.0 * (3.0 * u * u - 1.0) / 9.0 + 1.7 * 0.4 * 0.4;
  const double hxy = -1.7 * 0.4;
  const auto got = pot.evaluateCoordinates(x, v);
  close(got.product[0], hxx * v[0] + hxy * v[1], 4.0);
  close(got.product[1], hxy * v[0] + 1.7 * v[1], 4.0);
}

TEST_CASE("model capabilities report exact products") {
  const rgpot::MullerBrownPot pot;
  REQUIRE(pot.caps().analyticHvp);
  REQUIRE_FALSE(pot.caps().periodic);
}

TEST_CASE("atom carrier preserves force sign and analytic product") {
  rgpot::EckartPot pot(2.0, 1.5);
  const std::array<double, 3> x{0.4, 0.0, 0.0};
  const std::array<double, 3> direction{0.7, 0.0, 0.0};
  const std::array<int, 1> z{1};
  const std::array<double, 9> box{};
  std::array<double, 3> force{}, product{};
  rgpot::ForceInput in{1, x.data(), z.data(), box.data()};
  rgpot::ForceOut out{force.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(in, &out);
  pot.hessianVectorImpl(in, direction.data(), product.data());
  const auto natural =
      pot.evaluateCoordinates({x.data(), 1}, {direction.data(), 1});
  close(out.energy, natural.energy, 2.0);
  close(force[0], -natural.gradient[0], 2.0);
  close(product[0], natural.product[0], 2.0);
  close(force[1], 0.0, 2.0);
  close(product[2], 0.0, 2.0);
}

#ifdef RGPOT_HAS_CBE
TEST_CASE("external CBE frontend evaluates force and analytic product") {
  rgpot::CBEPot pot;
  const std::array<double, 18> x{
      0.0, 0.0, 0.0, 0.629, 0.629, 0.629, -0.629, -0.629, 0.629,
      -0.629, 0.629, -0.629, 0.629, -0.629, -0.629, 2.5, 0.0, 0.0};
  const std::array<int, 6> z{6, 1, 1, 1, 1, 1};
  const std::array<double, 9> box{};
  std::array<double, 18> force{}, direction{}, product{};
  direction[0] = 0.2;
  direction[15] = -0.2;
  const rgpot::ForceInput in{6, x.data(), z.data(), box.data()};
  rgpot::ForceOut out{force.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(in, &out);
  pot.hessianVectorImpl(in, direction.data(), product.data());
  REQUIRE(std::isfinite(out.energy));
  for (std::size_t i = 0; i < force.size(); ++i) {
    REQUIRE(std::isfinite(force[i]));
    REQUIRE(std::isfinite(product[i]));
  }
}
#endif
