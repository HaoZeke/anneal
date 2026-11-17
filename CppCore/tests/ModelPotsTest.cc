// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Models/ModelCapi.h"
#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/units.hpp"
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

std::array<double, 2>
mullerOracle(const std::array<double, 2> &x,
             const std::array<double, 2> &direction) {
  constexpr std::array<double, 4> amplitude{-200.0, -100.0, -170.0, 15.0};
  constexpr std::array<double, 4> a{-1.0, -1.0, -6.5, 0.7};
  constexpr std::array<double, 4> b{0.0, 0.0, 11.0, 0.6};
  constexpr std::array<double, 4> c{-10.0, -10.0, -6.5, 0.7};
  constexpr std::array<double, 4> xCentre{1.0, 0.0, -0.5, -1.0};
  constexpr std::array<double, 4> yCentre{0.0, 0.5, 1.5, 1.0};
  std::array<double, 3> hessian{};
  for (std::size_t i = 0; i < amplitude.size(); ++i) {
    const double dx = x[0] - xCentre[i];
    const double dy = x[1] - yCentre[i];
    const double sx = 2.0 * a[i] * dx + b[i] * dy;
    const double sy = b[i] * dx + 2.0 * c[i] * dy;
    const double term = amplitude[i] *
                        std::exp(a[i] * dx * dx + b[i] * dx * dy +
                                 c[i] * dy * dy);
    hessian[0] += term * (sx * sx + 2.0 * a[i]);
    hessian[1] += term * (sx * sy + b[i]);
    hessian[2] += term * (sy * sy + 2.0 * c[i]);
  }
  return {hessian[0] * direction[0] + hessian[1] * direction[1],
          hessian[1] * direction[0] + hessian[2] * direction[1]};
}

struct LepsCurve {
  double de, re, exponent, sato;
};

struct LepsRadial {
  double q, j, dq, dj, d2q, d2j;
};

LepsRadial lepsRadial(const LepsCurve &curve, double distance) {
  const double t = std::exp(-curve.exponent * (distance - curve.re));
  const double dt = -curve.exponent * t;
  const double d2t = curve.exponent * curve.exponent * t;
  const double cs = curve.de / (1.0 + curve.sato);
  const double ct = curve.de / (1.0 - curve.sato);
  const double es = cs * (t * t - 2.0 * t);
  const double et = ct * (t * t + 2.0 * t);
  const double des = cs * (2.0 * t - 2.0) * dt;
  const double det = ct * (2.0 * t + 2.0) * dt;
  const double d2es =
      cs * (2.0 * dt * dt + (2.0 * t - 2.0) * d2t);
  const double d2et =
      ct * (2.0 * dt * dt + (2.0 * t + 2.0) * d2t);
  return {0.5 * (es + et), 0.5 * (es - et), 0.5 * (des + det),
          0.5 * (des - det), 0.5 * (d2es + d2et),
          0.5 * (d2es - d2et)};
}

std::array<LepsCurve, 3> lepsCurves(rgpot::LepsSystem system) {
  if (system == rgpot::LepsSystem::HH2) {
    const LepsCurve hh{0.17446, 1.40083, 1.027, 0.132};
    return {hh, hh, hh};
  }
  if (system == rgpot::LepsSystem::FH2) {
    const LepsCurve fh{0.225138, 1.732879, 1.1742, 0.167};
    const LepsCurve hh{0.17451914, 1.4019877, 1.0277, 0.106};
    return {fh, hh, fh};
  }
  const LepsCurve hcl{0.16963487116282344, 2.4085504321417757,
                      0.9886770766652772, 0.067};
  const LepsCurve cl2{0.09238779594959665, 3.7565865631435686,
                      1.0591804130726892, -0.113};
  return {hcl, cl2, hcl};
}

std::array<double, 3>
collinearLepsOracle(rgpot::LepsSystem system,
                    const std::array<double, 3> &positions,
                    const std::array<double, 3> &direction) {
  const std::array<double, 3> distance{
      (positions[1] - positions[0]) / rgpot::units::BOHR_TO_ANGSTROM,
      (positions[2] - positions[1]) / rgpot::units::BOHR_TO_ANGSTROM,
      (positions[2] - positions[0]) / rgpot::units::BOHR_TO_ANGSTROM};
  const auto curves = lepsCurves(system);
  std::array<LepsRadial, 3> radial{};
  std::array<double, 3> j{};
  for (std::size_t i = 0; i < radial.size(); ++i) {
    radial[i] = lepsRadial(curves[i], distance[i]);
    j[i] = radial[i].j;
  }
  constexpr std::array<std::array<double, 3>, 3> exchangeMatrix{
      {{{2.0, -1.0, -1.0}}, {{-1.0, 2.0, -1.0}},
       {{-1.0, -1.0, 2.0}}}};
  std::array<double, 3> u{};
  double exchange = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k)
      u[i] += exchangeMatrix[i][k] * j[k];
    exchange += 0.5 * j[i] * u[i];
  }
  const double root = std::sqrt(exchange);
  std::array<std::array<double, 3>, 3> radialHessian{};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double first = u[i] * radial[i].dj;
      const double second = u[k] * radial[k].dj;
      const double rootHessian =
          (exchangeMatrix[i][k] * radial[i].dj * radial[k].dj +
           (i == k ? u[i] * radial[i].d2j : 0.0)) /
              (2.0 * root) -
          first * second / (4.0 * root * root * root);
      radialHessian[i][k] =
          (i == k ? radial[i].d2q : 0.0) - rootHessian;
    }
  }
  constexpr std::array<std::array<double, 3>, 3> bondMap{
      {{{-1.0, 1.0, 0.0}}, {{0.0, -1.0, 1.0}},
       {{-1.0, 0.0, 1.0}}}};
  std::array<double, 3> radialDirection{}, product{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t atom = 0; atom < 3; ++atom)
      radialDirection[i] +=
          bondMap[i][atom] * direction[atom] /
          rgpot::units::BOHR_TO_ANGSTROM;
  for (std::size_t atom = 0; atom < 3; ++atom)
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t k = 0; k < 3; ++k)
        product[atom] +=
            bondMap[i][atom] * radialHessian[i][k] * radialDirection[k];
  for (double &value : product)
    value *=
        rgpot::units::HARTREE_TO_EV / rgpot::units::BOHR_TO_ANGSTROM;
  return product;
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

TEST_CASE("LEPS presets match a collinear dense analytic oracle") {
  const std::array<double, 3> positions{0.1, 1.0, 2.3};
  const std::array<double, 3> direction{0.2, -0.4, 0.3};
  const std::array<double, 9> x{positions[0], 0.0, 0.0, positions[1], 0.0,
                                 0.0, positions[2], 0.0, 0.0};
  const std::array<double, 9> v{direction[0], 0.0, 0.0, direction[1], 0.0,
                                 0.0, direction[2], 0.0, 0.0};
  for (const auto system : {rgpot::LepsSystem::HH2, rgpot::LepsSystem::HCl2,
                            rgpot::LepsSystem::FH2}) {
    rgpot::LepsPot pot(system);
    const auto got = pot.evaluateCoordinates(x, v).product;
    const auto oracle = collinearLepsOracle(system, positions, direction);
    double scale = 1.0;
    for (double value : oracle)
      scale += std::abs(value);
    for (std::size_t atom = 0; atom < 3; ++atom)
      close(got[3 * atom], oracle[atom], scale);
  }
}

TEST_CASE("Muller variants match analytic dense products") {
  const std::array<double, 2> x{0.23, -0.17};
  const std::array<double, 2> v{0.31, -0.27};
  rgpot::MullerBrownPot published;
  rgpot::ModifiedMullerBrownPot modified(9.0, 5.0);
  const auto base = published.evaluateCoordinates(x, v);
  const auto got = modified.evaluateCoordinates(x, v);
  const auto dense = mullerOracle(x, v);
  close(base.product[0], dense[0], 4000.0);
  close(base.product[1], dense[1], 4000.0);
  const double f = 10.0 * std::numbers::pi;
  const double sx = std::sin(f * x[0]);
  const double sy = std::sin(f * x[1]);
  const double cx = std::cos(f * x[0]);
  const double cy = std::cos(f * x[1]);
  const double diagonal = -9.0 * f * f * sx * sy;
  const double mixed = 9.0 * f * f * cx * cy;
  close(got.product[0], dense[0] + diagonal * v[0] + mixed * v[1], 4000.0);
  close(got.product[1], dense[1] + mixed * v[0] + diagonal * v[1], 4000.0);
}

TEST_CASE("coupled proton well matches its two-by-two Hessian") {
  rgpot::CoupledProtonWellPot pot(2.0, 3.0, 1.7);
  const std::array<double, 2> x{0.7, -0.2};
  const std::array<double, 2> v{0.3, -0.5};
  const double u = x[0] / 3.0;
  const double alpha = 1.0 / 3.0;
  const double transverse = x[1] - alpha * x[0] * x[0];
  const double slope = -2.0 * alpha * x[0];
  const double hxx =
      8.0 * (3.0 * u * u - 1.0) / 9.0 +
      1.7 * (slope * slope - 2.0 * alpha * transverse);
  const double hxy = 1.7 * slope;
  const auto got = pot.evaluateCoordinates(x, v);
  close(got.product[0], hxx * v[0] + hxy * v[1], 4.0);
  close(got.product[1], hxy * v[0] + 1.7 * v[1], 4.0);
}

TEST_CASE("model capabilities report exact products") {
  const rgpot::MullerBrownPot pot;
  REQUIRE(pot.caps().analyticHvp);
  REQUIRE_FALSE(pot.caps().periodic);
}

TEST_CASE("built-in C handle dispatches the analytic product hook") {
  rgpot_builtin_model *model =
      rgpot_builtin_double_well_new(2, 2.0, 3.0, 1.5, 0.4);
  REQUIRE(model != nullptr);
  REQUIRE(rgpot_builtin_model_dimensions(model) == 2);
  const std::array<double, 2> x{0.7, -0.2};
  const std::array<double, 2> v{0.3, -0.5};
  std::array<double, 2> force{}, product{};
  double energy = 0.0;
  REQUIRE(rgpot_builtin_model_force(model, 1, x.data(), &energy,
                                    force.data()) == 0);
  REQUIRE(rgpot_builtin_model_hvp(model, 1, x.data(), v.data(),
                                  product.data()) == 0);
  rgpot::DoubleWellPot direct(2, 2.0, 3.0, 1.5, 0.4);
  const auto expected = direct.evaluateCoordinates(x, v);
  close(energy, expected.energy, 8.0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    close(force[i], -expected.gradient[i], 8.0);
    close(product[i], expected.product[i], 8.0);
  }
  rgpot_builtin_model_free(model);
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
