// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace rgpot {
namespace {

using Kernel = std::function<ModelEvaluation(std::span<const double>,
                                             std::span<const double>)>;

void requireSize(std::span<const double> x, std::span<const double> v,
                 std::size_t dimensions) {
  if (x.size() != dimensions || v.size() != dimensions)
    throw std::invalid_argument("analytic model coordinate dimension");
}

ModelEvaluation muller(std::span<const double> x, std::span<const double> v,
                       double ripple, double wave) {
  requireSize(x, v, 2);
  constexpr std::array<double, 4> amp{-200.0, -100.0, -170.0, 15.0};
  constexpr std::array<double, 4> aa{-1.0, -1.0, -6.5, 0.7};
  constexpr std::array<double, 4> bb{0.0, 0.0, 11.0, 0.6};
  constexpr std::array<double, 4> cc{-10.0, -10.0, -6.5, 0.7};
  constexpr std::array<double, 4> x0{1.0, 0.0, -0.5, -1.0};
  constexpr std::array<double, 4> y0{0.0, 0.5, 1.5, 1.0};
  ModelEvaluation out{0.0, std::vector<double>(2), std::vector<double>(2)};
  for (std::size_t i = 0; i < amp.size(); ++i) {
    const double dx = x[0] - x0[i];
    const double dy = x[1] - y0[i];
    const double sx = 2.0 * aa[i] * dx + bb[i] * dy;
    const double sy = bb[i] * dx + 2.0 * cc[i] * dy;
    const double term =
        amp[i] * std::exp(aa[i] * dx * dx + bb[i] * dx * dy + cc[i] * dy * dy);
    const double ds = sx * v[0] + sy * v[1];
    out.energy += term;
    out.gradient[0] += term * sx;
    out.gradient[1] += term * sy;
    out.product[0] +=
        term * (sx * ds + 2.0 * aa[i] * v[0] + bb[i] * v[1]);
    out.product[1] +=
        term * (sy * ds + bb[i] * v[0] + 2.0 * cc[i] * v[1]);
  }
  const double frequency = 2.0 * std::numbers::pi * wave;
  const double sx = std::sin(frequency * x[0]);
  const double cx = std::cos(frequency * x[0]);
  const double sy = std::sin(frequency * x[1]);
  const double cy = std::cos(frequency * x[1]);
  const double diagonal = -ripple * frequency * frequency * sx * sy;
  const double mixed = ripple * frequency * frequency * cx * cy;
  out.energy += ripple * sx * sy;
  out.gradient[0] += ripple * frequency * cx * sy;
  out.gradient[1] += ripple * frequency * sx * cy;
  out.product[0] += diagonal * v[0] + mixed * v[1];
  out.product[1] += mixed * v[0] + diagonal * v[1];
  return out;
}

struct Curve {
  double de;
  double re;
  double exponent;
  double sato;
};

struct Radial {
  double q, j, dq, dj, d2q, d2j;
};

Radial radial(const Curve &curve, double r) {
  const double t = std::exp(-curve.exponent * (r - curve.re));
  const double dt = -curve.exponent * t;
  const double d2t = curve.exponent * curve.exponent * t;
  const double cs = curve.de / (1.0 + curve.sato);
  const double ct = curve.de / (1.0 - curve.sato);
  const double es = cs * (t * t - 2.0 * t);
  const double et = ct * (t * t + 2.0 * t);
  const double des = cs * (2.0 * t - 2.0) * dt;
  const double det = ct * (2.0 * t + 2.0) * dt;
  const double d2es = cs * (2.0 * dt * dt + (2.0 * t - 2.0) * d2t);
  const double d2et = ct * (2.0 * dt * dt + (2.0 * t + 2.0) * d2t);
  return {0.5 * (es + et), 0.5 * (es - et), 0.5 * (des + det),
          0.5 * (des - det), 0.5 * (d2es + d2et),
          0.5 * (d2es - d2et)};
}

ModelEvaluation leps(std::span<const double> x, std::span<const double> v,
                     const std::array<Curve, 3> &curves) {
  requireSize(x, v, 9);
  constexpr std::array<std::array<std::size_t, 2>, 3> pairs{
      {{0, 1}, {1, 2}, {0, 2}}};
  std::array<std::array<double, 3>, 3> unit{}, dunit{};
  std::array<double, 3> dr{};
  std::array<Radial, 3> terms{};
  for (std::size_t p = 0; p < 3; ++p) {
    const auto [a, b] = pairs[p];
    std::array<double, 3> delta{}, ddelta{};
    double r2 = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
      delta[k] = x[3 * b + k] - x[3 * a + k];
      ddelta[k] = v[3 * b + k] - v[3 * a + k];
      r2 += delta[k] * delta[k];
    }
    const double r = std::sqrt(r2);
    if (!(r > 0.0))
      throw std::domain_error("LEPS zero bond length");
    for (std::size_t k = 0; k < 3; ++k) {
      unit[p][k] = delta[k] / r;
      dr[p] += unit[p][k] * ddelta[k];
    }
    for (std::size_t k = 0; k < 3; ++k)
      dunit[p][k] = (ddelta[k] - unit[p][k] * dr[p]) / r;
    terms[p] = radial(curves[p], r);
  }
  const std::array<double, 3> j{terms[0].j, terms[1].j, terms[2].j};
  const double exchange = j[0] * j[0] + j[1] * j[1] + j[2] * j[2] -
                          j[0] * j[1] - j[1] * j[2] - j[2] * j[0];
  if (!(exchange > 0.0))
    throw std::domain_error("LEPS exchange root is not differentiable");
  const double root = std::sqrt(exchange);
  const std::array<double, 3> u{2.0 * j[0] - j[1] - j[2],
                                2.0 * j[1] - j[0] - j[2],
                                2.0 * j[2] - j[0] - j[1]};
  std::array<double, 3> djdir{}, slope{}, dslope{};
  for (std::size_t p = 0; p < 3; ++p)
    djdir[p] = terms[p].dj * dr[p];
  const double droot =
      (u[0] * djdir[0] + u[1] * djdir[1] + u[2] * djdir[2]) / (2.0 * root);
  for (std::size_t p = 0; p < 3; ++p) {
    const double du =
        2.0 * djdir[p] - djdir[(p + 1) % 3] - djdir[(p + 2) % 3];
    const double rs = u[p] / (2.0 * root);
    const double drs = du / (2.0 * root) - rs * droot / root;
    slope[p] = terms[p].dq - rs * terms[p].dj;
    dslope[p] = terms[p].d2q * dr[p] - drs * terms[p].dj -
                rs * terms[p].d2j * dr[p];
  }
  ModelEvaluation out{terms[0].q + terms[1].q + terms[2].q - root,
                      std::vector<double>(9), std::vector<double>(9)};
  for (std::size_t p = 0; p < 3; ++p) {
    const auto [a, b] = pairs[p];
    for (std::size_t k = 0; k < 3; ++k) {
      const double g = slope[p] * unit[p][k];
      const double h = dslope[p] * unit[p][k] + slope[p] * dunit[p][k];
      out.gradient[3 * a + k] -= g;
      out.gradient[3 * b + k] += g;
      out.product[3 * a + k] -= h;
      out.product[3 * b + k] += h;
    }
  }
  return out;
}

std::array<Curve, 3> curves(LepsSystem system) {
  if (system == LepsSystem::HH2) {
    const Curve hh{0.17446, 1.40083, 1.027, 0.132};
    return {hh, hh, hh};
  }
  if (system == LepsSystem::FH2) {
    const Curve fh{0.225138, 1.732879, 1.1742, 0.167};
    const Curve hh{0.17451914, 1.4019877, 1.0277, 0.106};
    return {fh, hh, fh};
  }
  // Hartree and Bohr. Spectroscopic Morse curves and Sato parameters from
  // Connor, Mayne and Whitehead, J. Chem. Phys. 72, 6209 (1980).
  const Curve hcl{0.16963487116282344, 2.4085504321417757,
                  0.9886770766652772, 0.067};
  const Curve cl2{0.09238779594959665, 3.7565865631435686,
                  1.0591804130726892, -0.113};
  return {hcl, cl2, hcl};
}

} // namespace

struct AnalyticModelPot::Impl {
  std::size_t dimensions;
  Kernel kernel;
};

AnalyticModelPot::AnalyticModelPot(PotType type, std::unique_ptr<Impl> impl)
    : Potential<AnalyticModelPot>(type), impl_(std::move(impl)) {}
AnalyticModelPot::~AnalyticModelPot() = default;
AnalyticModelPot::AnalyticModelPot(AnalyticModelPot &&) noexcept = default;
AnalyticModelPot &
AnalyticModelPot::operator=(AnalyticModelPot &&) noexcept = default;

std::size_t AnalyticModelPot::dimensions() const noexcept {
  return impl_->dimensions;
}

ModelEvaluation AnalyticModelPot::evaluateCoordinates(
    std::span<const double> x, std::span<const double> v) const {
  return impl_->kernel(x, v);
}

void AnalyticModelPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const std::size_t carrier = 3 * in.nAtoms;
  if (carrier < dimensions())
    throw std::invalid_argument("atom carrier is shorter than model coordinates");
  std::vector<double> zero(dimensions());
  const auto got = evaluateCoordinates({in.pos, dimensions()}, zero);
  out->energy = got.energy;
  std::fill_n(out->F, carrier, 0.0);
  for (std::size_t i = 0; i < dimensions(); ++i)
    out->F[i] = -got.gradient[i];
}

void AnalyticModelPot::hessianVectorImpl(const ForceInput &in, const double *v,
                                         double *hv) const {
  const std::size_t carrier = 3 * in.nAtoms;
  if (carrier < dimensions())
    throw std::invalid_argument("atom carrier is shorter than model coordinates");
  const auto got =
      evaluateCoordinates({in.pos, dimensions()}, {v, dimensions()});
  std::fill_n(hv, carrier, 0.0);
  std::copy(got.product.begin(), got.product.end(), hv);
}

PotCaps AnalyticModelPot::caps() const noexcept {
  return {.periodic = false, .analyticHvp = true};
}

EckartPot::EckartPot(double v0, double width)
    : AnalyticModelPot(
          PotType::Eckart,
          std::make_unique<Impl>(Impl{
              1, [v0, width](std::span<const double> x,
                             std::span<const double> v) {
                requireSize(x, v, 1);
                if (!(std::abs(width) > 0.0))
                  throw std::invalid_argument("Eckart width is zero");
                const double z = x[0] / width;
                const double sech2 = 1.0 / std::pow(std::cosh(z), 2);
                const double tanh = std::tanh(z);
                return ModelEvaluation{
                    v0 * sech2,
                    {-2.0 * v0 * sech2 * tanh / width},
                    {2.0 * v0 * sech2 * (3.0 * tanh * tanh - 1.0) /
                     (width * width) * v[0]}};
              }})) {}

MullerBrownPot::MullerBrownPot()
    : AnalyticModelPot(
          PotType::MullerBrown,
          std::make_unique<Impl>(
              Impl{2, [](auto x, auto v) { return muller(x, v, 0.0, 0.0); }})) {}

ModifiedMullerBrownPot::ModifiedMullerBrownPot(double amplitude,
                                               double wave)
    : AnalyticModelPot(
          PotType::ModifiedMullerBrown,
          std::make_unique<Impl>(Impl{
              2, [amplitude, wave](auto x, auto v) {
                return muller(x, v, amplitude, wave);
              }})) {}

LepsPot::LepsPot(LepsSystem system)
    : AnalyticModelPot(
          system == LepsSystem::HH2
              ? PotType::LepsHH2
              : (system == LepsSystem::HCl2 ? PotType::LepsHCl2
                                             : PotType::LepsFH2),
          std::make_unique<Impl>(Impl{
              9, [parameters = curves(system)](auto x, auto v) {
                std::array<double, 9> xBohr{}, vBohr{};
                for (std::size_t i = 0; i < xBohr.size(); ++i) {
                  xBohr[i] = x[i] / units::BOHR_TO_ANGSTROM;
                  vBohr[i] = v[i] / units::BOHR_TO_ANGSTROM;
                }
                auto result = leps(xBohr, vBohr, parameters);
                result.energy *= units::HARTREE_TO_EV;
                const double derivativeScale =
                    units::HARTREE_TO_EV / units::BOHR_TO_ANGSTROM;
                for (std::size_t i = 0; i < xBohr.size(); ++i) {
                  result.gradient[i] *= derivativeScale;
                  result.product[i] *= derivativeScale;
                }
                return result;
              }})) {}

DoubleWellPot::DoubleWellPot(std::size_t dimensions, double v0, double x0,
                             double omega, double coupling)
    : AnalyticModelPot(
          PotType::DoubleWell,
          std::make_unique<Impl>(Impl{
              dimensions,
              [dimensions, v0, x0, omega, coupling](auto x, auto v) {
                requireSize(x, v, dimensions);
                if (dimensions == 0 || !(std::abs(x0) > 0.0))
                  throw std::invalid_argument("double-well parameters");
                const double u = x[0] / x0;
                const double k = omega * omega;
                ModelEvaluation out{v0 * std::pow(u * u - 1.0, 2),
                                    std::vector<double>(dimensions),
                                    std::vector<double>(dimensions)};
                out.gradient[0] = 4.0 * v0 * u * (u * u - 1.0) / x0;
                out.product[0] =
                    4.0 * v0 * (3.0 * u * u - 1.0) / (x0 * x0) * v[0];
                for (std::size_t i = 1; i < dimensions; ++i) {
                  const double q = x[i] - coupling * x[0];
                  const double dq = v[i] - coupling * v[0];
                  out.energy += 0.5 * k * q * q;
                  out.gradient[0] -= k * coupling * q;
                  out.gradient[i] = k * q;
                  out.product[0] -= k * coupling * dq;
                  out.product[i] = k * dq;
                }
                return out;
              }})) {}

CoupledProtonWellPot::CoupledProtonWellPot(double v0, double x0,
                                           double spring)
    : AnalyticModelPot(
          PotType::CoupledProtonWell,
          std::make_unique<Impl>(Impl{
              2, [v0, x0, spring](auto x, auto v) {
                requireSize(x, v, 2);
                if (!(std::abs(x0) > 0.0))
                  throw std::invalid_argument("coupled-well width is zero");
                const double u = x[0] / x0;
                const double alpha = 1.0 / x0;
                const double transverse = x[1] - alpha * x[0] * x[0];
                const double slope = -2.0 * alpha * x[0];
                const double quartic =
                    4.0 * v0 * u * (u * u - 1.0) / x0;
                const double curvature =
                    4.0 * v0 * (3.0 * u * u - 1.0) / (x0 * x0);
                const double hxx =
                    curvature +
                    spring * (slope * slope - 2.0 * alpha * transverse);
                const double hxy = spring * slope;
                return ModelEvaluation{
                    v0 * std::pow(u * u - 1.0, 2) +
                        0.5 * spring * transverse * transverse,
                    {quartic + spring * transverse * slope,
                     spring * transverse},
                    {hxx * v[0] + hxy * v[1],
                     hxy * v[0] + spring * v[1]}};
              }})) {}

} // namespace rgpot
