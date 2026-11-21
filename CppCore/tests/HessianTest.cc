// MIT License
// Copyright 2023--present rgpot developers
//
// Dense Hessians (Potential.hpp hooks hessianImpl and hessianAdImpl) of
// LJPot and MorsePot: each against central differences of the forces, their
// symmetry, their product against hessianVectorImpl, the dual-number
// Hessian against the analytic one, and the templated pair energy against
// the force kernel.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DenseHessianOracle.hpp"
#include "rgpot/Dual.hpp"
#include "rgpot/LennardJones/LJPairEnergy.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::LJConfig;
using rgpot::LJPot;
using rgpot::MorseConfig;
using rgpot::MorsePot;

namespace {

/// Positions, species and cell of a test geometry.
struct Geometry {
  std::vector<double> x;
  std::vector<int> z;
  std::array<double, 9> box{};

  [[nodiscard]] ForceInput input() const {
    return {.nAtoms = z.size(),
            .pos = x.data(),
            .atmnrs = z.data(),
            .box = box.data()};
  }
  [[nodiscard]] std::size_t dim() const { return x.size(); }
};

std::array<double, 9> cubic(double l) {
  return {l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l};
}

double wobble(std::size_t k) {
  return std::sin(1.7 * static_cast<double>(k) + 0.3);
}

/// A 13-atom cuboctahedron with nearest neighbours at @a spacing, every
/// coordinate pushed off its site, far from its periodic images. Pair
/// distances scale with @a spacing: at 1.12 none lies in [1.24, 1.50] or
/// beyond 2.31, at 2.9 none in [3.21, 3.89] or beyond 5.98.
Geometry cluster(double spacing) {
  const double s = spacing / std::sqrt(2.0);
  Geometry g;
  g.x = {0.0, 0.0, 0.0};
  for (int a = -1; a <= 1; a += 2) {
    for (int b = -1; b <= 1; b += 2) {
      g.x.insert(g.x.end(), {a * s, b * s, 0.0});
      g.x.insert(g.x.end(), {a * s, 0.0, b * s});
      g.x.insert(g.x.end(), {0.0, a * s, b * s});
    }
  }
  for (std::size_t k = 0; k < g.x.size(); ++k) {
    g.x[k] += 15.0 + 0.04 * spacing * wobble(k);
  }
  g.z.assign(13, 1);
  g.box = cubic(30.0);
  return g;
}

/// A 4 x 4 x 4 simple cubic lattice of spacing 1.5 in a periodic cell of
/// 6, every coordinate pushed off its site by up to 0.05, so pairs cross
/// the cell faces. No pair distance lies in [1.61, 1.98] or [2.27, 2.47].
Geometry lattice() {
  Geometry g;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        g.x.insert(g.x.end(), {0.2 + 1.5 * i, 0.2 + 1.5 * j, 0.2 + 1.5 * k});
      }
    }
  }
  for (std::size_t k = 0; k < g.x.size(); ++k) {
    g.x[k] += 0.05 * wobble(k);
  }
  g.z.assign(64, 1);
  g.box = cubic(6.0);
  return g;
}

/// lattice() in a cell whose second vector is (1.5, 6, 0). The shear is one
/// lattice spacing, so the crystal is the same, but the neighbour layer
/// takes the general minimum image and pairs across the y faces pair other
/// atoms. The gaps of lattice() hold.
Geometry sheared() {
  Geometry g = lattice();
  g.box[3] = 1.5;
  return g;
}

/// @a g with positions and cell scaled by @a f.
Geometry scaled(Geometry g, double f) {
  for (double &x : g.x) {
    x *= f;
  }
  for (double &l : g.box) {
    l *= f;
  }
  return g;
}

std::vector<double> direction(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) {
    v[k] = std::cos(0.9 * static_cast<double>(k) + 0.2);
  }
  return v;
}

std::vector<double> multiply(const std::vector<double> &h,
                             const std::vector<double> &v) {
  const std::size_t dim = v.size();
  std::vector<double> out(dim, 0.0);
  for (std::size_t r = 0; r < dim; ++r) {
    for (std::size_t c = 0; c < dim; ++c) {
      out[r] += h[r * dim + c] * v[c];
    }
  }
  return out;
}

double largestMagnitude(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double largestDifference(const std::vector<double> &a,
                         const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

bool exactlySymmetric(const std::vector<double> &h, std::size_t dim) {
  for (std::size_t r = 0; r < dim; ++r) {
    for (std::size_t c = r + 1; c < dim; ++c) {
      if (h[r * dim + c] != h[c * dim + r]) {
        return false;
      }
    }
  }
  return true;
}

template <typename Pot>
std::vector<double> analyticHessian(const Pot &pot, const Geometry &g) {
  std::vector<double> h(g.dim() * g.dim(), -1.0);
  pot.hessianImpl(g.input(), h.data());
  return h;
}

template <typename Pot>
std::vector<double> dualHessian(const Pot &pot, const Geometry &g) {
  std::vector<double> h(g.dim() * g.dim(), -1.0);
  pot.hessianAdImpl(g.input(), h.data());
  return h;
}

struct LjCase {
  const char *name;
  Geometry geometry;
  LJConfig config;
};

/// Cutoffs and switch onsets sit in the gaps of the pair distances, so the
/// finite differences never step across the @f$C^2@f$ joins there.
std::vector<LjCase> ljCases() {
  return {
      {"cluster, shifted", cluster(1.12), LJConfig{.cutoff = 3.0}},
      {"cluster, switched", cluster(1.12),
       LJConfig{.cutoff = 3.0, .switch_width = 1.65}},
      {"periodic lattice, shifted", lattice(), LJConfig{.cutoff = 2.35}},
      {"periodic lattice, switched", lattice(),
       LJConfig{.cutoff = 2.36, .switch_width = 0.56}},
      {"sheared lattice, shifted", sheared(), LJConfig{.cutoff = 2.35}},
      {"sheared lattice, switched", sheared(),
       LJConfig{.cutoff = 2.36, .switch_width = 0.56}},
  };
}

struct MorseCase {
  const char *name;
  Geometry geometry;
  MorseConfig config;
};

/// Near the Morse minimum at 2.897. The lattice is scaled to spacing 2.9,
/// where no pair distance lies in [3.10, 3.83] or [4.38, 4.77].
std::vector<MorseCase> morseCases() {
  const Geometry lat = scaled(lattice(), 2.9 / 1.5);
  const Geometry shear = scaled(sheared(), 2.9 / 1.5);
  return {
      {"cluster, shifted", cluster(2.9), MorseConfig{.cutoff = 7.0}},
      {"cluster, switched", cluster(2.9),
       MorseConfig{.cutoff = 7.0, .switch_width = 3.5}},
      {"periodic lattice, shifted", lat, MorseConfig{.cutoff = 4.6}},
      {"periodic lattice, switched", lat,
       MorseConfig{.cutoff = 4.6, .switch_width = 1.1}},
      {"sheared lattice, shifted", shear, MorseConfig{.cutoff = 4.6}},
      {"sheared lattice, switched", shear,
       MorseConfig{.cutoff = 4.6, .switch_width = 1.1}},
  };
}

} // namespace

TEST_CASE("LJPot advertises the dense and dual-number derivatives",
          "[Hessian]") {
  const auto caps = LJPot().caps();
  CHECK(caps.analyticHessian);
  CHECK(caps.adHessian);
  CHECK(caps.adHvp);
}

TEST_CASE("LJPot hessianImpl matches central differences of the forces",
          "[Hessian]") {
  for (const auto &c : ljCases()) {
    INFO(c.name);
    const LJPot pot(c.config);
    const auto h = analyticHessian(pot, c.geometry);
    const auto fd =
        richardsonHessian(pot, c.geometry.x, c.geometry.z, c.geometry.box);
    // The extrapolated differences of these geometries agree with the
    // closed form to about 1e-11 of the largest entry; 1e-9 leaves room for
    // other compilers' rounding and still sits orders below any missed or
    // misweighted pair term.
    CHECK(largestDifference(h, fd) <= 1e-9 * largestMagnitude(h));
    CHECK(largestMagnitude(h) > 1.0);
  }
}

TEST_CASE("LJPot hessianImpl is symmetric and agrees with hessianVectorImpl",
          "[Hessian]") {
  for (const auto &c : ljCases()) {
    INFO(c.name);
    const LJPot pot(c.config);
    const auto h = analyticHessian(pot, c.geometry);
    CHECK(exactlySymmetric(h, c.geometry.dim()));
    const auto v = direction(c.geometry.dim());
    std::vector<double> hv(v.size());
    pot.hessianVectorImpl(c.geometry.input(), v.data(), hv.data());
    const auto product = multiply(h, v);
    CHECK(largestDifference(product, hv) <= 1e-12 * largestMagnitude(hv));
  }
}

TEST_CASE("LJPot dual-number Hessian equals the analytic one to 1e-12",
          "[Hessian]") {
  for (const auto &c : ljCases()) {
    INFO(c.name);
    const LJPot pot(c.config);
    const auto analytic = analyticHessian(pot, c.geometry);
    const auto dual = dualHessian(pot, c.geometry);
    CHECK(exactlySymmetric(dual, c.geometry.dim()));
    const double scale = largestMagnitude(analytic);
    CHECK(largestDifference(dual, analytic) <= 1e-12 * scale);

    const auto v = direction(c.geometry.dim());
    std::vector<double> hvAnalytic(v.size());
    std::vector<double> hvDual(v.size());
    pot.hessianVectorImpl(c.geometry.input(), v.data(), hvAnalytic.data());
    pot.hessianVectorAdImpl(c.geometry.input(), v.data(), hvDual.data());
    CHECK(largestDifference(hvDual, hvAnalytic) <=
          1e-12 * largestMagnitude(hvAnalytic));
  }
}

TEST_CASE("lj::pairEnergy is the pair term the force kernel sums",
          "[Hessian]") {
  // Every pair of the cluster is inside both cutoffs and none has an image
  // inside them, so the pair sum over i < j is the whole energy.
  const Geometry g = cluster(1.12);
  for (const LJConfig config : {LJConfig{.cutoff = 6.0},
                                LJConfig{.cutoff = 6.0, .switch_width = 4.0}}) {
    const LJPot pot(config);
    const double a = std::pow(config.psi / config.cutoff, 6.0);
    rgpot::simd::LjParams params{};
    params.psi2 = config.psi * config.psi;
    params.four_u0 = 4.0 * config.u0;
    params.shift = 4.0 * config.u0 * a * (a - 1.0);
    if (config.switch_width != 0.0) {
      const auto sw =
          rgpot::QuinticSwitch::endingAt(config.cutoff, config.switch_width);
      params.use_switch = 1;
      params.r_on = sw.r_on;
      params.inv_width = sw.inv_width;
    }
    const std::size_t n = g.z.size();
    double energy = 0.0;
    std::vector<double> forces(g.dim(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        double d[3];
        double r2 = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
          d[k] = g.x[3 * i + k] - g.x[3 * j + k];
          r2 += d[k] * d[k];
        }
        const auto e = rgpot::ad::secondDerivative(
            [&](auto s) { return rgpot::lj::pairEnergy(s, params); }, r2);
        energy += e.value;
        for (std::size_t k = 0; k < 3; ++k) {
          forces[3 * i + k] -= 2.0 * e.first * d[k];
          forces[3 * j + k] += 2.0 * e.first * d[k];
        }
      }
    }
    std::vector<double> kernel(g.dim(), 0.0);
    ForceOut out{.F = kernel.data(),
                 .energy = 0.0,
                 .variance = 0.0,
                 .stress = {},
                 .has_stress = 0};
    pot.forceImpl(g.input(), &out);
    CHECK(std::abs(energy - out.energy) <= 1e-13 * std::abs(out.energy));
    CHECK(largestDifference(forces, kernel) <=
          1e-12 * largestMagnitude(kernel));
  }
}

TEST_CASE("LJPot dense Hessians refuse coincident atoms", "[Hessian]") {
  const LJPot pot(LJConfig{.cutoff = 3.0});
  Geometry g;
  g.x = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
  g.z = {1, 1};
  g.box = cubic(30.0);
  std::vector<double> h(36);
  std::vector<double> v(6, 0.5);
  std::vector<double> hv(6);
  CHECK_THROWS_AS(pot.hessianImpl(g.input(), h.data()), std::domain_error);
  CHECK_THROWS_AS(pot.hessianAdImpl(g.input(), h.data()), std::domain_error);
  CHECK_THROWS_AS(pot.hessianVectorAdImpl(g.input(), v.data(), hv.data()),
                  std::domain_error);
}

TEST_CASE("A single atom has a zero dense Hessian", "[Hessian]") {
  const LJPot pot;
  Geometry g;
  g.x = {1.0, 2.0, 3.0};
  g.z = {1};
  g.box = cubic(30.0);
  CHECK(analyticHessian(pot, g) == std::vector<double>(9, 0.0));
  CHECK(dualHessian(pot, g) == std::vector<double>(9, 0.0));
}

TEST_CASE("MorsePot advertises an analytic Hessian and no dual numbers",
          "[Hessian]") {
  const auto caps = MorsePot().caps();
  CHECK(caps.analyticHessian);
  CHECK_FALSE(caps.adHessian);
  CHECK_FALSE(caps.adHvp);
}

TEST_CASE("MorsePot hessianImpl matches central differences of the forces",
          "[Hessian]") {
  for (const auto &c : morseCases()) {
    INFO(c.name);
    const MorsePot pot(c.config);
    const auto h = analyticHessian(pot, c.geometry);
    const auto fd =
        richardsonHessian(pot, c.geometry.x, c.geometry.z, c.geometry.box);
    CHECK(largestDifference(h, fd) <= 1e-9 * largestMagnitude(h));
    CHECK(largestMagnitude(h) > 0.1);
  }
}

TEST_CASE("MorsePot hessianImpl is symmetric and agrees with hessianVectorImpl",
          "[Hessian]") {
  for (const auto &c : morseCases()) {
    INFO(c.name);
    const MorsePot pot(c.config);
    const auto h = analyticHessian(pot, c.geometry);
    CHECK(exactlySymmetric(h, c.geometry.dim()));
    const auto v = direction(c.geometry.dim());
    std::vector<double> hv(v.size());
    pot.hessianVectorImpl(c.geometry.input(), v.data(), hv.data());
    CHECK(largestDifference(multiply(h, v), hv) <=
          1e-12 * largestMagnitude(hv));
  }
}

TEST_CASE("MorsePot refuses the dual-number hooks it does not advertise",
          "[Hessian]") {
  const MorsePot pot;
  const Geometry g = cluster(2.9);
  std::vector<double> h(g.dim() * g.dim());
  std::vector<double> v(g.dim(), 0.5);
  std::vector<double> hv(g.dim());
  CHECK_THROWS_AS(pot.hessianAdImpl(g.input(), h.data()), std::runtime_error);
  CHECK_THROWS_AS(pot.hessianVectorAdImpl(g.input(), v.data(), hv.data()),
                  std::runtime_error);
}
