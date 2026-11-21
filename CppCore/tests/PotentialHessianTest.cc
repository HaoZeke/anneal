// MIT License
// Copyright 2023--present rgpot developers
//
// Dense Hessians and derivative schemes through PotentialHandle and the Rust
// core: the callbacks from_impl registers from caps(), the scheme every
// request reports, the refusals that make no force call, and agreement with
// the kernels and with HessianVectorCache.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/potential.hpp"

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::InputSpec;
using rgpot::LJPot;
using rgpot::MorsePot;
using rgpot::PotentialHandle;

namespace {

/// Morse seen through forceImpl only, with its force calls counted.
struct CountingMorse {
  MorsePot pot;
  mutable std::uint64_t calls = 0;
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    pot.forceImpl(in, out);
  }
};

struct System {
  std::vector<double> x;
  std::vector<int> z;
  std::vector<double> box;
  System(double spacing, double cell, int atomic_number)
      : box{cell, 0.0, 0.0, 0.0, cell, 0.0, 0.0, 0.0, cell} {
    const double s = spacing / std::sqrt(2.0);
    const double sites[][3] = {{0, 0, 0},  {s, s, 0},  {s, 0, s}, {0, s, s},
                               {-s, s, 0}, {s, -s, 0}, {0, -s, s}};
    for (std::size_t i = 0; i < 7; ++i) {
      for (std::size_t k = 0; k < 3; ++k) {
        x.push_back(0.5 * cell + sites[i][k] +
                    0.03 * spacing * std::sin(1.3 * double(3 * i + k)));
      }
      z.push_back(atomic_number);
    }
  }
  std::size_t dim() const { return x.size(); }
  InputSpec input() {
    return InputSpec(z.size(), x.data(), z.data(), box.data());
  }
  ForceInput legacy() const {
    return {.nAtoms = z.size(),
            .pos = x.data(),
            .atmnrs = z.data(),
            .box = box.data()};
  }
};

System ljCluster() { return System(1.12, 30.0, 1); }
System ptCluster() { return System(2.897, 20.0, 78); }

std::vector<double> direction(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) {
    v[k] = std::sin(0.7 * double(k) + 0.4);
  }
  return v;
}

double largest(const std::vector<double> &a) {
  double m = 0.0;
  for (double v : a) {
    m = std::max(m, std::abs(v));
  }
  return m;
}

double largestDifference(const std::vector<double> &a,
                         const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
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

} // namespace

TEST_CASE("from_impl registers every derivative callback LJPot advertises",
          "[hessian][handle]") {
  auto s = ljCluster();
  LJPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  auto input = s.input();
  const std::size_t dim = s.dim();

  const auto caps = pot.derivative_caps();
  CHECK(caps.hvp_analytic);
  CHECK(caps.hvp_ad);
  CHECK(caps.hvp_fd);
  CHECK(caps.hessian_analytic);
  CHECK(caps.hessian_ad);
  CHECK(caps.hessian_fd);
  CHECK(caps.hvp_forward_force_calls == 1);
  CHECK(caps.hvp_central_force_calls == 2);
  CHECK(caps.hessian_central_force_calls_per_atom == 6);

  std::vector<double> want(dim * dim);
  lj.hessianImpl(s.legacy(), want.data());
  std::vector<double> want_ad(dim * dim);
  lj.hessianAdImpl(s.legacy(), want_ad.data());

  rgpot_hessian_scheme_t used = RGPOT_HESSIAN_CENTRAL;
  const auto exact = pot.hessian(input, RGPOT_HESSIAN_EXACT, 0.0, &used);
  CHECK(used == RGPOT_HESSIAN_ANALYTIC);
  CHECK(exact == want);
  const auto ad = pot.hessian(input, RGPOT_HESSIAN_AD, 0.0, &used);
  CHECK(used == RGPOT_HESSIAN_AD);
  CHECK(ad == want_ad);
  CHECK(largestDifference(ad, exact) <= 1e-12 * largest(exact));
  CHECK(exactlySymmetric(exact, dim));
  CHECK(exactlySymmetric(ad, dim));

  const auto v = direction(dim);
  std::vector<double> hv(dim);
  CHECK(pot.hvp(input, v.data(), hv.data(), 0.0, RGPOT_HVP_AUTO) ==
        RGPOT_HVP_ANALYTIC);
  CHECK(pot.hvp(input, v.data(), hv.data(), 0.0, RGPOT_HVP_EXACT) ==
        RGPOT_HVP_ANALYTIC);
  std::vector<double> want_hv(dim);
  lj.hessianVectorAdImpl(s.legacy(), v.data(), want_hv.data());
  CHECK(pot.hvp(input, v.data(), hv.data(), 0.0, RGPOT_HVP_AD) == RGPOT_HVP_AD);
  CHECK(hv == want_hv);

  const auto counts = pot.derivative_counts();
  CHECK(counts.hessian_analytic == 1);
  CHECK(counts.hessian_ad == 1);
  CHECK(counts.hessian_central == 0);
  CHECK(counts.hessian_force_calls == 0);
  CHECK(counts.hvp_analytic == 2);
  CHECK(counts.hvp_ad == 1);
  CHECK(counts.hvp_forward == 0);
  CHECK(pot.hvp_counts().force_calls == 0);
}

TEST_CASE("Exact takes the dual-number callbacks when the analytic ones are "
          "cleared",
          "[hessian][handle]") {
  auto s = ljCluster();
  LJPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  auto input = s.input();
  pot.set_hessian_callback(nullptr);
  pot.set_hvp_callback(nullptr);
  CHECK_FALSE(pot.derivative_caps().hessian_analytic);
  CHECK_FALSE(pot.derivative_caps().hvp_analytic);

  rgpot_hessian_scheme_t used = RGPOT_HESSIAN_CENTRAL;
  pot.hessian(input, RGPOT_HESSIAN_EXACT, 0.0, &used);
  CHECK(used == RGPOT_HESSIAN_AD);
  rgpot_hvp_scheme_t hvp_used = RGPOT_HVP_AUTO;
  pot.hvp(input, direction(s.dim()), 0.0, RGPOT_HVP_EXACT, &hvp_used);
  CHECK(hvp_used == RGPOT_HVP_AD);
  CHECK_THROWS_WITH(pot.hessian(input, RGPOT_HESSIAN_ANALYTIC),
                    Catch::Matchers::ContainsSubstring(
                        "RGPOT_HESSIAN_ANALYTIC: no Hessian callback"));

  // Auto never takes the dual-number product, and says it differenced.
  pot.hvp(input, direction(s.dim()), 0.0, RGPOT_HVP_AUTO, &hvp_used);
  CHECK(hvp_used == RGPOT_HVP_FORWARD);
  CHECK(pot.derivative_counts().hvp_forward == 1);
}

TEST_CASE("from_impl registers the analytic Hessian of MorsePot only",
          "[hessian][handle]") {
  auto s = ptCluster();
  MorsePot morse;
  auto pot = PotentialHandle::from_impl(morse);
  auto input = s.input();
  const std::size_t dim = s.dim();

  const auto caps = pot.derivative_caps();
  CHECK(caps.hessian_analytic);
  CHECK_FALSE(caps.hessian_ad);
  CHECK(caps.hvp_analytic);
  CHECK_FALSE(caps.hvp_ad);

  std::vector<double> want(dim * dim);
  morse.hessianImpl(s.legacy(), want.data());
  std::vector<double> got(dim * dim);
  CHECK(pot.hessian(input, got.data(), RGPOT_HESSIAN_EXACT) ==
        RGPOT_HESSIAN_ANALYTIC);
  CHECK(got == want);

  CHECK_THROWS_WITH(
      pot.hessian(input, RGPOT_HESSIAN_AD),
      Catch::Matchers::ContainsSubstring(
          "RGPOT_HESSIAN_AD: no dual-number Hessian callback is registered"));
  const auto v = direction(dim);
  CHECK_THROWS_WITH(
      pot.hvp(input, v, 0.0, RGPOT_HVP_AD),
      Catch::Matchers::ContainsSubstring(
          "RGPOT_HVP_AD: no dual-number Hessian-vector callback"));
  CHECK(pot.derivative_counts().hessian_ad == 0);
  CHECK(pot.derivative_counts().hvp_ad == 0);
}

TEST_CASE("A forces-only handle refuses exact schemes without a force call",
          "[hessian][handle]") {
  auto s = ptCluster();
  CountingMorse fd;
  auto pot = PotentialHandle::from_impl(fd);
  auto input = s.input();
  const auto v = direction(s.dim());

  const auto caps = pot.derivative_caps();
  CHECK_FALSE(caps.hvp_analytic);
  CHECK_FALSE(caps.hvp_ad);
  CHECK_FALSE(caps.hessian_analytic);
  CHECK_FALSE(caps.hessian_ad);
  CHECK(caps.hvp_fd);
  CHECK(caps.hessian_fd);

  for (const auto scheme :
       {RGPOT_HESSIAN_ANALYTIC, RGPOT_HESSIAN_AD, RGPOT_HESSIAN_EXACT}) {
    CHECK_THROWS_AS(pot.hessian(input, scheme), rgpot::Error);
  }
  for (const auto scheme :
       {RGPOT_HVP_ANALYTIC, RGPOT_HVP_AD, RGPOT_HVP_EXACT}) {
    CHECK_THROWS_AS(pot.hvp(input, v, 0.0, scheme), rgpot::Error);
  }
  CHECK_THROWS_WITH(pot.hessian(input, RGPOT_HESSIAN_EXACT),
                    Catch::Matchers::ContainsSubstring(
                        "RGPOT_HESSIAN_EXACT: neither an analytic nor a "
                        "dual-number Hessian callback is registered"));
  CHECK(fd.calls == 0);
  const auto counts = pot.derivative_counts();
  CHECK(counts.hessian_central + counts.hessian_analytic + counts.hessian_ad ==
        0);
  CHECK(counts.hvp_forward + counts.hvp_central + counts.hvp_analytic +
            counts.hvp_ad ==
        0);

  rgpot_hvp_scheme_t used = RGPOT_HVP_EXACT;
  pot.hvp(input, v, 0.0, RGPOT_HVP_AUTO, &used);
  CHECK(used == RGPOT_HVP_FORWARD);
  CHECK(pot.derivative_counts().hvp_forward == 1);
}

TEST_CASE("The central Hessian of a handle costs 6N force calls",
          "[hessian][handle]") {
  auto s = ptCluster();
  CountingMorse fd;
  auto pot = PotentialHandle::from_impl(fd);
  auto input = s.input();
  const std::size_t dim = s.dim();

  rgpot_hessian_scheme_t used = RGPOT_HESSIAN_EXACT;
  const auto central = pot.hessian(input, RGPOT_HESSIAN_CENTRAL, 0.0, &used);
  CHECK(used == RGPOT_HESSIAN_CENTRAL);
  CHECK(fd.calls == 2 * dim);
  CHECK(pot.derivative_counts().hessian_force_calls == 2 * dim);
  CHECK(pot.derivative_counts().hessian_central == 1);
  CHECK(pot.hvp_counts().force_calls == 0);
  CHECK(exactlySymmetric(central, dim));

  std::vector<double> exact(dim * dim);
  fd.pot.hessianImpl(s.legacy(), exact.data());
  CHECK(largestDifference(central, exact) <= 1e-5 * largest(exact));

  rgpot::HessianVectorCache cache;
  CountingMorse direct;
  std::vector<double> want(dim * dim);
  CHECK(cache.hessian(direct, s.legacy(), want.data(),
                      rgpot::HessianScheme::Central,
                      2e-3) == rgpot::HessianScheme::Central);
  const auto got = pot.hessian(input, RGPOT_HESSIAN_CENTRAL, 2e-3);
  CHECK(largestDifference(got, want) <= 1e-12 * largest(want));

  pot.reset_hvp_counts();
  const auto zero = pot.derivative_counts();
  CHECK(zero.hessian_central == 0);
  CHECK(zero.hessian_force_calls == 0);
}

TEST_CASE("Handle Hessian requests refuse a bad step or scheme",
          "[hessian][handle]") {
  auto s = ljCluster();
  LJPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  auto input = s.input();
  std::vector<double> h(s.dim() * s.dim());
  for (const double step : {-1e-3, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
    CHECK_THROWS_AS(pot.hessian(input, h.data(), RGPOT_HESSIAN_CENTRAL, step),
                    rgpot::Error);
  }
  for (const int raw : {0, 1, 6}) {
    CHECK_THROWS_WITH(
        pot.hessian(input, h.data(), static_cast<rgpot_hessian_scheme_t>(raw)),
        Catch::Matchers::ContainsSubstring("unknown scheme"));
  }
  const auto counts = pot.derivative_counts();
  CHECK(counts.hessian_central + counts.hessian_analytic + counts.hessian_ad ==
        0);
}

TEST_CASE("A throwing Hessian kernel reaches the caller with its message",
          "[hessian][handle]") {
  LJPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  std::vector<double> x(6, 0.0);
  std::vector<int> z{1, 1};
  std::vector<double> box{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  InputSpec input(z.size(), x.data(), z.data(), box.data());
  for (const auto scheme : {RGPOT_HESSIAN_ANALYTIC, RGPOT_HESSIAN_AD}) {
    CHECK_THROWS_WITH(pot.hessian(input, scheme),
                      Catch::Matchers::ContainsSubstring("coincident atoms"));
  }
}
