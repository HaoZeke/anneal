// MIT License
// Copyright 2023--present rgpot developers
//
// The schemes of HessianVector.hpp: which one a request resolves to, that
// the scheme that ran is returned and counted, that Analytic, Ad and Exact
// refuse what a potential lacks without a force call, the dense Hessian of
// each scheme, and the capability report.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"

using rgpot::DerivativeCaps;
using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::HessianScheme;
using rgpot::HessianVectorCache;
using rgpot::HvpScheme;
using rgpot::LJConfig;
using rgpot::LJPot;
using rgpot::MorseConfig;
using rgpot::MorsePot;

namespace {

/// A Morse kernel seen through forceImpl only, with its force calls
/// counted.
struct ForcesOnly {
  MorsePot pot;
  mutable std::uint64_t calls = 0;
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    pot.forceImpl(in, out);
  }
};

/// LJPot with its analytic derivatives hidden, so only the dual-number ones
/// and the finite differences remain.
struct DualOnly {
  LJPot pot{LJConfig{.cutoff = 3.0}};
  mutable std::uint64_t calls = 0;
  [[nodiscard]] rgpot::PotCaps caps() const noexcept {
    auto c = pot.caps();
    c.analyticHvp = false;
    c.analyticHessian = false;
    return c;
  }
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    pot.forceImpl(in, out);
  }
  void hessianAdImpl(const ForceInput &in, double *h) const {
    pot.hessianAdImpl(in, h);
  }
  void hessianVectorAdImpl(const ForceInput &in, const double *v,
                           double *hv) const {
    pot.hessianVectorAdImpl(in, v, hv);
  }
};

/// Forces of +-1.5e308 on a coordinate moved up or down from @c x0, so that
/// every central difference overflows.
struct Cliff {
  std::vector<double> x0;
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    for (std::size_t k = 0; k < x0.size(); ++k) {
      const double d = in.pos[k] - x0[k];
      out->F[k] = d > 0.0 ? 1.5e308 : (d < 0.0 ? -1.5e308 : 0.0);
    }
    out->energy = 0.0;
    out->variance = 0.0;
  }
};

static_assert(!rgpot::AnalyticHessianVector<ForcesOnly>);
static_assert(rgpot::AdHessianVectorKernel<DualOnly>);
static_assert(!rgpot::AnalyticHessianKernel<DualOnly>);

/// @a n atoms of a cuboctahedron of nearest-neighbour distance @a spacing,
/// centre first, each coordinate pushed off its site.
struct System {
  std::vector<double> x;
  std::vector<int> z;
  std::array<double, 9> box{30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0};

  explicit System(double spacing) {
    const double s = spacing / std::sqrt(2.0);
    x = {0.0, 0.0, 0.0};
    for (int a = -1; a <= 1; a += 2) {
      for (int b = -1; b <= 1; b += 2) {
        x.insert(x.end(), {a * s, b * s, 0.0});
        x.insert(x.end(), {a * s, 0.0, b * s});
        x.insert(x.end(), {0.0, a * s, b * s});
      }
    }
    for (std::size_t k = 0; k < x.size(); ++k) {
      x[k] +=
          15.0 + 0.04 * spacing * std::sin(1.7 * static_cast<double>(k) + 0.3);
    }
    z.assign(13, 1);
  }
  [[nodiscard]] ForceInput input() const {
    return {.nAtoms = z.size(),
            .pos = x.data(),
            .atmnrs = z.data(),
            .box = box.data()};
  }
  [[nodiscard]] std::size_t dim() const { return x.size(); }
};

const System &morseSystem() {
  static const System s(2.897);
  return s;
}

const System &ljSystem() {
  static const System s(1.12);
  return s;
}

std::vector<double> direction(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) {
    v[k] = std::cos(0.9 * static_cast<double>(k) + 0.2);
  }
  return v;
}

double largestDifference(const std::vector<double> &a,
                         const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

double largestMagnitude(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
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

TEST_CASE("derivativeCaps reports what each potential has", "[schemes]") {
  const DerivativeCaps lj = rgpot::derivativeCaps(LJPot());
  CHECK(lj.hvpAnalytic);
  CHECK(lj.hvpAd);
  CHECK(lj.hvpFd);
  CHECK(lj.hessianAnalytic);
  CHECK(lj.hessianAd);
  CHECK(lj.hessianFd);

  const DerivativeCaps morse = rgpot::derivativeCaps(MorsePot());
  CHECK(morse.hvpAnalytic);
  CHECK_FALSE(morse.hvpAd);
  CHECK(morse.hessianAnalytic);
  CHECK_FALSE(morse.hessianAd);

  const DerivativeCaps forces = rgpot::derivativeCaps(ForcesOnly());
  CHECK_FALSE(forces.hvpAnalytic);
  CHECK_FALSE(forces.hvpAd);
  CHECK(forces.hvpFd);
  CHECK_FALSE(forces.hessianAnalytic);
  CHECK_FALSE(forces.hessianAd);
  CHECK(forces.hessianFd);
  CHECK(forces.hvpForwardForceCalls == 1);
  CHECK(forces.hvpCentralForceCalls == 2);
  CHECK(forces.hessianCentralForceCallsPerAtom == 6);

  const DerivativeCaps dual = rgpot::derivativeCaps(DualOnly());
  CHECK_FALSE(dual.hvpAnalytic);
  CHECK(dual.hvpAd);
  CHECK_FALSE(dual.hessianAnalytic);
  CHECK(dual.hessianAd);
}

TEST_CASE("apply returns and counts the scheme that ran", "[schemes]") {
  const auto v = direction(morseSystem().dim());
  std::vector<double> hv(v.size());

  SECTION("Auto falls back to Forward without an analytic product") {
    ForcesOnly pot;
    HessianVectorCache cache;
    CHECK(cache.apply(pot, morseSystem().input(), v.data(), hv.data()) ==
          HvpScheme::Forward);
    CHECK(cache.apply(pot, morseSystem().input(), v.data(), hv.data(), 0.0,
                      HvpScheme::Central) == HvpScheme::Central);
    const auto counts = cache.derivativeCounts();
    CHECK(counts.hvpForward == 1);
    CHECK(counts.hvpCentral == 1);
    CHECK(counts.hvpAnalytic == 0);
    CHECK(cache.counts().products == 2);
    CHECK(pot.calls == 4);
  }

  SECTION("Auto and Exact take the analytic product when there is one") {
    MorsePot pot;
    HessianVectorCache cache;
    CHECK(cache.apply(pot, morseSystem().input(), v.data(), hv.data()) ==
          HvpScheme::Analytic);
    CHECK(cache.apply(pot, morseSystem().input(), v.data(), hv.data(), 0.0,
                      HvpScheme::Exact) == HvpScheme::Analytic);
    CHECK(cache.derivativeCounts().hvpAnalytic == 2);
    CHECK(cache.counts().analytic == 2);
    CHECK(cache.counts().forceCalls == 0);
  }

  SECTION("Exact takes the dual-number product when it is the only one") {
    DualOnly pot;
    HessianVectorCache cache;
    CHECK(cache.apply(pot, ljSystem().input(), v.data(), hv.data(), 0.0,
                      HvpScheme::Exact) == HvpScheme::Ad);
    CHECK(cache.apply(pot, ljSystem().input(), v.data(), hv.data()) ==
          HvpScheme::Forward);
    const auto counts = cache.derivativeCounts();
    CHECK(counts.hvpAd == 1);
    CHECK(counts.hvpForward == 1);
    CHECK(cache.counts().analytic == 0);
  }

  SECTION("A zero direction counts under its scheme without a kernel call") {
    MorsePot pot;
    HessianVectorCache cache;
    const std::vector<double> zero(v.size(), 0.0);
    CHECK(cache.apply(pot, morseSystem().input(), zero.data(), hv.data(), 0.0,
                      HvpScheme::Exact) == HvpScheme::Analytic);
    CHECK(std::all_of(hv.begin(), hv.end(), [](double x) { return x == 0.0; }));
    CHECK(cache.derivativeCounts().hvpAnalytic == 1);
    CHECK(cache.counts().analytic == 0);
    CHECK(cache.counts().products == 1);
  }
}

TEST_CASE("The dual-number product equals the analytic one", "[schemes]") {
  LJPot pot(LJConfig{.cutoff = 3.0, .switch_width = 1.65});
  const auto v = direction(ljSystem().dim());
  std::vector<double> analytic(v.size());
  std::vector<double> dual(v.size());
  HessianVectorCache cache;
  CHECK(cache.apply(pot, ljSystem().input(), v.data(), analytic.data(), 0.0,
                    HvpScheme::Analytic) == HvpScheme::Analytic);
  CHECK(cache.apply(pot, ljSystem().input(), v.data(), dual.data(), 0.0,
                    HvpScheme::Ad) == HvpScheme::Ad);
  CHECK(largestDifference(dual, analytic) <=
        1e-12 * largestMagnitude(analytic));
  CHECK(cache.counts().forceCalls == 0);
}

TEST_CASE("Analytic, Ad and Exact refuse without a force call", "[schemes]") {
  const auto v = direction(morseSystem().dim());
  std::vector<double> hv(v.size());
  std::vector<double> h(v.size() * v.size());
  ForcesOnly forces;
  MorsePot morse;
  HessianVectorCache cache;
  const ForceInput in = morseSystem().input();
  for (const HvpScheme scheme :
       {HvpScheme::Analytic, HvpScheme::Ad, HvpScheme::Exact}) {
    CHECK_THROWS_AS(cache.apply(forces, in, v.data(), hv.data(), 0.0, scheme),
                    std::invalid_argument);
  }
  CHECK_THROWS_AS(
      cache.apply(morse, in, v.data(), hv.data(), 0.0, HvpScheme::Ad),
      std::invalid_argument);
  for (const HessianScheme scheme :
       {HessianScheme::Analytic, HessianScheme::Ad, HessianScheme::Exact}) {
    CHECK_THROWS_AS(cache.hessian(forces, in, h.data(), scheme),
                    std::invalid_argument);
  }
  CHECK_THROWS_AS(cache.hessian(morse, in, h.data(), HessianScheme::Ad),
                  std::invalid_argument);
  CHECK(forces.calls == 0);
  const auto counts = cache.derivativeCounts();
  CHECK(counts.hvpForward + counts.hvpCentral + counts.hvpAnalytic +
            counts.hvpAd ==
        0);
  CHECK(counts.hessianCentral + counts.hessianAnalytic + counts.hessianAd == 0);
  CHECK(cache.counts().products == 0);
}

TEST_CASE("hessian refuses malformed requests", "[schemes]") {
  MorsePot pot;
  HessianVectorCache cache;
  const ForceInput in = morseSystem().input();
  std::vector<double> h(morseSystem().dim() * morseSystem().dim());
  CHECK_THROWS_AS(cache.hessian(pot, in, nullptr, HessianScheme::Analytic),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      cache.hessian(pot, in, h.data(), HessianScheme::Central, -1.0),
      std::invalid_argument);
  for (const int raw : {0, 1, 6}) {
    CHECK_THROWS_AS(
        cache.hessian(pot, in, h.data(), static_cast<HessianScheme>(raw)),
        std::invalid_argument);
  }
  CHECK_THROWS_AS(cache.apply(pot, in, h.data(), h.data() + in.nAtoms * 3, 0.0,
                              static_cast<HvpScheme>(6)),
                  std::invalid_argument);
}

TEST_CASE("hessian refuses an output that overlaps its input", "[schemes]") {
  const System &s = morseSystem();
  const std::size_t dim = s.dim();
  ForcesOnly pot;
  HessianVectorCache cache;
  std::vector<double> shared(dim * dim);
  std::copy(s.x.begin(), s.x.end(), shared.end() - static_cast<long>(dim));
  const ForceInput tail{.nAtoms = s.z.size(),
                        .pos = shared.data() + (dim * dim - dim),
                        .atmnrs = s.z.data(),
                        .box = s.box.data()};
  CHECK_THROWS_AS(
      cache.hessian(pot, tail, shared.data(), HessianScheme::Central),
      std::invalid_argument);
  std::vector<double> h(dim * dim);
  std::copy(s.box.begin(), s.box.end(), h.begin() + static_cast<long>(dim));
  const ForceInput cell{.nAtoms = s.z.size(),
                        .pos = s.x.data(),
                        .atmnrs = s.z.data(),
                        .box = h.data() + dim};
  CHECK_THROWS_AS(cache.hessian(pot, cell, h.data(), HessianScheme::Central),
                  std::invalid_argument);
  const ForceInput species{.nAtoms = s.z.size(),
                           .pos = s.x.data(),
                           .atmnrs =
                               reinterpret_cast<const int *>(h.data() + dim),
                           .box = s.box.data()};
  CHECK_THROWS_AS(cache.hessian(pot, species, h.data(), HessianScheme::Central),
                  std::invalid_argument);
  // The largest atom count whose (3N)^2 doubles fit in size_t: there the
  // output's end address passes the top of the address space.
  const ForceInput largest{.nAtoms = 506166749,
                           .pos = h.data(),
                           .atmnrs = s.z.data(),
                           .box = s.box.data()};
  CHECK_THROWS_AS(cache.hessian(pot, largest, h.data(), HessianScheme::Central),
                  std::invalid_argument);
  CHECK(pot.calls == 0);
  CHECK(cache.derivativeCounts().hessianForceCalls == 0);
}

TEST_CASE("An atom count whose derivatives overflow is refused unread",
          "[schemes]") {
  const System &s = morseSystem();
  ForcesOnly pot;
  HessianVectorCache cache;
  std::vector<double> h(s.dim() * s.dim());
  std::vector<double> v(s.dim(), 1.0);
  std::vector<double> hv(s.dim());
  const auto claiming = [&s](std::size_t nAtoms) {
    return ForceInput{.nAtoms = nAtoms,
                      .pos = s.x.data(),
                      .atmnrs = s.z.data(),
                      .box = s.box.data()};
  };
  const std::size_t wraps = std::numeric_limits<std::size_t>::max() / 3 + 1;
  const std::size_t squareOverflows = std::size_t{1} << 31;
  CHECK_THROWS_AS(
      cache.hessian(pot, claiming(wraps), h.data(), HessianScheme::Central),
      std::invalid_argument);
  CHECK_THROWS_AS(cache.hessian(pot, claiming(squareOverflows), h.data(),
                                HessianScheme::Central),
                  std::invalid_argument);
  CHECK_THROWS_AS(cache.apply(pot, claiming(wraps), v.data(), hv.data()),
                  std::invalid_argument);
  const auto pastPtrdiff =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
          (3 * sizeof(double)) +
      1;
  CHECK_THROWS_AS(cache.apply(pot, claiming(pastPtrdiff), v.data(), hv.data()),
                  std::invalid_argument);
  CHECK(pot.calls == 0);
}

TEST_CASE("A central Hessian that overflows is refused", "[schemes]") {
  const System &s = morseSystem();
  Cliff pot{s.x};
  HessianVectorCache cache;
  std::vector<double> h(s.dim() * s.dim());
  CHECK_THROWS_AS(
      cache.hessian(pot, s.input(), h.data(), HessianScheme::Central),
      std::domain_error);
  CHECK(cache.derivativeCounts().hessianCentral == 0);
}

TEST_CASE("hessian returns and counts the scheme that ran", "[schemes]") {
  const std::size_t dim = ljSystem().dim();
  std::vector<double> analytic(dim * dim);
  std::vector<double> dual(dim * dim);

  LJPot lj(LJConfig{.cutoff = 3.0});
  HessianVectorCache cache;
  CHECK(cache.hessian(lj, ljSystem().input(), analytic.data(),
                      HessianScheme::Exact) == HessianScheme::Analytic);
  CHECK(cache.hessian(lj, ljSystem().input(), dual.data(), HessianScheme::Ad) ==
        HessianScheme::Ad);
  CHECK(largestDifference(dual, analytic) <=
        1e-12 * largestMagnitude(analytic));

  DualOnly dualOnly;
  std::vector<double> viaExact(dim * dim);
  CHECK(cache.hessian(dualOnly, ljSystem().input(), viaExact.data(),
                      HessianScheme::Exact) == HessianScheme::Ad);
  CHECK(viaExact == dual);
  CHECK(dualOnly.calls == 0);

  const auto counts = cache.derivativeCounts();
  CHECK(counts.hessianAnalytic == 1);
  CHECK(counts.hessianAd == 2);
  CHECK(counts.hessianCentral == 0);
  CHECK(counts.hessianForceCalls == 0);
}

TEST_CASE("A central Hessian costs 6N force calls and is symmetric",
          "[schemes]") {
  const System &s = morseSystem();
  const std::size_t dim = s.dim();
  ForcesOnly forces;
  HessianVectorCache cache;
  std::vector<double> central(dim * dim);
  CHECK(cache.hessian(forces, s.input(), central.data(), HessianScheme::Central,
                      1e-4) == HessianScheme::Central);
  CHECK(forces.calls == 6 * s.z.size());
  CHECK(cache.derivativeCounts().hessianForceCalls == 6 * s.z.size());
  CHECK(cache.derivativeCounts().hessianCentral == 1);
  CHECK(cache.counts().forceCalls == 0);
  CHECK(cache.counts().products == 0);
  CHECK(exactlySymmetric(central, dim));

  std::vector<double> analytic(dim * dim);
  MorsePot().hessianImpl(s.input(), analytic.data());
  const double scale = largestMagnitude(analytic);
  const double fine = largestDifference(central, analytic) / scale;
  CHECK(fine < 1e-7);

  // Second order: halving the step from 2e-2 quarters the error.
  auto error = [&](double step) {
    HessianVectorCache fresh;
    std::vector<double> h(dim * dim);
    fresh.hessian(forces, s.input(), h.data(), HessianScheme::Central, step);
    return largestDifference(h, analytic) / scale;
  };
  CHECK_THAT(std::log2(error(2e-2) / error(1e-2)),
             Catch::Matchers::WithinAbs(2.0, 0.1));
}

TEST_CASE("resetCounts zeros every counter", "[schemes]") {
  LJPot pot(LJConfig{.cutoff = 3.0});
  ForcesOnly forces;
  HessianVectorCache cache;
  const auto v = direction(ljSystem().dim());
  std::vector<double> hv(v.size());
  std::vector<double> h(v.size() * v.size());
  cache.apply(pot, ljSystem().input(), v.data(), hv.data());
  cache.apply(pot, ljSystem().input(), v.data(), hv.data(), 0.0, HvpScheme::Ad);
  cache.apply(forces, morseSystem().input(), v.data(), hv.data());
  cache.apply(forces, morseSystem().input(), v.data(), hv.data(), 0.0,
              HvpScheme::Central);
  cache.hessian(pot, ljSystem().input(), h.data(), HessianScheme::Analytic);
  cache.hessian(pot, ljSystem().input(), h.data(), HessianScheme::Ad);
  cache.hessian(forces, morseSystem().input(), h.data(),
                HessianScheme::Central);
  const auto before = cache.derivativeCounts();
  CHECK(before.hvpAnalytic == 1);
  CHECK(before.hvpAd == 1);
  CHECK(before.hvpForward == 1);
  CHECK(before.hvpCentral == 1);
  CHECK(before.hessianAnalytic == 1);
  CHECK(before.hessianAd == 1);
  CHECK(before.hessianCentral == 1);
  CHECK(before.hessianForceCalls == 78);

  cache.resetCounts();
  const auto after = cache.derivativeCounts();
  for (const std::uint64_t c :
       {after.hvpForward, after.hvpCentral, after.hvpAnalytic, after.hvpAd,
        after.hessianCentral, after.hessianAnalytic, after.hessianAd,
        after.hessianForceCalls}) {
    CHECK(c == 0);
  }
  const auto hvp = cache.counts();
  CHECK(hvp.products == 0);
  CHECK(hvp.forceCalls == 0);
  CHECK(hvp.analytic == 0);
}
