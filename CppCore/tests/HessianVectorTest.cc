// MIT License
// Copyright 2023--present rgpot developers
//
// Hessian-vector products (HessianVector.hpp): the analytic Morse product
// against a dense Hessian built here, the orders of the finite differences,
// the force calls a Lanczos loop makes, and when the cached centre is
// dropped.

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
#include "rgpot/Morse/MorsePot.hpp"

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::HessianVectorCache;
using rgpot::HvpScheme;
using rgpot::MorseConfig;
using rgpot::MorsePot;

namespace {

constexpr double kCell = 30.0;

/// A 13-atom cuboctahedron at the Morse distance, each atom pushed off its
/// site so no pair sits at a symmetric point of the Hessian.
std::vector<double> cluster() {
  const double s = 2.897 / std::sqrt(2.0);
  std::vector<double> x = {0.0, 0.0, 0.0};
  for (int a = -1; a <= 1; a += 2) {
    for (int b = -1; b <= 1; b += 2) {
      x.insert(x.end(), {a * s, b * s, 0.0});
      x.insert(x.end(), {a * s, 0.0, b * s});
      x.insert(x.end(), {0.0, a * s, b * s});
    }
  }
  for (std::size_t k = 0; k < x.size(); ++k) {
    x[k] += 15.0 + 0.11 * std::sin(1.7 * static_cast<double>(k) + 0.3);
  }
  return x;
}

std::array<double, 9> cell(double l = kCell) {
  return {l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l};
}

std::vector<double> direction(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) {
    v[k] = std::cos(0.9 * static_cast<double>(k) + 0.2);
  }
  return v;
}

/// Dense Morse Hessian of a cluster with no periodic neighbours, from the
/// closed-form pair derivatives.
std::vector<double> denseHessian(const MorseConfig &c, const std::vector<double> &x) {
  const std::size_t n = x.size() / 3;
  const std::size_t dim = x.size();
  std::vector<double> h(dim * dim, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double d[3];
      double r2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        d[k] = x[3 * i + k] - x[3 * j + k];
        r2 += d[k] * d[k];
      }
      const double r = std::sqrt(r2);
      if (r > c.cutoff) {
        continue;
      }
      const double e = std::exp(-c.a * (r - c.re));
      const double d1 = 2.0 * c.De * c.a * e * (1.0 - e);
      const double d2 = 2.0 * c.De * c.a * c.a * e * (2.0 * e - 1.0);
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
          const double uu = d[p] * d[q] / r2;
          const double k = d2 * uu + (d1 / r) * ((p == q ? 1.0 : 0.0) - uu);
          h[(3 * i + p) * dim + 3 * i + q] += k;
          h[(3 * j + p) * dim + 3 * j + q] += k;
          h[(3 * i + p) * dim + 3 * j + q] -= k;
          h[(3 * j + p) * dim + 3 * i + q] -= k;
        }
      }
    }
  }
  return h;
}

std::vector<double> multiply(const std::vector<double> &h, const std::vector<double> &v) {
  const std::size_t dim = v.size();
  std::vector<double> out(dim, 0.0);
  for (std::size_t r = 0; r < dim; ++r) {
    for (std::size_t c = 0; c < dim; ++c) {
      out[r] += h[r * dim + c] * v[c];
    }
  }
  return out;
}

double norm(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v) {
    s += x * x;
  }
  return std::sqrt(s);
}

double relativeError(const std::vector<double> &got, const std::vector<double> &want) {
  std::vector<double> diff(got.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    diff[k] = got[k] - want[k];
  }
  return norm(diff) / norm(want);
}

/// A Morse kernel seen through forceImpl only, so every product is a finite
/// difference, with the force calls it receives counted.
struct CountingMorse {
  MorsePot pot;
  mutable std::uint64_t calls = 0;
  explicit CountingMorse(const MorseConfig &c = {}) : pot(c) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    pot.forceImpl(in, out);
  }
};

static_assert(rgpot::AnalyticHessianVector<MorsePot>);
static_assert(!rgpot::AnalyticHessianVector<CountingMorse>);

struct System {
  std::vector<double> x = cluster();
  std::vector<int> z = std::vector<int>(13, 78);
  std::array<double, 9> box = cell();
  ForceInput input() const {
    return {.nAtoms = z.size(), .pos = x.data(), .atmnrs = z.data(), .box = box.data()};
  }
};

std::vector<double> apply(HessianVectorCache &cache, auto &pot, const System &s,
                          const std::vector<double> &v, double step, HvpScheme scheme) {
  std::vector<double> hv(v.size());
  cache.apply(pot, s.input(), v.data(), hv.data(), step, scheme);
  return hv;
}

/// Smallest eigenvalue of the symmetric tridiagonal (alpha, beta) by
/// Sturm-sequence bisection.
double smallestRitz(const std::vector<double> &alpha, const std::vector<double> &beta) {
  double lo = 0.0;
  double hi = 0.0;
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    const double off = (i > 0 ? std::abs(beta[i - 1]) : 0.0) +
                       (i + 1 < alpha.size() ? std::abs(beta[i]) : 0.0);
    lo = std::min(lo, alpha[i] - off);
    hi = std::max(hi, alpha[i] + off);
  }
  auto below = [&](double mu) {
    int count = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < alpha.size(); ++i) {
      const double b2 = i > 0 ? beta[i - 1] * beta[i - 1] : 0.0;
      q = alpha[i] - mu - (i > 0 ? b2 / q : 0.0);
      if (q == 0.0) {
        q = 1e-300;
      }
      count += q < 0.0 ? 1 : 0;
    }
    return count;
  };
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (below(mid) >= 1 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

/// k Lanczos steps from @a start with products from @a product.
template <typename Product>
double lanczos(std::size_t k, std::vector<double> q, Product &&product) {
  const double n0 = norm(q);
  for (double &x : q) {
    x /= n0;
  }
  std::vector<double> prev(q.size(), 0.0);
  std::vector<double> alpha;
  std::vector<double> beta;
  double b = 0.0;
  for (std::size_t step = 0; step < k; ++step) {
    std::vector<double> w = product(q);
    double a = 0.0;
    for (std::size_t i = 0; i < q.size(); ++i) {
      a += w[i] * q[i];
    }
    for (std::size_t i = 0; i < q.size(); ++i) {
      w[i] -= a * q[i] + b * prev[i];
    }
    alpha.push_back(a);
    b = norm(w);
    if (step + 1 < k) {
      beta.push_back(b);
    }
    prev = q;
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] = w[i] / b;
    }
  }
  return smallestRitz(alpha, beta);
}

} // namespace

TEST_CASE("MorsePot hessianVectorImpl matches the dense Hessian", "[HessianVector]") {
  System s;
  MorseConfig config{};
  MorsePot pot(config);
  const auto v = direction(s.x.size());
  std::vector<double> hv(v.size());
  pot.hessianVectorImpl(s.input(), v.data(), hv.data());
  const auto want = multiply(denseHessian(config, s.x), v);
  CHECK(relativeError(hv, want) < 1e-12);
}

TEST_CASE("MorsePot hessianVectorImpl differentiates the switch", "[HessianVector]") {
  // Most pairs of the cluster fall inside the switch region.
  System s;
  MorseConfig config{.cutoff = 6.0, .switch_width = 3.5};
  CountingMorse fd(config);
  MorsePot pot(config);
  const auto v = direction(s.x.size());
  HessianVectorCache cache;
  const auto analytic = apply(cache, pot, s, v, 0.0, HvpScheme::Analytic);
  const auto central = apply(cache, fd, s, v, 1e-4, HvpScheme::Central);
  CHECK(relativeError(central, analytic) < 1e-7);
}

TEST_CASE("Finite differences converge at first and second order", "[HessianVector]") {
  System s;
  CountingMorse fd;
  const auto v = direction(s.x.size());
  const auto want = multiply(denseHessian(MorseConfig{}, s.x), v);
  auto order = [&](HvpScheme scheme, double h) {
    HessianVectorCache cache;
    const double coarse = relativeError(apply(cache, fd, s, v, h, scheme), want);
    const double fine = relativeError(apply(cache, fd, s, v, h / 2.0, scheme), want);
    return std::log2(coarse / fine);
  };
  CHECK_THAT(order(HvpScheme::Forward, 4e-3), Catch::Matchers::WithinAbs(1.0, 0.1));
  CHECK_THAT(order(HvpScheme::Central, 2e-2), Catch::Matchers::WithinAbs(2.0, 0.1));

  HessianVectorCache cache;
  const double forward = relativeError(apply(cache, fd, s, v, 0.0, HvpScheme::Forward), want);
  const double central = relativeError(apply(cache, fd, s, v, 0.0, HvpScheme::Central), want);
  CHECK(forward < 1e-2);
  CHECK(central < 1e-5);
  CHECK(central < 1e-2 * forward);
}

TEST_CASE("A Lanczos loop of k products costs k + 1 force calls", "[HessianVector]") {
  System s;
  const std::size_t k = 12;
  const auto start = direction(s.x.size());

  CountingMorse fd;
  HessianVectorCache cache;
  const double cached = lanczos(k, start, [&](const std::vector<double> &q) {
    return apply(cache, fd, s, q, 0.0, HvpScheme::Forward);
  });
  CHECK(fd.calls == k + 1);
  const auto counts = cache.counts();
  CHECK(counts.products == k);
  CHECK(counts.forceCalls == k + 1);
  CHECK(counts.centreReused == k - 1);

  // The loop a force-only consumer writes: centre and displaced forces for
  // every product.
  CountingMorse naive;
  const double uncached = lanczos(k, start, [&](const std::vector<double> &q) {
    HessianVectorCache fresh;
    return apply(fresh, naive, s, q, 0.0, HvpScheme::Forward);
  });
  CHECK(naive.calls == 2 * k);
  CHECK_THAT(cached, Catch::Matchers::WithinRel(uncached, 1e-12));

  MorsePot pot;
  HessianVectorCache exact;
  const double analytic = lanczos(k, start, [&](const std::vector<double> &q) {
    return apply(exact, pot, s, q, 0.0, HvpScheme::Auto);
  });
  CHECK(exact.counts().forceCalls == 0);
  CHECK(exact.counts().analytic == k);
  CHECK_THAT(cached, Catch::Matchers::WithinAbs(analytic, 1e-2 * std::abs(analytic)));
}

TEST_CASE("The centre call primes the products", "[HessianVector]") {
  System s;
  CountingMorse fd;
  HessianVectorCache cache;
  const auto centre = cache.centre(fd, s.input());
  const auto v = direction(s.x.size());
  for (int i = 0; i < 4; ++i) {
    apply(cache, fd, s, v, 0.0, HvpScheme::Forward);
  }
  CHECK(fd.calls == 5);
  CHECK(cache.counts().centreReused == 4);

  std::vector<double> f(s.x.size());
  ForceOut out{.F = f.data(), .energy = 0.0, .variance = 0.0, .stress = {}, .has_stress = 0};
  MorsePot(MorseConfig{}).forceImpl(s.input(), &out);
  CHECK(centre.energy == out.energy);
  CHECK(centre.forces == f);
}

TEST_CASE("A new geometry, cell or species is a new centre", "[HessianVector]") {
  System s;
  CountingMorse fd;
  MorsePot exact;
  HessianVectorCache cache;
  const auto v = direction(s.x.size());
  apply(cache, fd, s, v, 0.0, HvpScheme::Forward);
  REQUIRE(fd.calls == 2);

  auto expectRecompute = [&](const char *what) {
    INFO(what);
    const std::uint64_t before = fd.calls;
    const auto hv = apply(cache, fd, s, v, 0.0, HvpScheme::Forward);
    CHECK(fd.calls == before + 2);
    HessianVectorCache scratch;
    const auto want = apply(scratch, exact, s, v, 0.0, HvpScheme::Analytic);
    CHECK(relativeError(hv, want) < 1e-2);
    const std::uint64_t again = fd.calls;
    apply(cache, fd, s, v, 0.0, HvpScheme::Forward);
    CHECK(fd.calls == again + 1);
  };

  s.x[4] = std::nextafter(s.x[4], std::numeric_limits<double>::infinity());
  expectRecompute("one coordinate moved by one ulp");
  s.x[7] += 0.05;
  expectRecompute("one coordinate moved");
  s.box = cell(31.0);
  expectRecompute("cell changed");
  s.z[3] = 79;
  expectRecompute("species changed");
  cache.invalidate();
  expectRecompute("invalidated by hand");
}

TEST_CASE("Signed zero is a different centre", "[HessianVector]") {
  System s;
  s.x[0] = 0.0;
  CountingMorse fd;
  HessianVectorCache cache;
  cache.centre(fd, s.input());
  s.x[0] = -0.0;
  cache.centre(fd, s.input());
  CHECK(fd.calls == 2);
}

TEST_CASE("Products are linear and refuse malformed requests", "[HessianVector]") {
  System s;
  CountingMorse fd;
  HessianVectorCache cache;
  const auto v = direction(s.x.size());
  auto scaled = [&](double c) {
    auto out = v;
    for (double &x : out) {
      x *= c;
    }
    return out;
  };
  // Forward displaces along +v/|v| whatever the length, so it scales with
  // a positive factor. A negative one moves the displacement to the other
  // side, where the O(h) term has the same sign; central is odd.
  auto expectScaled = [](const std::vector<double> &got, const std::vector<double> &base,
                         double c) {
    for (std::size_t k = 0; k < got.size(); ++k) {
      CHECK_THAT(got[k], Catch::Matchers::WithinAbs(c * base[k], 1e-12 * std::abs(c * base[k])));
    }
  };
  const auto forward = apply(cache, fd, s, v, 0.0, HvpScheme::Forward);
  expectScaled(apply(cache, fd, s, scaled(3.0), 0.0, HvpScheme::Forward), forward, 3.0);
  const auto central = apply(cache, fd, s, v, 0.0, HvpScheme::Central);
  expectScaled(apply(cache, fd, s, scaled(-3.0), 0.0, HvpScheme::Central), central, -3.0);

  const std::uint64_t before = fd.calls;
  const std::vector<double> zero(v.size(), 0.0);
  const auto hz = apply(cache, fd, s, zero, 0.0, HvpScheme::Central);
  CHECK(fd.calls == before);
  CHECK(std::all_of(hz.begin(), hz.end(), [](double x) { return x == 0.0; }));

  std::vector<double> out(v.size());
  CHECK_THROWS_AS(cache.apply(fd, s.input(), v.data(), out.data(), -1.0), std::invalid_argument);
  CHECK_THROWS_AS(cache.apply(fd, s.input(), v.data(), out.data(), 0.0, HvpScheme::Analytic),
                  std::invalid_argument);
  CHECK_THROWS_AS(cache.apply(fd, s.input(), v.data(), out.data(), 0.0, static_cast<HvpScheme>(7)),
                  std::invalid_argument);
  auto bad = v;
  bad[2] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(cache.apply(fd, s.input(), bad.data(), out.data()), std::invalid_argument);
  auto alias = v;
  CHECK_THROWS_AS(cache.apply(fd, s.input(), alias.data(), alias.data() + 1),
                  std::invalid_argument);
  CHECK(fd.calls == before);
}
