// MIT License
// Copyright 2023--present rgpot developers
//
// Milliseconds per dense Hessian and per Hessian-vector product for LJPot on
// a free cluster, through rgpot::HessianVectorCache, with each result's
// error against the analytic Hessian. The cluster is the n fcc sites nearest
// a point, at the pair-minimum spacing 2^(1/6) psi, plus uniform noise of
// --sigma per coordinate, in a box with 1.1 cutoffs of vacuum on each face so
// that no image is in range. The default LJConfig has a 15 Angstrom cutoff,
// so every pair interacts at these sizes.
//
// One line per scheme at each size:
//
//   hessian analytic | ad | central     the dense Hessian
//   hvp analytic | ad | central         one product
//   hvp forward-first                   a product at a new centre
//   hvp forward-reuse                   a product at the cached centre
//   matvec                              the product with the stored Hessian
//
// and the break-even k* = (t_dense - t_hvp) / t_apply + 1 for the analytic,
// dual-number and finite-difference pairs. Errors are the largest absolute
// gap over the largest absolute entry of the analytic Hessian or of H v.
//
// OMP_NUM_THREADS is set to 1. Pin the process, as with taskset -c 0.
//
// hessian_bench [--sizes 38,100,500] [--runs 5] [--seconds 0.2]
//               [--sigma 0.02] [--seed 7]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

namespace {

using rgpot::ForceInput;
using rgpot::HessianScheme;
using rgpot::HessianVectorCache;
using rgpot::HvpScheme;
using rgpot::LJPot;

struct Opts {
  std::vector<int> sizes{38, 100, 500};
  int runs = 5;
  double seconds = 0.2;
  double sigma = 0.02;
  std::uint64_t seed = 7;
};

std::vector<int> parseInts(const std::string &text) {
  std::vector<int> out;
  std::stringstream ss(text);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    out.push_back(std::atoi(tok.c_str()));
  }
  return out;
}

Opts parseOpts(int argc, char **argv) {
  Opts o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--sizes") {
      o.sizes = parseInts(v);
    } else if (k == "--runs") {
      o.runs = std::atoi(v.c_str());
    } else if (k == "--seconds") {
      o.seconds = std::atof(v.c_str());
    } else if (k == "--sigma") {
      o.sigma = std::atof(v.c_str());
    } else if (k == "--seed") {
      o.seed = std::strtoull(v.c_str(), nullptr, 10);
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (o.sizes.empty() || o.runs < 1 || !(o.seconds > 0.0)) {
    std::fprintf(stderr, "need --sizes, --runs >= 1 and --seconds > 0\n");
    std::exit(2);
  }
  return o;
}

void serial() {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
#endif
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
  ForceInput input() const {
    return {.nAtoms = z.size(),
            .pos = pos.data(),
            .atmnrs = z.data(),
            .box = box.data()};
  }
};

System cluster(std::size_t n, double nn, double vacuum, double sigma,
               std::uint64_t seed) {
  const double a = nn * std::sqrt(2.0);
  int m = 2;
  while (4.0 * m * m * m < 2.0 * static_cast<double>(n) + 64.0) {
    ++m;
  }
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  const double c = 0.5 * (m - 1) * a + 0.25 * a;
  std::vector<std::pair<double, std::array<double, 3>>> sites;
  for (int ix = 0; ix < m; ++ix) {
    for (int iy = 0; iy < m; ++iy) {
      for (int iz = 0; iz < m; ++iz) {
        for (const auto &b : kBasis) {
          const std::array<double, 3> r{(ix + b[0]) * a, (iy + b[1]) * a,
                                        (iz + b[2]) * a};
          const double d = (r[0] - c) * (r[0] - c) + (r[1] - c) * (r[1] - c) +
                           (r[2] - c) * (r[2] - c);
          sites.emplace_back(d, r);
        }
      }
    }
  }
  std::sort(sites.begin(), sites.end(),
            [](const auto &l, const auto &r) { return l.first < r.first; });
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-sigma, sigma);
  System s;
  double lo = 1e300;
  double hi = -1e300;
  for (std::size_t k = 0; k < n; ++k) {
    for (int q = 0; q < 3; ++q) {
      const double x = sites[k].second[q] + u(rng);
      s.pos.push_back(x);
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
  }
  for (double &x : s.pos) {
    x += vacuum - lo;
  }
  const double side = (hi - lo) + 2.0 * vacuum;
  s.box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  s.z.assign(n, 18);
  return s;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Timing {
  double median_ms;
  double min_ms;
  long reps;
};

Timing timeIt(const std::function<void()> &op, int runs, double seconds) {
  using clock = std::chrono::steady_clock;
  op();
  long reps = 1;
  for (;;) {
    const auto t0 = clock::now();
    for (long r = 0; r < reps; ++r) {
      op();
    }
    const double s = std::chrono::duration<double>(clock::now() - t0).count();
    if (s >= seconds || reps >= (1L << 30)) {
      break;
    }
    reps = s > 0.0
               ? std::max(reps + 1, static_cast<long>(reps * 1.2 * seconds / s))
               : 2 * reps;
  }
  std::vector<double> ms;
  for (int k = 0; k < runs; ++k) {
    const auto t0 = clock::now();
    for (long r = 0; r < reps; ++r) {
      op();
    }
    ms.push_back(1e3 *
                 std::chrono::duration<double>(clock::now() - t0).count() /
                 static_cast<double>(reps));
  }
  return {median(ms), *std::min_element(ms.begin(), ms.end()), reps};
}

double maxAbs(const std::vector<double> &a) {
  double m = 0.0;
  for (double x : a) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double relGap(const std::vector<double> &a, const std::vector<double> &ref) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - ref[i]));
  }
  return m / maxAbs(ref);
}

void matvec(const std::vector<double> &h, const std::vector<double> &v,
            std::vector<double> &out) {
  const std::size_t d = v.size();
  for (std::size_t i = 0; i < d; ++i) {
    double acc = 0.0;
    const double *row = h.data() + i * d;
    for (std::size_t j = 0; j < d; ++j) {
      acc += row[j] * v[j];
    }
    out[i] = acc;
  }
}

void row(int n, const char *what, const char *scheme, const Timing &t,
         double err, const char *note = "") {
  std::printf("| %4d | %-7s | %-13s | %12.6f | %12.6f | %9ld | %10.3e |%s\n", n,
              what, scheme, t.median_ms, t.min_ms, t.reps, err, note);
}

} // namespace

int main(int argc, char **argv) {
  serial();
  const Opts o = parseOpts(argc, argv);
  const rgpot::LJConfig cfg{};
  LJPot pot(cfg);
  const double nn = std::pow(2.0, 1.0 / 6.0) * cfg.psi;

  std::printf(
      "# LJPot u0=%g psi=%g cutoff=%g switch_width=%g; runs=%d seconds=%g "
      "sigma=%g seed=%llu\n",
      cfg.u0, cfg.psi, cfg.cutoff, cfg.switch_width, o.runs, o.seconds, o.sigma,
      static_cast<unsigned long long>(o.seed));
  std::printf("# default step %g\n", rgpot::kHvpDefaultStep);
  std::printf("|    N | what    | scheme        |  median (ms) |     min (ms) "
              "|      reps "
              "|  rel. err. |\n");
  std::printf("|------+---------+---------------+--------------+--------------+"
              "-----------"
              "+------------|\n");

  for (const int n : o.sizes) {
    const System s = cluster(static_cast<std::size_t>(n), nn, 1.1 * cfg.cutoff,
                             o.sigma, o.seed + static_cast<std::uint64_t>(n));
    const ForceInput in = s.input();
    const std::size_t dim = s.pos.size();
    HessianVectorCache cache;

    std::vector<double> ha(dim * dim);
    if (cache.hessian(pot, in, ha.data(), HessianScheme::Analytic) !=
        HessianScheme::Analytic) {
      std::fprintf(stderr, "analytic Hessian refused\n");
      return 1;
    }
    std::vector<double> v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      v[i] = std::sin(0.37 * static_cast<double>(i) + 0.1);
    }
    std::vector<double> hv_ref(dim);
    matvec(ha, v, hv_ref);

    std::vector<double> h(dim * dim);
    std::vector<double> hv(dim);

    double t_dense[3] = {};
    const std::pair<const char *, HessianScheme> dense[] = {
        {"analytic", HessianScheme::Analytic},
        {"ad", HessianScheme::Ad},
        {"central", HessianScheme::Central}};
    for (int k = 0; k < 3; ++k) {
      const HessianScheme want = dense[k].second;
      const Timing t = timeIt(
          [&] {
            if (cache.hessian(pot, in, h.data(), want) != want) {
              std::fprintf(stderr, "Hessian scheme %s not taken\n",
                           dense[k].first);
              std::exit(1);
            }
          },
          o.runs, o.seconds);
      t_dense[k] = t.median_ms;
      row(n, "hessian", dense[k].first, t, relGap(h, ha));
    }

    auto hvp = [&](HvpScheme want, bool fresh) {
      return [&cache, &pot, &in, &v, &hv, want, fresh] {
        if (fresh) {
          cache.invalidate();
        }
        if (cache.apply(pot, in, v.data(), hv.data(), 0.0, want) != want) {
          std::fprintf(stderr, "product scheme not taken\n");
          std::exit(1);
        }
      };
    };
    const Timing t_an =
        timeIt(hvp(HvpScheme::Analytic, false), o.runs, o.seconds);
    row(n, "hvp", "analytic", t_an, relGap(hv, hv_ref));
    const Timing t_ad = timeIt(hvp(HvpScheme::Ad, false), o.runs, o.seconds);
    row(n, "hvp", "ad", t_ad, relGap(hv, hv_ref));
    const Timing t_ce =
        timeIt(hvp(HvpScheme::Central, false), o.runs, o.seconds);
    row(n, "hvp", "central", t_ce, relGap(hv, hv_ref));
    const Timing t_ff =
        timeIt(hvp(HvpScheme::Forward, true), o.runs, o.seconds);
    row(n, "hvp", "forward-first", t_ff, relGap(hv, hv_ref));
    cache.invalidate();
    const Timing t_fr =
        timeIt(hvp(HvpScheme::Forward, false), o.runs, o.seconds);
    row(n, "hvp", "forward-reuse", t_fr, relGap(hv, hv_ref));
    const Timing t_mv = timeIt([&] { matvec(ha, v, hv); }, o.runs, o.seconds);
    row(n, "matvec", "stored", t_mv, relGap(hv, hv_ref));

    const double k_an = (t_dense[0] - t_an.median_ms) / t_an.median_ms + 1.0;
    const double k_ad = (t_dense[1] - t_ad.median_ms) / t_ad.median_ms + 1.0;
    const double k_fd = (t_dense[2] - t_ff.median_ms) / t_fr.median_ms + 1.0;
    std::printf("# N=%d k*: analytic %.2f, ad %.2f, finite differences %.1f "
                "(6N-1 = %d)\n",
                n, k_an, k_ad, k_fd, 6 * n - 1);
  }
  return 0;
}
