// MIT License
// Copyright 2023--present rgpot developers
//
// Microseconds per Sutton-Chen call on a free cluster with no cutoff, the
// way a quench calls it. Five modes evaluate the same frames:
//
//   entry   rgpot_sutton_chen_energy_forces, the C entry point
//   fresh   a SuttonChenPot built for each call
//   kept    one SuttonChenPot kept across calls, default neighbour mode
//   stream  the same, in nlist::Mode::Stream
//   plain   a half-pair loop that keeps (a/r)^m, (a/r)^n and 1/r^2 from
//           the density pass for the force pass
//
// Each call is the cluster plus Gaussian noise of --sigma per coordinate,
// cycling over 64 frames. The clusters are fcc cuboctahedra at the bulk
// nearest-neighbour distance (13, 55, 147, 309, 561, 923 atoms) and the
// deposited Pt38 minimum (38). A check line per mode gives the largest
// energy and force gaps to the plain loop over the 64 frames and a hash of
// the energy and force bits, which two builds can compare.
//
// OMP_NUM_THREADS is set to 1. Pin the process, as with taskset -c 0.
//
// sc_call_bench [--sizes 13,38,55,147,561] [--family 12,6,144.41]...
//               [--runs 3] [--work 2e7] [--sigma 0.02] [--seed 7]
//               [--pt38 CppCore/tests/data/sc108/pt38]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

constexpr int kFrames = 64;

struct Family {
  int n = 12;
  int m = 6;
  double c = 144.41;
};

struct Opts {
  std::vector<int> sizes{13, 38, 55, 147, 561};
  std::vector<Family> families{{12, 6, 144.41}, {9, 6, 39.432}};
  int runs = 3;
  double work = 2e7;
  double sigma = 0.02;
  std::uint64_t seed = 7;
  std::string pt38 = "CppCore/tests/data/sc108/pt38";
};

std::vector<std::string> split(const std::string &text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    out.push_back(tok);
  }
  return out;
}

Opts parse(int argc, char **argv) {
  Opts o;
  bool own_family = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--sizes") {
      o.sizes.clear();
      for (const auto &s : split(v)) {
        o.sizes.push_back(std::atoi(s.c_str()));
      }
    } else if (k == "--family") {
      const auto f = split(v);
      if (f.size() != 3) {
        std::fprintf(stderr, "--family takes n,m,c\n");
        std::exit(2);
      }
      if (!own_family) {
        o.families.clear();
        own_family = true;
      }
      o.families.push_back({std::atoi(f[0].c_str()), std::atoi(f[1].c_str()),
                            std::atof(f[2].c_str())});
    } else if (k == "--runs") {
      o.runs = std::atoi(v.c_str());
    } else if (k == "--work") {
      o.work = std::atof(v.c_str());
    } else if (k == "--sigma") {
      o.sigma = std::atof(v.c_str());
    } else if (k == "--seed") {
      o.seed =
          static_cast<std::uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
    } else if (k == "--pt38") {
      o.pt38 = v;
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
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

/// fcc sites (i, j, k) * nn / sqrt(2) with i + j + k even inside the
/// cuboctahedron |i|, |j|, |k| <= shells, |i| + |j| + |k| <= 2 shells.
std::vector<double> cuboctahedron(int shells, double nn) {
  const double h = nn / std::sqrt(2.0);
  std::vector<double> pos;
  for (int i = -shells; i <= shells; ++i) {
    for (int j = -shells; j <= shells; ++j) {
      for (int k = -shells; k <= shells; ++k) {
        if ((i + j + k) % 2 != 0 ||
            std::abs(i) + std::abs(j) + std::abs(k) > 2 * shells) {
          continue;
        }
        pos.push_back(i * h);
        pos.push_back(j * h);
        pos.push_back(k * h);
      }
    }
  }
  return pos;
}

/// Coordinates of a deposited point file: a parameter line, then x y z.
std::vector<double> pointFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  double head[5];
  for (double &h : head) {
    in >> h;
  }
  std::vector<double> pos;
  double x = 0.0;
  while (in >> x) {
    pos.push_back(x);
  }
  return pos;
}

std::vector<double> cluster(int size, double nn, const Opts &o) {
  if (size == 38) {
    return pointFile(o.pt38);
  }
  for (int shells = 1; shells <= 8; ++shells) {
    const int count = (10 * shells * shells * shells + 15 * shells * shells +
                       11 * shells + 3) /
                      3;
    if (count == size) {
      auto pos = cuboctahedron(shells, nn);
      if (pos.size() != static_cast<std::size_t>(3 * size)) {
        throw std::runtime_error("cuboctahedron count");
      }
      return pos;
    }
  }
  throw std::runtime_error("no cluster of " + std::to_string(size) + " atoms");
}

std::vector<std::vector<double>> frames(const std::vector<double> &base,
                                        const Opts &o) {
  std::mt19937_64 gen(o.seed);
  std::normal_distribution<double> noise(0.0, o.sigma);
  std::vector<std::vector<double>> out(kFrames, base);
  for (auto &f : out) {
    for (double &x : f) {
      x += noise(gen);
    }
  }
  return out;
}

double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

double powAr(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

/// The reference: one pass for the densities that keeps the powers, one
/// pass for the forces that reads them. With a = 1 the operations are
/// those of SuttonChenPot, so the energy and forces carry the same bits.
struct Plain {
  std::vector<double> terms;
  std::vector<double> weight;

  void operator()(const rgpot::SuttonChenConfig &cfg, const double *x,
                  std::size_t n, double *energy, double *forces) {
    const double a2 = cfg.a * cfg.a;
    weight.assign(n, 0.0);
    terms.resize(3 * (n * (n - 1) / 2));
    double epair = 0.0;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double dx = x[3 * i] - x[3 * j];
        const double dy = x[3 * i + 1] - x[3 * j + 1];
        const double dz = x[3 * i + 2] - x[3 * j + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 > 0.0)) {
          continue;
        }
        const double inv = 1.0 / r2;
        const double ar2 = a2 * inv;
        const double phi = powAr(ar2, cfg.m);
        const double rep = powAr(ar2, cfg.n);
        terms[k] = phi;
        terms[k + 1] = rep;
        terms[k + 2] = inv;
        k += 3;
        weight[i] += phi;
        weight[j] += phi;
        epair += cfg.epsilon * rep;
      }
    }
    double eemb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!(weight[i] > 0.0)) {
        weight[i] = 0.0;
        continue;
      }
      const double root = std::sqrt(weight[i]);
      eemb -= cfg.epsilon * cfg.c * root;
      weight[i] = cfg.epsilon * cfg.c / (2.0 * root);
    }
    std::fill(forces, forces + 3 * n, 0.0);
    k = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double dx = x[3 * i] - x[3 * j];
        const double dy = x[3 * i + 1] - x[3 * j + 1];
        const double dz = x[3 * i + 2] - x[3 * j + 2];
        if (!(dx * dx + dy * dy + dz * dz > 0.0)) {
          continue;
        }
        const double phi = terms[k];
        const double rep = terms[k + 1];
        const double inv = terms[k + 2];
        k += 3;
        const double s =
            (cfg.epsilon * static_cast<double>(cfg.n) * rep -
             static_cast<double>(cfg.m) * phi * (weight[i] + weight[j])) *
            inv;
        forces[3 * i] += s * dx;
        forces[3 * i + 1] += s * dy;
        forces[3 * i + 2] += s * dz;
        forces[3 * j] -= s * dx;
        forces[3 * j + 1] -= s * dy;
        forces[3 * j + 2] -= s * dz;
      }
    }
    *energy = epair + eemb;
  }
};

void evalPot(const rgpot::SuttonChenPot &pot, const double *x, std::size_t n,
             double *energy, double *forces) {
  rgpot::ForceOut out{};
  out.F = forces;
  const rgpot::ForceInput in{n, x, nullptr, nullptr};
  pot.forceImpl(in, &out);
  *energy = out.energy;
}

using Call = std::function<void(const double *, double *, double *)>;

struct Mode {
  const char *name;
  Call call;
};

std::uint64_t fnv(std::uint64_t h, const void *data, std::size_t len) {
  const auto *b = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < len; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

} // namespace

int main(int argc, char **argv) {
  const Opts o = parse(argc, argv);
  serial();
#if defined(__VERSION__)
  std::printf("compiler: %s\n", __VERSION__);
#endif
  std::printf("sigma: %g seed: %llu frames: %d runs: %d work: %g threads: %d\n",
              o.sigma, static_cast<unsigned long long>(o.seed), kFrames, o.runs,
              o.work, rgpot::nlist::threadsRequested());
  for (const Family &fam : o.families) {
    const rgpot_sc_config raw{1.0, fam.c, 1.0, fam.n, fam.m, 0.0, 0};
    rgpot::SuttonChenConfig cfg;
    cfg.epsilon = raw.epsilon;
    cfg.c = raw.c;
    cfg.a = raw.a;
    cfg.n = raw.n;
    cfg.m = raw.m;
    cfg.cutoff = raw.cutoff;
    const double nn = rgpot::sutton_chen_scales(cfg).nearest_neighbor;
    const std::string label =
        std::to_string(fam.n) + "-" + std::to_string(fam.m);
    std::printf("family: %s c: %g nearest_neighbour: %.6f\n", label.c_str(),
                fam.c, nn);
    for (int size : o.sizes) {
      const auto fr = frames(cluster(size, nn, o), o);
      const std::size_t n = fr[0].size() / 3;
      const auto natoms = static_cast<std::int32_t>(n);
      const rgpot::SuttonChenPot kept(cfg);
      rgpot::SuttonChenPot stream(cfg);
      stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
      Plain plain;
      const std::vector<Mode> modes{
          {"entry",
           [&](const double *x, double *e, double *f) {
             if (rgpot_sutton_chen_energy_forces(&raw, natoms, x, nullptr, e,
                                                 f) != 0) {
               throw std::runtime_error(
                   "rgpot_sutton_chen_energy_forces failed");
             }
           }},
          {"fresh",
           [&](const double *x, double *e, double *f) {
             const rgpot::SuttonChenPot pot(cfg);
             evalPot(pot, x, n, e, f);
           }},
          {"kept", [&](const double *x, double *e,
                       double *f) { evalPot(kept, x, n, e, f); }},
          {"stream", [&](const double *x, double *e,
                         double *f) { evalPot(stream, x, n, e, f); }},
          {"plain", [&](const double *x, double *e,
                        double *f) { plain(cfg, x, n, e, f); }},
      };

      std::vector<double> ref_e(kFrames);
      std::vector<double> ref_f(static_cast<std::size_t>(kFrames) * 3 * n);
      for (int k = 0; k < kFrames; ++k) {
        plain(cfg, fr[k].data(), n, &ref_e[k],
              ref_f.data() + static_cast<std::size_t>(k) * 3 * n);
      }
      std::vector<double> force(3 * n);
      for (const Mode &mode : modes) {
        double max_de = 0.0;
        double max_df = 0.0;
        std::uint64_t bits = 1469598103934665603ULL;
        for (int k = 0; k < kFrames; ++k) {
          double e = 0.0;
          mode.call(fr[k].data(), &e, force.data());
          bits = fnv(bits, &e, sizeof(e));
          bits = fnv(bits, force.data(), force.size() * sizeof(double));
          max_de = std::max(max_de, std::fabs(e - ref_e[k]));
          const double *rf = ref_f.data() + static_cast<std::size_t>(k) * 3 * n;
          for (std::size_t q = 0; q < 3 * n; ++q) {
            max_df = std::max(max_df, std::fabs(force[q] - rf[q]));
          }
        }
        std::printf("check family=%s n_atoms=%zu mode=%s max_abs_dE=%.3g "
                    "max_abs_dF=%.3g bits=%016llx\n",
                    label.c_str(), n, mode.name, max_de, max_df,
                    static_cast<unsigned long long>(bits));
      }

      const double pairs =
          0.5 * static_cast<double>(n) * static_cast<double>(n - 1);
      const int calls = static_cast<int>(
          std::clamp(o.work / pairs, static_cast<double>(kFrames), 1e5));
      std::vector<std::vector<double>> us(modes.size());
      double sink = 0.0;
      for (int run = 0; run < o.runs; ++run) {
        for (std::size_t q = 0; q < modes.size(); ++q) {
          double e = 0.0;
          for (int k = 0; k < kFrames; ++k) {
            modes[q].call(fr[k].data(), &e, force.data());
          }
          const auto t0 = std::chrono::steady_clock::now();
          for (int k = 0; k < calls; ++k) {
            modes[q].call(fr[static_cast<std::size_t>(k % kFrames)].data(), &e,
                          force.data());
            sink += e;
          }
          const auto t1 = std::chrono::steady_clock::now();
          us[q].push_back(
              std::chrono::duration<double, std::micro>(t1 - t0).count() /
              calls);
        }
      }
      for (std::size_t q = 0; q < modes.size(); ++q) {
        std::printf("result family=%s n_atoms=%zu mode=%s runs=%d calls=%d "
                    "median_us_per_call=%.3f min=%.3f max=%.3f all=",
                    label.c_str(), n, modes[q].name, o.runs, calls,
                    median(us[q]),
                    *std::min_element(us[q].begin(), us[q].end()),
                    *std::max_element(us[q].begin(), us[q].end()));
        for (std::size_t r = 0; r < us[q].size(); ++r) {
          std::printf("%s%.3f", r ? "," : "", us[q][r]);
        }
        std::printf("\n");
      }
      if (!std::isfinite(sink)) {
        std::printf("non-finite energy\n");
        return 1;
      }
      std::fflush(stdout);
    }
  }
  return 0;
}
