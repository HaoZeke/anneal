// MIT License
// Copyright 2023--present rgpot developers
//
// Smoke timing for MorsePot across OMP_NUM_THREADS = 1, 2, 4, 8 on:
//   1. 6x6x6 FCC Cu vacancy (863 atoms, Makri/Ortner/Kermode NEB parameters:
//      De = 1.0, a = 4.0 / 2.55, re = 2.55, cutoff = 5.3, below kTeamMinAtoms)
//   2. 14x14x14 FCC Cu vacancy (10975 atoms, same parameters, stored list)
//   3. 10000-atom Pt cluster in a vacuum-padded cell (default MorseConfig,
//      stored list)
//
// Cycles across perturbed Verlet-reused frames and checks bit-for-bit
// agreement of energies and forces against the 1-thread reference.
//
// morse_omp_bench [--threads 1,2,4,8] [--runs 5] [--seconds 0.15]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Slices.hpp"

namespace {

struct Opts {
  std::vector<int> threads{1, 2, 4, 8};
  int runs = 5;
  double seconds = 0.15;
  int frames = 16;
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
    if (k == "--threads") {
      o.threads = parseInts(v);
    } else if (k == "--runs") {
      o.runs = std::atoi(v.c_str());
    } else if (k == "--seconds") {
      o.seconds = std::atof(v.c_str());
    } else if (k == "--frames") {
      o.frames = std::atoi(v.c_str());
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
  }
  return o;
}

void setThreads(int n) {
  const std::string s = std::to_string(n);
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", s.c_str());
#else
  setenv("OMP_NUM_THREADS", s.c_str(), 1);
#endif
}

struct Case {
  std::string name;
  rgpot::MorseConfig cfg;
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

Case makeCuVacancy(const char *label, int side) {
  Case c;
  c.name = label;
  c.cfg = {.De = 1.0, .a = 4.0 / 2.55, .re = 2.55, .cutoff = 5.3, .switch_width = 0.0};
  const double a_lat = 2.55 * std::sqrt(2.0);
  const double L = static_cast<double>(side) * a_lat;
  c.box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  bool dropped = false;
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &b : kBasis) {
          if (!dropped) {
            dropped = true;
            continue;
          }
          c.pos.push_back((static_cast<double>(ix) + b[0]) * a_lat);
          c.pos.push_back((static_cast<double>(iy) + b[1]) * a_lat);
          c.pos.push_back((static_cast<double>(iz) + b[2]) * a_lat);
          c.z.push_back(29);
        }
      }
    }
  }
  return c;
}

Case makePtCluster(const char *label, int n) {
  Case c;
  c.name = label;
  c.cfg = rgpot::MorseConfig{};
  const double a_lat = c.cfg.re * std::sqrt(2.0);
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  int k = 1;
  while (std::pow(2.0 * k + 1.0, 3.0) * 4.0 < 4.0 * static_cast<double>(n)) {
    ++k;
  }
  struct Site {
    double r2, x, y, z;
  };
  std::vector<Site> sites;
  for (int ix = -k; ix <= k; ++ix) {
    for (int iy = -k; iy <= k; ++iy) {
      for (int iz = -k; iz <= k; ++iz) {
        for (const auto &b : kBasis) {
          const double x = (ix + b[0]) * a_lat;
          const double y = (iy + b[1]) * a_lat;
          const double z = (iz + b[2]) * a_lat;
          sites.push_back({x * x + y * y + z * z, x, y, z});
        }
      }
    }
  }
  std::stable_sort(sites.begin(), sites.end(),
                   [](const Site &l, const Site &r) { return l.r2 < r.r2; });
  double extent = 0.0;
  for (int s = 0; s < n; ++s) {
    c.pos.push_back(sites[static_cast<std::size_t>(s)].x);
    c.pos.push_back(sites[static_cast<std::size_t>(s)].y);
    c.pos.push_back(sites[static_cast<std::size_t>(s)].z);
    c.z.push_back(78);
  }
  for (double x : c.pos) {
    extent = std::max(extent, std::abs(x));
  }
  const double side = 2.0 * (extent + 2.0 * c.cfg.cutoff) + 1.0;
  for (double &x : c.pos) {
    x += 0.5 * side;
  }
  c.box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  return c;
}

std::uint64_t bitsOf(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

} // namespace

int main(int argc, char **argv) {
  const Opts o = parseOpts(argc, argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  setThreads(2);
  std::printf("openmp: %s runs: %d seconds: %g frames: %d\n",
              rgpot::threadsRequested() == 2 ? "yes" : "no", o.runs, o.seconds, o.frames);

  const std::vector<Case> cases{
      makeCuVacancy("cu_vac_6x6x6", 6),
      makeCuVacancy("cu_vac_14x14x14", 14),
      makePtCluster("pt_cluster_10000", 10000),
  };

  for (const Case &c : cases) {
    const std::size_t n = c.z.size();
    std::mt19937_64 gen(o.seed);
    std::normal_distribution<double> noise(0.0, o.sigma);
    std::vector<std::vector<double>> frames(static_cast<std::size_t>(o.frames), c.pos);
    for (auto &f : frames) {
      for (double &x : f) {
        x += noise(gen);
      }
    }

    std::vector<double> ref_e;
    std::vector<double> ref_f;
    double t1_median = 0.0;

    for (std::size_t q = 0; q < o.threads.size(); ++q) {
      const int th = o.threads[q];
      setThreads(th);
      rgpot::MorsePot pot(c.cfg);
      pot.setNeighbourMode(rgpot::nlist::Mode::Stored);

      std::vector<double> energy(frames.size(), 0.0);
      std::vector<double> forces(frames.size() * 3 * n, 0.0);
      for (std::size_t k = 0; k < frames.size(); ++k) {
        rgpot::ForceOut out{};
        out.F = forces.data() + k * 3 * n;
        const rgpot::ForceInput in{n, frames[k].data(), c.z.data(), c.box.data()};
        pot.forceImpl(in, &out);
        energy[k] = out.energy;
      }
      if (q == 0) {
        ref_e = energy;
        ref_f = forces;
      }

      std::size_t bits_apart = 0;
      for (std::size_t k = 0; k < energy.size(); ++k) {
        bits_apart += bitsOf(energy[k]) == bitsOf(ref_e[k]) ? 0 : 1;
      }
      for (std::size_t k = 0; k < forces.size(); ++k) {
        bits_apart += bitsOf(forces[k]) == bitsOf(ref_f[k]) ? 0 : 1;
      }
      if (bits_apart != 0) {
        std::fprintf(stderr, "bit mismatch on %s threads=%d apart=%zu\n", c.name.c_str(), th,
                     bits_apart);
        return 1;
      }

      std::vector<double> fbuf(3 * n, 0.0);
      std::vector<double> us;
      double sink = 0.0;
      std::size_t total_calls = 0;
      for (int r = 0; r < o.runs; ++r) {
        std::size_t done = 0;
        const auto t0 = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        while (done < 2 || elapsed < o.seconds) {
          rgpot::ForceOut out{};
          out.F = fbuf.data();
          const rgpot::ForceInput in{
              n, frames[done % frames.size()].data(), c.z.data(), c.box.data()};
          pot.forceImpl(in, &out);
          sink += out.energy;
          ++done;
          elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        us.push_back(1e6 * elapsed / static_cast<double>(done));
        total_calls += done;
      }
      if (!std::isfinite(sink)) {
        return 1;
      }
      const double med = median(us);
      if (q == 0) {
        t1_median = med;
      }
      const double speedup = t1_median > 0.0 ? t1_median / med : 1.0;
      std::printf("smoke case=%s n_atoms=%zu threads=%d median_us=%.2f min_us=%.2f "
                  "speedup=%.2fx calls=%zu reused=%d bits_apart=%zu\n",
                  c.name.c_str(), n, th, med, *std::min_element(us.begin(), us.end()), speedup,
                  total_calls, pot.listReused() ? 1 : 0, bits_apart);
    }
  }
  return 0;
}
