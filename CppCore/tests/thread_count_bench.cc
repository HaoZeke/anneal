// MIT License
// Copyright 2023--present rgpot developers
//
// Microseconds per call of each classical potential against
// OMP_NUM_THREADS, and whether the result bits move with it. Each
// potential takes a free cluster: the `n` sites of its lattice nearest a
// lattice site. The Lennard-Jones cluster, ZBL and Sutton-Chen take no
// cell; the others put the cluster in the middle of a cell wide enough
// that no periodic image reaches it. Sutton-Chen has no cutoff, as a
// global optimisation calls it.
//
// A call is the cluster plus Gaussian noise of --sigma per coordinate,
// cycling over --frames frames, on one potential object kept across the
// calls. A line per potential, size and thread count gives the median
// microseconds per call of --runs runs of at least --seconds each, and,
// against the first thread count listed, the frames whose energy bits
// differ, the largest energy gap and the force components whose bits
// differ. `bits` hashes the energy and force bits of every frame, so two
// builds can be compared. --mode sets the neighbour mode of every
// potential; Sutton-Chen walks every pair in any mode.
//
// thread_count_bench [--pots sc,lj,ljcluster,morse,zbl,eam,sw,tersoff]
//                    [--sizes 38,55,80,500,4000,10000] [--threads 1,8]
//                    [--runs 5] [--seconds 0.2] [--frames 16]
//                    [--sigma 0.02] [--seed 7] [--mode auto|stored|stream]
//
// Pin the process to as many cores as the largest thread count.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

struct Opts {
  std::vector<std::string> pots{"sc", "lj", "ljcluster", "morse", "zbl", "eam", "sw", "tersoff"};
  std::vector<int> sizes{38, 55, 80, 500, 4000, 10000};
  std::vector<int> threads{1, 8};
  int runs = 5;
  double seconds = 0.2;
  int frames = 16;
  double sigma = 0.02;
  std::uint64_t seed = 7;
  std::string mode = "auto";
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

std::vector<int> integers(const std::string &text) {
  std::vector<int> out;
  for (const auto &s : split(text)) {
    out.push_back(std::atoi(s.c_str()));
  }
  return out;
}

Opts parse(int argc, char **argv) {
  Opts o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--pots") {
      o.pots = split(v);
    } else if (k == "--sizes") {
      o.sizes = integers(v);
    } else if (k == "--threads") {
      o.threads = integers(v);
    } else if (k == "--runs") {
      o.runs = std::atoi(v.c_str());
    } else if (k == "--seconds") {
      o.seconds = std::atof(v.c_str());
    } else if (k == "--frames") {
      o.frames = std::atoi(v.c_str());
    } else if (k == "--sigma") {
      o.sigma = std::atof(v.c_str());
    } else if (k == "--seed") {
      o.seed = static_cast<std::uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
    } else if (k == "--mode") {
      o.mode = v;
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (o.runs < 1 || o.frames < 1 || o.threads.empty()) {
    std::fprintf(stderr, "--runs, --frames and --threads need at least one\n");
    std::exit(2);
  }
  if (o.mode != "auto" && o.mode != "stored" && o.mode != "stream") {
    std::fprintf(stderr, "--mode is auto, stored or stream\n");
    std::exit(2);
  }
  return o;
}

rgpot::nlist::Mode modeOf(const std::string &name) {
  if (name == "stored") {
    return rgpot::nlist::Mode::Stored;
  }
  return name == "stream" ? rgpot::nlist::Mode::Stream : rgpot::nlist::Mode::Auto;
}

void setThreads(int n) {
  const std::string value = std::to_string(n);
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value.c_str());
#else
  setenv("OMP_NUM_THREADS", value.c_str(), 1);
#endif
}

using Basis = std::vector<std::array<double, 3>>;

const Basis kFcc{{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
const Basis kDiamond{{0, 0, 0},          {0.5, 0.5, 0},      {0.5, 0, 0.5},
                     {0, 0.5, 0.5},      {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                     {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};

/// The `n` sites of a cubic lattice of constant `a` nearest the site at
/// the origin, by distance and then by the order they were made in.
std::vector<double> nearestSites(const Basis &basis, double a, int n) {
  int k = 1;
  while (std::pow(2.0 * k + 1.0, 3.0) * static_cast<double>(basis.size()) <
         4.0 * static_cast<double>(n)) {
    ++k;
  }
  struct Site {
    double r2;
    double x;
    double y;
    double z;
  };
  std::vector<Site> sites;
  for (int ix = -k; ix <= k; ++ix) {
    for (int iy = -k; iy <= k; ++iy) {
      for (int iz = -k; iz <= k; ++iz) {
        for (const auto &p : basis) {
          const double x = (ix + p[0]) * a;
          const double y = (iy + p[1]) * a;
          const double z = (iz + p[2]) * a;
          sites.push_back({x * x + y * y + z * z, x, y, z});
        }
      }
    }
  }
  std::stable_sort(sites.begin(), sites.end(),
                   [](const Site &l, const Site &r) { return l.r2 < r.r2; });
  std::vector<double> pos;
  for (int s = 0; s < n; ++s) {
    pos.push_back(sites[static_cast<std::size_t>(s)].x);
    pos.push_back(sites[static_cast<std::size_t>(s)].y);
    pos.push_back(sites[static_cast<std::size_t>(s)].z);
  }
  return pos;
}

/// A cubic cell around the cluster with `reach` of vacuum on every side.
/// The cluster moves to the middle of the cell: the bins of the neighbour
/// layer cover the cell, and put every atom outside it in an edge bin.
std::array<double, 9> cellAround(std::vector<double> &pos, double reach) {
  double extent = 0.0;
  for (const double x : pos) {
    extent = std::max(extent, std::abs(x));
  }
  const double side = 2.0 * (extent + reach) + 1.0;
  for (double &x : pos) {
    x += 0.5 * side;
  }
  return {side, 0, 0, 0, side, 0, 0, 0, side};
}

using Call = std::function<void(const double *, double *, double *)>;

struct Case;

/// A potential kept across the calls of one thread count. The call reads
/// the species and cell of the case, which outlives it.
using Make = std::function<Call(const Case &)>;

struct Case {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
  bool boxed = false;
  rgpot::nlist::Mode mode = rgpot::nlist::Mode::Auto;
  Make make;
};

template <typename Pot, typename... Args>
Make keep(Args... args) {
  return [args...](const Case &c) -> Call {
    auto pot = std::make_shared<Pot>(args...);
    if (c.mode != rgpot::nlist::Mode::Auto) {
      pot->setNeighbourMode(c.mode);
    }
    const std::size_t n = c.z.size();
    const int *z = c.z.data();
    const double *box = c.boxed ? c.box.data() : nullptr;
    return [pot, n, z, box](const double *x, double *energy, double *forces) {
      std::fill(forces, forces + 3 * n, 0.0);
      rgpot::ForceOut out{};
      out.F = forces;
      const rgpot::ForceInput in{n, x, z, box};
      pot->forceImpl(in, &out);
      *energy = out.energy;
    };
  };
}

Case makeCase(const std::string &name, int n, rgpot::nlist::Mode mode) {
  Case c;
  c.mode = mode;
  const auto atoms = static_cast<std::size_t>(n);
  if (name == "sc") {
    rgpot::SuttonChenConfig cfg;
    cfg.cutoff = 0.0;
    const double nn = rgpot::sutton_chen_scales(cfg).nearest_neighbor;
    c.pos = nearestSites(kFcc, nn * std::sqrt(2.0), n);
    c.z.assign(atoms, 78);
    c.make = keep<rgpot::SuttonChenPot>(cfg);
  } else if (name == "lj" || name == "ljcluster") {
    c.pos = nearestSites(kFcc, std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0), n);
    c.z.assign(atoms, 18);
    if (name == "lj") {
      rgpot::LJConfig cfg;
      cfg.cutoff = 2.5;
      c.box = cellAround(c.pos, 2.0 * (cfg.cutoff + 1.0));
      c.boxed = true;
      c.make = keep<rgpot::LJPot>(cfg);
    } else {
      rgpot::LJClusterConfig cfg;
      cfg.cutoff = 2.5;
      c.make = keep<rgpot::LJClusterPot>(cfg);
    }
  } else if (name == "morse") {
    const rgpot::MorseConfig cfg;
    c.pos = nearestSites(kFcc, cfg.re * std::sqrt(2.0), n);
    c.z.assign(atoms, 78);
    c.box = cellAround(c.pos, 2.0 * cfg.cutoff);
    c.boxed = true;
    c.make = keep<rgpot::MorsePot>(cfg);
  } else if (name == "zbl") {
    c.pos = nearestSites(kDiamond, 5.431, n);
    c.z.assign(atoms, 14);
    c.make = keep<rgpot::ZBLPot>();
  } else if (name == "eam") {
    c.pos = nearestSites(kFcc, 4.05, n);
    c.z.assign(atoms, 13);
    c.box = cellAround(c.pos, 15.0);
    c.boxed = true;
    c.make = keep<rgpot::EAMPot>();
  } else if (name == "sw" || name == "tersoff") {
    c.pos = nearestSites(kDiamond, 5.431, n);
    c.z.assign(atoms, 14);
    c.box = cellAround(c.pos, 10.0);
    c.boxed = true;
    c.make = name == "sw" ? keep<rgpot::StillingerWeberSi>() : keep<rgpot::TersoffSiC>();
  } else {
    throw std::invalid_argument("unknown potential " + name);
  }
  return c;
}

std::uint64_t bitsOf(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

// FNV-1a over the bytes of the energies, then the forces.
std::uint64_t hashBits(const std::vector<double> &energy, const std::vector<double> &forces) {
  std::uint64_t h = 14695981039346656037ULL;
  auto mix = [&h](double v) {
    const std::uint64_t b = bitsOf(v);
    for (int s = 0; s < 64; s += 8) {
      h ^= (b >> s) & 0xffU;
      h *= 1099511628211ULL;
    }
  };
  for (const double v : energy) {
    mix(v);
  }
  for (const double v : forces) {
    mix(v);
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
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
#if defined(__VERSION__)
  std::printf("compiler: %s\n", __VERSION__);
#endif
  setThreads(2);
  std::printf("openmp: %s mode: %s frames: %d runs: %d seconds: %g sigma: %g seed: %llu\n",
              rgpot::threadsRequested() == 2 ? "yes" : "no", o.mode.c_str(), o.frames, o.runs,
              o.seconds, o.sigma, static_cast<unsigned long long>(o.seed));
  for (const std::string &name : o.pots) {
    for (const int size : o.sizes) {
      const Case c = makeCase(name, size, modeOf(o.mode));
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
      for (std::size_t q = 0; q < o.threads.size(); ++q) {
        setThreads(o.threads[q]);
        const Call call = c.make(c);
        std::vector<double> energy(frames.size());
        std::vector<double> forces(frames.size() * 3 * n);
        for (std::size_t k = 0; k < frames.size(); ++k) {
          call(frames[k].data(), &energy[k], forces.data() + k * 3 * n);
        }
        if (q == 0) {
          ref_e = energy;
          ref_f = forces;
        }
        std::size_t e_apart = 0;
        double max_de = 0.0;
        for (std::size_t k = 0; k < frames.size(); ++k) {
          e_apart += bitsOf(energy[k]) == bitsOf(ref_e[k]) ? 0 : 1;
          max_de = std::max(max_de, std::abs(energy[k] - ref_e[k]));
        }
        std::size_t f_apart = 0;
        for (std::size_t k = 0; k < forces.size(); ++k) {
          f_apart += bitsOf(forces[k]) == bitsOf(ref_f[k]) ? 0 : 1;
        }
        std::vector<double> force(3 * n);
        double sink = 0.0;
        std::vector<double> us;
        std::size_t calls = 0;
        for (int run = 0; run < o.runs; ++run) {
          std::size_t done = 0;
          const auto t0 = std::chrono::steady_clock::now();
          double elapsed = 0.0;
          while (done < 2 || elapsed < o.seconds) {
            double e = 0.0;
            call(frames[done % frames.size()].data(), &e, force.data());
            sink += e;
            ++done;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          }
          us.push_back(1e6 * elapsed / static_cast<double>(done));
          calls += done;
        }
        if (!std::isfinite(sink)) {
          std::printf("non-finite energy for %s at %zu atoms\n", name.c_str(), n);
          return 1;
        }
        std::printf("result pot=%s n_atoms=%zu threads=%d median_us_per_call=%.3f min=%.3f "
                    "calls=%zu energy_bits_apart=%zu/%zu max_abs_dE=%.3g "
                    "force_bits_apart=%zu/%zu bits=%016llx\n",
                    name.c_str(), n, o.threads[q], median(us),
                    *std::min_element(us.begin(), us.end()), calls, e_apart, frames.size(),
                    max_de, f_apart, forces.size(),
                    static_cast<unsigned long long>(hashBits(energy, forces)));
      }
    }
  }
  return 0;
}
