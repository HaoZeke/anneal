// MIT License
// Copyright 2023--present rgpot developers

// Strong and weak OpenMP ladders for Ewald, PME, D3 and D4.
// Sizes run from 128 to 8192 atoms. Thread counts are powers of two
// up to the physical cores. Each point is at least 6 repeats and the
// spread is the type-7 interquartile range. Every point records the
// hostname, the date and the CPU.

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int threads_now() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void set_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}

double quantile7(std::vector<double> x, double p) {
  std::sort(x.begin(), x.end());
  const double h = (static_cast<double>(x.size()) - 1.0) * p;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const auto hi = static_cast<std::size_t>(std::ceil(h));
  if (lo == hi) {
    return x[lo];
  }
  return x[lo] + (h - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) {
        return line.substr(pos + 2);
      }
    }
  }
  return "unknown";
}

std::string hostname() {
  std::ifstream in("/etc/hostname");
  std::string host;
  std::getline(in, host);
  return host;
}

std::string date_now() {
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::string text = std::ctime(&now);
  if (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  return text;
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> q;
  double box[9]{};
};

System make_system(int n, unsigned seed) {
  System s;
  const double side = std::cbrt(static_cast<double>(n)) * 3.2;
  s.box[0] = s.box[4] = s.box[8] = side;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.4, side - 0.4);
  double qsum = 0.0;
  for (int i = 0; i < n; ++i) {
    s.pos.push_back(u(rng));
    s.pos.push_back(u(rng));
    s.pos.push_back(u(rng));
    s.z.push_back(i % 2 == 0 ? 8 : 1);
    const double q = (i % 2 == 0 ? 0.4 : -0.4);
    s.q.push_back(q);
    qsum += q;
  }
  s.q[0] -= qsum;
  return s;
}

std::string pipe_first(const std::string &cmd) {
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return "unknown";
  }
  char buf[256] = {};
  if (fgets(buf, sizeof(buf), pipe) == nullptr) {
    pclose(pipe);
    return "unknown";
  }
  pclose(pipe);
  std::string text(buf);
  const auto cut = text.find_first_of(" \n");
  if (cut != std::string::npos) {
    text.resize(cut);
  }
  return text;
}

const std::string &binary_sha() {
  static const std::string value = pipe_first("sha256sum /proc/self/exe");
  return value;
}

#ifndef RGPOT_GIT_HEAD
#define RGPOT_GIT_HEAD "unknown"
#endif
#ifndef RGPOT_GIT_TREE
#define RGPOT_GIT_TREE "unknown"
#endif

const char *git_head() { return RGPOT_GIT_HEAD; }

const char *git_tree() { return RGPOT_GIT_TREE; }

template <class Fn>
std::vector<double> repeat(int nrep, Fn &&fn) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(nrep));
  for (int i = 0; i < nrep; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return samples;
}

void report(std::ostream &out, const std::string &ladder, const std::string &name, int n, int threads,
            const std::vector<double> &samples, double energy) {
  const double med = quantile7(samples, 0.5);
  const double iqr = quantile7(samples, 0.75) - quantile7(samples, 0.25);
  out << "run hostname=" << hostname() << " date=" << date_now() << " cpu=" << cpu_model() << '\n';
  out << "binary_sha256=" << binary_sha() << " head=" << git_head() << " tree=" << git_tree() << '\n';
  out << ladder << ' ' << name << " n=" << n << " threads=" << threads
      << " repeats=" << samples.size() << " median_us=" << med << " iqr7_us=" << iqr
      << " energy_eV=" << energy << '\n';
  out << "samples_us";
  for (double sample : samples) {
    out << ' ' << sample;
  }
  out << '\n';
  out.flush();
}

double time_ewald(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                  const std::string &ladder) {
  set_threads(threads);
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::Ewald;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.35;
  cfg.kmax = 5;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.periodic = true;
  rgpot::lr::CoulombPot pot(cfg, std::make_shared<rgpot::lr::FixedCharges>(s.q, 0.0));
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  const auto samples = repeat(nrep, eval);
  const double energy = eval();
  report(log, ladder, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, energy);
  report(echo, ladder, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, energy);
  return energy;
}

double time_pme(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                const std::string &ladder) {
  set_threads(threads);
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::SPME;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.35;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 32;
  cfg.spline_order = 4;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.periodic = true;
  rgpot::lr::CoulombPot pot(cfg, std::make_shared<rgpot::lr::FixedCharges>(s.q, 0.0));
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  const auto samples = repeat(nrep, eval);
  const double energy = eval();
  report(log, ladder, "pme", static_cast<int>(s.z.size()), threads_now(), samples, energy);
  report(echo, ladder, "pme", static_cast<int>(s.z.size()), threads_now(), samples, energy);
  return energy;
}

double time_disp(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                 const std::string &ladder, rgpot::lr::DispersionKind kind, const char *name) {
  set_threads(threads);
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = kind;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.charge = 0.0;
  cfg.cutoff = 12.0;
  cfg.short_cutoff = 3.0;
  cfg.periodic = true;
  // Bohr. The same radii are used for the s-dftd3 and dftd4 timings.
  cfg.lib_disp2 = 15.0;
  cfg.lib_disp3 = 10.0;
  cfg.lib_cn = 10.0;
  rgpot::lr::DispersionPot pot(cfg);
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  const auto samples = repeat(nrep, eval);
  const double energy = eval();
  report(log, ladder, name, static_cast<int>(s.z.size()), threads_now(), samples, energy);
  report(echo, ladder, name, static_cast<int>(s.z.size()), threads_now(), samples, energy);
  return energy;
}

} // namespace

int main(int argc, char **argv) {
  const int nrep = argc > 1 ? std::atoi(argv[1]) : 6;
  if (nrep < 6) {
    std::cerr << "repeats must be at least 6\n";
    return 2;
  }
  const int physical = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  int max_threads = 1;
  for (int t = 1; t <= physical; t *= 2) {
    max_threads = t;
  }
  const char *log_path = std::getenv("LR_LOG");
  std::ofstream log(log_path != nullptr ? log_path : "lr_phys_bench.log");
  log << "hostname " << hostname() << '\n';
  log << "date " << date_now() << '\n';
  log << "cpu " << cpu_model() << '\n';
  log << "physical_cores " << physical << '\n';
  log << "max_threads " << max_threads << '\n';
  log << "repeats " << nrep << '\n';
  log << "pme_grid 32\n";
  log << "ewald_kmax 5\n";
  log << "disp_cutoff_bohr disp2=15 disp3=10 cn=10\n";

  const char *methods = std::getenv("LR_METHODS");
  const char *ladders = std::getenv("LR_LADDERS");
  auto ladder_on = [&](const char *name) {
    if (ladders == nullptr || ladders[0] == '\0') {
      return true;
    }
    return std::strstr(ladders, name) != nullptr;
  };
  const int nmin = std::getenv("LR_NMIN") ? std::atoi(std::getenv("LR_NMIN")) : 0;
  const int nmax = std::getenv("LR_NMAX") ? std::atoi(std::getenv("LR_NMAX")) : 1000000;
  const int tmin = std::getenv("LR_TMIN") ? std::atoi(std::getenv("LR_TMIN")) : 1;
  const int tmax = std::getenv("LR_TMAX") ? std::atoi(std::getenv("LR_TMAX")) : 1000000;
  auto wanted = [&](const char *name) {
    if (methods == nullptr || methods[0] == '\0') {
      return true;
    }
    return std::strstr(methods, name) != nullptr;
  };
  const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
  if (ladder_on("strong"))
  for (int n : sizes) {
    if (n < nmin || n > nmax) {
      continue;
    }
    const System sys = make_system(n, 1000u + static_cast<unsigned>(n));
    for (int threads = 1; threads <= max_threads && threads <= tmax; threads *= 2) {
      if (threads < tmin) {
        continue;
      }
      if (wanted("ewald")) {
        time_ewald(sys, threads, nrep, log, std::cout, "strong");
      }
      if (wanted("pme")) {
        time_pme(sys, threads, nrep, log, std::cout, "strong");
      }
      if (wanted("d3")) {
        time_disp(sys, threads, nrep, log, std::cout, "strong", rgpot::lr::DispersionKind::D3BJ, "d3");
      }
      if (wanted("d4")) {
        time_disp(sys, threads, nrep, log, std::cout, "strong", rgpot::lr::DispersionKind::D4, "d4");
      }
    }
  }
  // Weak ladder: atoms per thread stay at 1024, so 1 thread is 1024
  // atoms and the full machine is 8192. A second ladder starts at 128.
  if (ladder_on("weak"))
  for (int base : {128, 1024}) {
    for (int threads = 1; threads <= max_threads; threads *= 2) {
      const int n = base * threads;
      if (n > 8192 || n < nmin || n > nmax || threads < tmin || threads > tmax) {
        continue;
      }
      const System sys = make_system(n, 2000u + static_cast<unsigned>(n));
      const std::string ladder = base == 128 ? "weak128" : "weak1024";
      if (wanted("ewald")) {
        time_ewald(sys, threads, nrep, log, std::cout, ladder);
      }
      if (wanted("pme")) {
        time_pme(sys, threads, nrep, log, std::cout, ladder);
      }
      if (wanted("d3")) {
        time_disp(sys, threads, nrep, log, std::cout, ladder, rgpot::lr::DispersionKind::D3BJ, "d3");
      }
      if (wanted("d4")) {
        time_disp(sys, threads, nrep, log, std::cout, ladder, rgpot::lr::DispersionKind::D4, "d4");
      }
    }
  }
  return 0;
}
