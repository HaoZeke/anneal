// MIT License
// Copyright 2023--present rgpot developers

// Serial head-to-head and OpenMP scaling for the long-range kernels.
// Each sample is one force evaluation. The log records hostname, date
// and the CPU model. Repeats are at least 6. The reported spread is the
// type-7 interquartile range.

#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Dispersion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> q;
  double box[9];
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

template <class Fn>
std::vector<double> repeat(int nrep, Fn &&fn) {
  fn();
  std::vector<double> samples;
  for (int i = 0; i < nrep; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return samples;
}

void report(std::ostream &out, const std::string &name, int n, int threads,
            const std::vector<double> &samples, double energy) {
  const double med = quantile7(samples, 0.5);
  const double iqr = quantile7(samples, 0.75) - quantile7(samples, 0.25);
  out << name << " n=" << n << " threads=" << threads << " repeats=" << samples.size()
      << " median_us=" << med << " iqr7_us=" << iqr << " energy_eV=" << energy << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const int nrep = argc > 1 ? std::atoi(argv[1]) : 6;
  const int physical = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  std::ofstream log("lr_phys_bench.log");
  log << "hostname " << hostname() << '\n';
  log << "date " << std::chrono::system_clock::now().time_since_epoch().count() << '\n';
  {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log << "date_utc " << std::ctime(&now);
  }
  log << "cpu " << cpu_model() << '\n';
  log << "physical_cores " << physical << '\n';
  log << "repeats " << nrep << '\n';

  auto time_ewald = [&](const System &s, int threads) {
    set_threads(threads);
    rgpot::lr::CoulombConfig cfg;
    cfg.method = rgpot::lr::CoulombMethod::Ewald;
    cfg.cutoff = 8.0;
    cfg.alpha = 0.35;
    cfg.kmax = 6;
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
    report(log, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, eval());
    report(std::cout, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, eval());
  };

  auto time_qdo = [&](const System &s, int threads) {
    set_threads(threads);
    rgpot::lr::DispersionConfig cfg;
    cfg.kind = rgpot::lr::DispersionKind::QDO;
    cfg.cutoff = 10.0;
    cfg.short_cutoff = 3.0;
    cfg.periodic = true;
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
    report(log, "qdo", static_cast<int>(s.z.size()), threads_now(), samples, eval());
    report(std::cout, "qdo", static_cast<int>(s.z.size()), threads_now(), samples, eval());
  };

  const System base = make_system(128, 7);
  for (int threads = 1; threads <= physical; threads *= 2) {
    time_ewald(base, threads);
    time_qdo(base, threads);
  }
  for (int threads = 1; threads <= physical; threads *= 2) {
    const System grown = make_system(32 * threads, 11 + threads);
    time_ewald(grown, threads);
  }
  log.close();
  return 0;
}
