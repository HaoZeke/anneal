// MIT License
// Copyright 2026 rgpot developers

#include "rgpot/LongRange/Electro.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> q;
  double box[9]{};
};

System make_system(int n) {
  System s;
  const double side = std::cbrt(static_cast<double>(n)) * 3.2;
  s.box[0] = s.box[4] = s.box[8] = side;
  std::mt19937 rng(1000u + static_cast<unsigned>(n));
  std::uniform_real_distribution<double> uniform(0.4, side - 0.4);
  double qsum = 0.0;
  for (int i = 0; i < n; ++i) {
    s.pos.push_back(uniform(rng));
    s.pos.push_back(uniform(rng));
    s.pos.push_back(uniform(rng));
    s.z.push_back(i % 2 == 0 ? 8 : 1);
    const double charge = i % 2 == 0 ? 0.4 : -0.4;
    s.q.push_back(charge);
    qsum += charge;
  }
  s.q[0] -= qsum;
  return s;
}

void set_threads(int threads) {
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#pragma omp parallel
  {
  }
#else
  (void)threads;
#endif
}

double evaluate(const System &system, rgpot::lr::CoulombPot &pot,
                std::vector<double> &forces) {
  rgpot::ForceOut out{};
  out.F = forces.data();
  const rgpot::ForceInput in{system.z.size(), system.pos.data(),
                             system.z.data(), system.box};
  pot.forceImpl(in, &out);
  return out.energy;
}

double force_norm(const std::vector<double> &forces) {
  double force_norm = 0.0;
  for (const double force : forces) {
    force_norm += force * force;
  }
  return std::sqrt(force_norm);
}

void dump(const System &system, int threads) {
  set_threads(threads);
  rgpot::lr::CoulombConfig config;
  config.method = rgpot::lr::CoulombMethod::DSF;
  config.cutoff = 5.0;
  config.alpha = 0.3;
  config.periodic = false;
  config.stress = false;
  rgpot::lr::CoulombPot pot(
      config, std::make_shared<rgpot::lr::FixedCharges>(system.q, 0.0));
  std::vector<double> forces(system.pos.size());
  const double energy = evaluate(system, pot, forces);
  std::cout << std::scientific << std::setprecision(17);
  std::cout << "energy_eV " << energy << '\n';
  for (std::size_t i = 0; i < system.z.size(); ++i) {
    std::cout << "force_eV_A " << forces[3 * i] << ' '
              << forces[3 * i + 1] << ' ' << forces[3 * i + 2] << '\n';
  }
}

void time_cell(const std::string &arm, int round, int n, int threads,
               int repeats) {
  set_threads(threads);
  const System system = make_system(n);
  rgpot::lr::CoulombConfig config;
  config.method = rgpot::lr::CoulombMethod::DSF;
  config.cutoff = 5.0;
  config.alpha = 0.3;
  config.periodic = false;
  config.stress = false;
  rgpot::lr::CoulombPot pot(
      config, std::make_shared<rgpot::lr::FixedCharges>(system.q, 0.0));
  std::vector<double> forces(system.pos.size());
  for (int warmup = 0; warmup < 5; ++warmup) {
    evaluate(system, pot, forces);
  }
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    const double energy = evaluate(system, pot, forces);
    const auto end = std::chrono::steady_clock::now();
    const double wall_us =
        std::chrono::duration<double, std::micro>(end - begin).count();
    const double norm = force_norm(forces);
    std::cout << arm << ',' << round << ",dsf," << n << ',' << threads
              << ',' << repeat << ',' << std::setprecision(17) << wall_us
              << ',' << energy << ',' << norm << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 4 && std::string(argv[1]) == "dump") {
    dump(make_system(std::atoi(argv[2])), std::atoi(argv[3]));
    return 0;
  }
  if (argc != 4) {
    std::cerr << "usage: dsf_phys_bench ARM ROUND REPEATS\n";
    return 2;
  }
  const std::string arm = argv[1];
  const int round = std::atoi(argv[2]);
  const int repeats = std::atoi(argv[3]);
  if (repeats < 25) {
    std::cerr << "repeats must be at least 25\n";
    return 2;
  }
  std::cout << "arm,round,method,natoms,threads,repeat,wall_us,energy_eV,"
               "force_norm\n";
  for (const int n : {128, 256, 512, 1024}) {
    for (const int threads : {1, 2, 4, 8}) {
      time_cell(arm, round, n, threads, repeats);
    }
  }
  return 0;
}
