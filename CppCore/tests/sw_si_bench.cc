// MIT License
// Copyright 2023--present rgpot developers
//
// Calls per second of StillingerWeberSi::energyAndGradient on a 216-atom
// diamond cell, and of forceImpl, which also writes the stress. Three
// repeats, CPU model printed from /proc/cpuinfo.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/nlist/HalfShell.hpp"

namespace {

std::string cpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.compare(0, key.size(), key) != 0) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string name = line.substr(colon + 1);
    const auto first = name.find_first_not_of(" \t");
    if (first != std::string::npos) {
      name.erase(0, first);
    }
    return name;
  }
  return "unknown";
}

double pairMinimumU() {
  auto df = [](double u) {
    const double s = 1.0 / (u - rgpot::StillingerWeberSi::kCut);
    const double inv = 1.0 / u;
    const double um4 = inv * inv * inv * inv;
    const double g = rgpot::StillingerWeberSi::kB * um4 - 1.0;
    const double es = std::exp(s);
    const double dgdu = -rgpot::StillingerWeberSi::kP *
                        rgpot::StillingerWeberSi::kB * um4 * inv;
    return rgpot::StillingerWeberSi::kA * es * (dgdu - g * s * s);
  };
  double u = 1.12;
  for (int it = 0; it < 12; ++it) {
    const double h = 1e-7;
    u -= df(u) / ((df(u + h) - df(u - h)) / (2.0 * h));
  }
  return u;
}

} // namespace

int main() {
  const double u = pairMinimumU();
  const double a0 = u * rgpot::StillingerWeberSi::kSigma * 4.0 / std::sqrt(3.0);
  const int nrep = 3;
  const double basis[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                              {0.5, 0, 0.5},      {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nrep * nrep * nrep * 8 * 3));
  for (int ix = 0; ix < nrep; ++ix) {
    for (int iy = 0; iy < nrep; ++iy) {
      for (int iz = 0; iz < nrep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a0);
          pos.push_back((iy + b[1]) * a0);
          pos.push_back((iz + b[2]) * a0);
        }
      }
    }
  }
  const std::size_t n_atoms = pos.size() / 3;
  const double box[9] = {nrep * a0, 0, 0, 0, nrep * a0, 0, 0, 0, nrep * a0};
  std::vector<double> grad(pos.size());
  rgpot::StillingerWeberSi pot;
  double energy = 0.0;
  const std::vector<int> types(n_atoms, 14);
  auto call = [&](rgpot::nlist::NeighborList *list, bool stress) {
    if (stress) {
      rgpot::ForceOut out{grad.data(), 0.0, 0.0, {}, 0};
      pot.forceImpl({n_atoms, pos.data(), types.data(), box}, &out);
      energy = out.energy;
    } else {
      pot.energyAndGradient(pos.data(), nullptr, box, n_atoms, &energy,
                            grad.data(), list);
    }
  };
  auto report = [&](const char *mode, int calls,
                    rgpot::nlist::NeighborList *list, bool stress) {
    for (int i = 0; i < 5; ++i) {
      call(list, stress);
    }
    std::cout << "mode: " << mode << "\n";
    std::cout << "entry: " << (stress ? "forceImpl" : "energyAndGradient")
              << "\n";
    std::cout << "calls_per_repeat: " << calls << "\n";
    for (int rep = 1; rep <= 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int c = 0; c < calls; ++c) {
        call(list, stress);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      const double rate = static_cast<double>(calls) / sec;
      std::cout << "repeat " << rep << ": " << rate << " calls/s\n";
    }
  };
  std::cout << "cpu: " << cpuModel() << "\n";
  std::cout << "potential: StillingerWeberSi\n";
  std::cout << "n_atoms: " << n_atoms << "\n";
  std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
  std::cout.precision(6);
  report("rebuild", 100, nullptr, false);
  rgpot::nlist::NeighborList held;
  report("held", 200, &held, false);
  report("forceImpl_stress", 100, nullptr, true);
  std::cout << "energy_eV: " << energy << "\n";
  return energy < 0.0 ? 0 : 1;
}
