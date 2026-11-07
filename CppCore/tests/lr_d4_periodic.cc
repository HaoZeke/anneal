// MIT License
// Copyright 2023--present rgpot developers

// Periodic D4 on a geometry passed as text, for the dftd4 comparison.
// The first line is "n side". Each later line is "z x y z".

#include "rgpot/LongRange/Dispersion.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef RGPOT_GIT_HEAD
#define RGPOT_GIT_HEAD "unknown"
#endif
#ifndef RGPOT_GIT_TREE
#define RGPOT_GIT_TREE "unknown"
#endif
#ifndef RGPOT_GIT_DIRTY
#define RGPOT_GIT_DIRTY -1
#endif
#ifndef RGPOT_COMPILER
#define RGPOT_COMPILER "unknown"
#endif

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: lr_d4_periodic geometry.txt\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  std::string header;
  std::getline(in, header);
  std::istringstream hs(header);
  int n = 0;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  hs >> n >> ax;
  if (!(hs >> ay)) {
    ay = ax;
  }
  if (!(hs >> az)) {
    az = ax;
  }
  if (!hs || n < 1 || !(ax > 0.0) || !(ay > 0.0) || !(az > 0.0)) {
    std::cerr << "bad geometry\n";
    return 2;
  }
  std::vector<double> pos(static_cast<std::size_t>(3 * n));
  std::vector<int> z(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    in >> z[static_cast<std::size_t>(i)] >> pos[static_cast<std::size_t>(3 * i)] >>
        pos[static_cast<std::size_t>(3 * i + 1)] >> pos[static_cast<std::size_t>(3 * i + 2)];
  }
  double box[9] = {};
  box[0] = ax;
  box[4] = ay;
  box[8] = az;
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D4;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.periodic = true;
  cfg.cutoff = std::min(ax, std::min(ay, az));
  cfg.short_cutoff = 2.0;
  cfg.skin = 0.0;
  cfg.lib_disp2 = 60.0;
  cfg.lib_disp3 = 40.0;
  cfg.lib_cn = 30.0;
  rgpot::lr::DispersionPot pot(cfg);
  std::vector<double> force(pos.size());
  rgpot::ForceOut out{};
  out.F = force.data();
  rgpot::ForceInput fin{static_cast<std::size_t>(n), pos.data(), z.data(), box};
  pot.forceImpl(fin, &out);
  std::cout.setf(std::ios::scientific);
  std::cout.precision(17);
  std::cout << "provenance head=" << RGPOT_GIT_HEAD << " tree=" << RGPOT_GIT_TREE
            << " dirty=" << RGPOT_GIT_DIRTY << " compiler=" << RGPOT_COMPILER << '\n';
  std::cout << "energy_eV " << out.energy << '\n';
  for (int i = 0; i < n; ++i) {
    std::cout << "force_eV_A " << force[static_cast<std::size_t>(3 * i)] << ' '
              << force[static_cast<std::size_t>(3 * i + 1)] << ' '
              << force[static_cast<std::size_t>(3 * i + 2)] << '\n';
  }
  return 0;
}
