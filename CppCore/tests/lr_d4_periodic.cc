// MIT License
// Copyright 2023--present rgpot developers

// Periodic D4 on a geometry passed as text, for the dftd4 comparison.
// The first line is "n side". Each later line is "z x y z".

#include "rgpot/LongRange/Dispersion.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: lr_d4_periodic geometry.txt\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  int n = 0;
  double side = 0.0;
  in >> n >> side;
  if (!in || n < 1 || !(side > 0.0)) {
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
  box[0] = box[4] = box[8] = side;
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D4;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.periodic = true;
  cfg.cutoff = side;
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
  std::cout << "energy_eV " << out.energy << '\n';
  for (int i = 0; i < n; ++i) {
    std::cout << "force_eV_A " << force[static_cast<std::size_t>(3 * i)] << ' '
              << force[static_cast<std::size_t>(3 * i + 1)] << ' '
              << force[static_cast<std::size_t>(3 * i + 2)] << '\n';
  }
  return 0;
}
