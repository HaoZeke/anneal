// MIT License
// Copyright 2023--present rgpot developers

// Batch energies and gradients for D3(BJ), D4 and MBD.
// Positions are Angstrom. The printout is Hartree and Hartree/Bohr.

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/units.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double grad_of_force(double force) {
  return force / rgpot::units::NEG_GRAD_TO_FORCE;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: lr_sets geometries.txt\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << '\n';
    return 2;
  }
  int nsys = 0;
  in >> nsys;
  std::cout.setf(std::ios::scientific);
  std::cout.precision(17);
  for (int s = 0; s < nsys; ++s) {
    std::string name, kind;
    int nat = 0, periodic = 0;
    in >> name >> kind >> nat >> periodic;
    std::vector<int> z(static_cast<std::size_t>(nat));
    std::vector<double> pos(static_cast<std::size_t>(nat) * 3);
    for (int i = 0; i < nat; ++i) {
      in >> z[static_cast<std::size_t>(i)] >> pos[static_cast<std::size_t>(3 * i)] >>
          pos[static_cast<std::size_t>(3 * i + 1)] >> pos[static_cast<std::size_t>(3 * i + 2)];
    }
    double box[9] = {};
    if (periodic) {
      for (int i = 0; i < 9; ++i) {
        in >> box[i];
      }
    }
    if (!in) {
      std::cerr << "truncated geometry file at " << name << '\n';
      return 2;
    }
    rgpot::lr::DispersionConfig cfg;
    cfg.functional = "pbe";
    cfg.atm = true;
    cfg.charge = 0.0;
    cfg.periodic = periodic != 0;
    cfg.cutoff = periodic ? 25.0 : 40.0;
    cfg.short_cutoff = 4.0;
    cfg.skin = 1.0;
    cfg.mbd_beta = 0.83;
    if (kind == "d3") {
      cfg.kind = rgpot::lr::DispersionKind::D3BJ;
    } else if (kind == "d4") {
      cfg.kind = rgpot::lr::DispersionKind::D4;
    } else if (kind == "mbd") {
      cfg.kind = rgpot::lr::DispersionKind::MBD;
    } else {
      std::cerr << "unknown kind " << kind << '\n';
      return 2;
    }
    rgpot::lr::DispersionPot pot(cfg);
    std::vector<double> f(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput fin{static_cast<std::size_t>(nat), pos.data(), z.data(), periodic ? box : nullptr};
    pot.forceImpl(fin, &out);
    const double ha = out.energy * rgpot::units::EV_TO_HARTREE;
    std::cout << "SYS " << name << ' ' << kind << ' ' << ha;
    for (double fi : f) {
      std::cout << ' ' << grad_of_force(fi);
    }
    std::cout << '\n';
  }
  return 0;
}
