// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot-lj reads one geometry on stdin and prints energy and forces.
//
//   u0 cutoff psi
//   n
//   ax ay az bx by bz cx cy cz
//   z x y z
//   ...
//
// `rgpot-lj --kJ-per-eV` prints the library's kJ/mol per eV and exits.

#include "lj_eval.hpp"
#include "rgpot/units.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--kJ-per-eV") {
    const double per_ev = rgpot::units::unit_conversion_factor("eV", "kJ/mol");
    std::printf("%.16e\n", per_ev);
    return 0;
  }
  double u0 = 0.0;
  double cutoff = 0.0;
  double psi = 0.0;
  if (std::scanf("%lf %lf %lf", &u0, &cutoff, &psi) != 3) {
    std::fprintf(stderr, "expected u0 cutoff psi\n");
    return 2;
  }
  int n = 0;
  if (std::scanf("%d", &n) != 1 || n < 0) {
    std::fprintf(stderr, "expected an atom count\n");
    return 2;
  }
  double box[9];
  for (double &value : box) {
    if (std::scanf("%lf", &value) != 1) {
      std::fprintf(stderr, "expected nine cell values\n");
      return 2;
    }
  }
  std::vector<int> atomic_numbers(static_cast<size_t>(n));
  std::vector<double> xyz(static_cast<size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    if (std::scanf("%d %lf %lf %lf", &atomic_numbers[static_cast<size_t>(i)],
                   &xyz[static_cast<size_t>(3 * i)], &xyz[static_cast<size_t>(3 * i + 1)],
                   &xyz[static_cast<size_t>(3 * i + 2)]) != 4) {
      std::fprintf(stderr, "expected z x y z for atom %d\n", i);
      return 2;
    }
  }
  std::vector<double> forces(static_cast<size_t>(n) * 3);
  double energy = 0.0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz.data(), atomic_numbers.data(), static_cast<size_t>(n), box, u0,
                    cutoff, psi, forces.data(), &energy, &has_stress, stress) != 0) {
    std::fprintf(stderr, "evaluation rejected the geometry\n");
    return 1;
  }
  std::printf("energy %.16e\n", energy);
  for (int i = 0; i < n; ++i) {
    std::printf("force %d %.16e %.16e %.16e\n", i, forces[static_cast<size_t>(3 * i)],
                forces[static_cast<size_t>(3 * i + 1)],
                forces[static_cast<size_t>(3 * i + 2)]);
  }
  return 0;
}
