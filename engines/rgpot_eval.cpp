// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot-eval builds a potential the way pair_coeff * * Name ... does and
// evaluates one geometry read on stdin. The arguments are the name and
// its key value options:
//
//   rgpot-eval Name [key value ...] < geometry
//
// with the geometry
//
//   n
//   ax ay az bx by bz cx cy cz
//   z x y z
//   ...
//
// It prints the energy, one force line per atom and, when the kernel
// reports one, the row-major Cauchy stress in eV/Angstrom^3.

#include "pot_spec.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2 || argc % 2 != 0) {
    std::fprintf(stderr, "usage: rgpot-eval Name [key value ...] < geometry\n");
    return 2;
  }
  rgpot::spec::Options options;
  for (int k = 2; k + 1 < argc; k += 2) {
    options.emplace_back(argv[k], argv[k + 1]);
  }
  rgpot::spec::Built built = rgpot::spec::build(argv[1], options, 0.0);
  if (!built.pot) {
    std::fprintf(stderr, "%s\n", built.error.c_str());
    return 1;
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
  const auto count = static_cast<std::size_t>(n);
  std::vector<int> z(count);
  std::vector<double> xyz(3 * count);
  for (std::size_t i = 0; i < count; ++i) {
    if (std::scanf("%d %lf %lf %lf", &z[i], &xyz[3 * i], &xyz[3 * i + 1], &xyz[3 * i + 2]) != 4) {
      std::fprintf(stderr, "expected z x y z for atom %zu\n", i);
      return 2;
    }
  }
  std::vector<double> forces(3 * count, 0.0);
  rgpot::ForceInput in{count, xyz.data(), z.data(), box};
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  try {
    built.eval(in, &out);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "the potential failed with a non-standard exception\n");
    return 1;
  }
  std::printf("energy %.17g\n", out.energy);
  for (std::size_t i = 0; i < count; ++i) {
    std::printf("force %zu %.17g %.17g %.17g\n", i, forces[3 * i], forces[3 * i + 1],
                forces[3 * i + 2]);
  }
  if (out.has_stress != 0) {
    std::printf("stress");
    for (const double s : out.stress) {
      std::printf(" %.17g", s);
    }
    std::printf("\n");
  }
  return 0;
}
