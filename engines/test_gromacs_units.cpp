// MIT License
// Copyright 2023--present rgpot developers
//
// The stored conversion matches the constants GROMACS publishes.

#include "rgpot/ExternalPot/md_units.hpp"

#include <gromacs/math/units.h>

#include <cmath>
#include <cstdio>

int main() {
  if (gmx::c_nm2A != rgpot::md::kAngstromPerNm) {
    std::fprintf(stderr, "nm scale %g gromacs %g\n", rgpot::md::kAngstromPerNm,
                 gmx::c_nm2A);
    return 1;
  }
  const double kj = gmx::c_faraday / gmx::c_kilo;
  if (std::abs(kj - rgpot::md::kKjPerEv) > 1e-12) {
    std::fprintf(stderr, "kJ/mol per eV stored %.16e gromacs %.16e\n",
                 rgpot::md::kKjPerEv, kj);
    return 1;
  }
  std::printf("gromacs units kJ/mol per eV %.16e\n", kj);
  return 0;
}
