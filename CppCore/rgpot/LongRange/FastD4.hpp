#pragma once
// Cutoff EEQ and a neighbour-list D4(BJ) evaluation.
// Gas-phase reference sets stay on dftd4. This kernel is the large-system
// path: real-space pairs come from the linkcell list, the reciprocal
// Coulomb part is a short lattice sum, and the charge block is solved
// by conjugate gradient instead of a dense factorization.

#include "rgpot/LongRange/Pairs.hpp"

#include <cstddef>

namespace rgpot::lr {

struct D4BjParam {
  double s6 = 1.0;
  double s8 = 0.95948085;
  double s9 = 1.0;
  double a1 = 0.38574991;
  double a2 = 4.80688534;
  double alp = 16.0;
};

bool d4bj_param(const char *functional, D4BjParam *out);

// Positions and the box are Angstrom. Energy is eV. Forces are eV/Angstrom
// and are added to. Dispersion cutoffs are Angstrom.
void d4_evaluate(const double *pos, const int *z, std::size_t n, const double *box, bool periodic,
                 double charge, const D4BjParam &param, double cut2, double cut3, double cut_cn,
                 double *energy, double *F, double strain[6]);

} // namespace rgpot::lr
