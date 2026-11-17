#pragma once
// Neighbour-list D3(BJ) with the simple-dftd3 reference coefficients.
// Pair, three-body and coordination-number loops each keep a private
// accumulator per thread and add it once after the parallel region.

#include "rgpot/LongRange/Pairs.hpp"

#include <cstddef>

namespace rgpot::lr {

struct D3BjParam {
  double s6 = 1.0;
  double s8 = 0.7875;
  double s9 = 1.0;
  double a1 = 0.4289;
  double a2 = 4.4407;
  double alp = 14.0;
};

// True when the functional has a tabulated BJ parameter set.
bool d3bj_param(const char *functional, D3BjParam *out);

// Positions and box are Angstrom. Energy is eV. F is eV/Angstrom and is
// added to. Cutoffs are Angstrom. A null box or periodic == false is a
// molecule.
void d3bj_evaluate(const double *pos, const int *z, std::size_t n, const double *box,
                   bool periodic, const std::vector<Pair> &pairs, const D3BjParam &param,
                   double cut2, double cut3, double cut_cn, double *energy, double *F,
                   double strain[6]);

} // namespace rgpot::lr
