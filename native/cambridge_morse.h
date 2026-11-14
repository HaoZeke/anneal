#ifndef RGPOT_CAMBRIDGE_MORSE_H
#define RGPOT_CAMBRIDGE_MORSE_H
/*
 * Free-cluster pair sums used by the Cambridge tables and by GMIN.
 *
 * Neither sum has a cutoff. A neighbour list that dropped a distant pair
 * would change the energy: the Morse tail is exp(rho(1-r)), which is not
 * zero, and the Lennard-Jones tail is not zero either. The walk is every
 * pair i < j, in that order.
 *
 * Energy and the force coefficient (dV/dr)/r are one pass. Each force
 * component is an fma into the running sum, which is the anneal
 * PairPotential accumulation. libm exp is the Morse transcendental:
 * on this kernel it is faster than the portable polynomial in
 * rgpot/simd/exp.hpp (about 29 us against 38 us at 80 atoms).
 */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

void rgpot_cambridge_morse(const double *x, size_t n, double rho, double *energy,
                           double *gradient);

void rgpot_cluster_lj(const double *x, size_t n, double *energy, double *gradient);

#ifdef __cplusplus
}
#endif
#endif
