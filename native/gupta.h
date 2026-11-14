#ifndef RGPOT_GUPTA_H
#define RGPOT_GUPTA_H
/*
 * Cleri–Rosato Gupta sum over every pair. There is no cutoff: the
 * Cambridge minima are the all-pairs energies. The density is accumulated
 * in label order, then the force uses the stored pair derivative, so a
 * second exponential is not taken and the sum matches that order.
 */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

void rgpot_gupta(const double *x, size_t n, const unsigned char *slot, size_t n_species,
                 const double *a, const double *xi, const double *p, const double *q,
                 const double *r0, double *energy, double *gradient);

#ifdef __cplusplus
}
#endif
#endif
