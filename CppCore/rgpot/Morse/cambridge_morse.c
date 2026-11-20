#include "cambridge_morse.h"

#include <math.h>

#include "../cpu_features.h"

// The AVX2 clones and their CPU probe are x86 only. Other targets build the
// scalar kernels alone.
// The clones stay static and reached only behind rgpot_have_avx2_fma(): the ISA
// audit accepts their AVX only after proving that from the library.

static void morse_scalar(const double *x, size_t n, double rho, double *energy,
                         double *gradient) {
#include "cambridge_morse_body.inc"
}

static void lj_scalar(const double *x, size_t n, double *energy,
                      double *gradient) {
#include "cluster_lj_body.inc"
}

#ifdef RGPOT_AVX2_CLONE
RGPOT_TARGET_AVX2
static void morse_avx2(const double *x, size_t n, double rho, double *energy,
                       double *gradient) {
#include "cambridge_morse_body.inc"
}

RGPOT_TARGET_AVX2
static void lj_avx2(const double *x, size_t n, double *energy,
                    double *gradient) {
#include "cluster_lj_body.inc"
}
#endif

void rgpot_cambridge_morse(const double *x, size_t n, double rho,
                           double *energy, double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (rgpot_have_avx2_fma()) {
    morse_avx2(x, n, rho, energy, gradient);
    return;
  }
#endif
  morse_scalar(x, n, rho, energy, gradient);
}

void rgpot_cluster_lj(const double *x, size_t n, double *energy,
                      double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (rgpot_have_avx2_fma()) {
    lj_avx2(x, n, energy, gradient);
    return;
  }
#endif
  lj_scalar(x, n, energy, gradient);
}

RGPOT_HIDDEN int rgpot_pair_test_have_avx2_fma(void) {
  return rgpot_have_avx2_fma();
}

RGPOT_HIDDEN void rgpot_pair_test_morse_scalar(const double *x, size_t n,
                                               double rho, double *energy,
                                               double *gradient) {
  morse_scalar(x, n, rho, energy, gradient);
}

RGPOT_HIDDEN void rgpot_pair_test_morse_avx2(const double *x, size_t n,
                                             double rho, double *energy,
                                             double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (rgpot_have_avx2_fma()) {
    morse_avx2(x, n, rho, energy, gradient);
    return;
  }
#endif
  morse_scalar(x, n, rho, energy, gradient);
}

RGPOT_HIDDEN void rgpot_pair_test_lj_scalar(const double *x, size_t n,
                                            double *energy, double *gradient) {
  lj_scalar(x, n, energy, gradient);
}

RGPOT_HIDDEN void rgpot_pair_test_lj_avx2(const double *x, size_t n,
                                          double *energy, double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (rgpot_have_avx2_fma()) {
    lj_avx2(x, n, energy, gradient);
    return;
  }
#endif
  lj_scalar(x, n, energy, gradient);
}
