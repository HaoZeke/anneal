#include "cambridge_morse.h"

#include <math.h>

#if defined(__GNUC__)
#define RGPOT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define RGPOT_TARGET_AVX2
#endif

static void morse_scalar(const double *x, size_t n, double rho, double *energy,
                         double *gradient) {
#include "cambridge_morse_body.inc"
}

RGPOT_TARGET_AVX2
static void morse_avx2(const double *x, size_t n, double rho, double *energy,
                       double *gradient) {
#include "cambridge_morse_body.inc"
}

static void lj_scalar(const double *x, size_t n, double *energy, double *gradient) {
#include "cluster_lj_body.inc"
}

RGPOT_TARGET_AVX2
static void lj_avx2(const double *x, size_t n, double *energy, double *gradient) {
#include "cluster_lj_body.inc"
}

static int have_avx2_fma(void) {
#if defined(__GNUC__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return 0;
#endif
}

void rgpot_cambridge_morse(const double *x, size_t n, double rho, double *energy,
                           double *gradient) {
  if (have_avx2_fma()) {
    morse_avx2(x, n, rho, energy, gradient);
    return;
  }
  morse_scalar(x, n, rho, energy, gradient);
}

void rgpot_cluster_lj(const double *x, size_t n, double *energy, double *gradient) {
  if (have_avx2_fma()) {
    lj_avx2(x, n, energy, gradient);
    return;
  }
  lj_scalar(x, n, energy, gradient);
}
