#include "cambridge_morse.h"

#include <math.h>

// The AVX2 clones and their CPU probe are x86 only. Other targets build the
// scalar kernels alone.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define RGPOT_AVX2_CLONE 1
#define RGPOT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

static void morse_scalar(const double *x, size_t n, double rho, double *energy,
                         double *gradient) {
#include "cambridge_morse_body.inc"
}

static void lj_scalar(const double *x, size_t n, double *energy, double *gradient) {
#include "cluster_lj_body.inc"
}

#ifdef RGPOT_AVX2_CLONE
RGPOT_TARGET_AVX2
static void morse_avx2(const double *x, size_t n, double rho, double *energy,
                       double *gradient) {
#include "cambridge_morse_body.inc"
}

RGPOT_TARGET_AVX2
static void lj_avx2(const double *x, size_t n, double *energy, double *gradient) {
#include "cluster_lj_body.inc"
}

static int have_avx2_fma(void) {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

void rgpot_cambridge_morse(const double *x, size_t n, double rho, double *energy,
                           double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (have_avx2_fma()) {
    morse_avx2(x, n, rho, energy, gradient);
    return;
  }
#endif
  morse_scalar(x, n, rho, energy, gradient);
}

void rgpot_cluster_lj(const double *x, size_t n, double *energy, double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (have_avx2_fma()) {
    lj_avx2(x, n, energy, gradient);
    return;
  }
#endif
  lj_scalar(x, n, energy, gradient);
}
