#include "gupta.h"

#include <math.h>
#include <stdlib.h>

// The AVX2 clone and its CPU probe are x86 only. Other targets build the
// scalar kernel alone.
// The clone stays static and reached only behind have_avx2_fma(): the ISA
// audit accepts its AVX only after proving that from the library.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define RGPOT_AVX2_CLONE 1
#define RGPOT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define RGPOT_THREAD_LOCAL __declspec(thread)
#else
#define RGPOT_THREAD_LOCAL _Thread_local
#endif

static RGPOT_THREAD_LOCAL double *rho_buf;
static RGPOT_THREAD_LOCAL size_t rho_cap;
static RGPOT_THREAD_LOCAL double *pack_buf;
static RGPOT_THREAD_LOCAL size_t pack_cap;

static int rgpot_gupta_ensure(size_t n, size_t pairs) {
  if (rho_cap < n) {
    double *next = (double *)realloc(rho_buf, n * sizeof(double));
    if (next == 0) {
      return 1;
    }
    rho_buf = next;
    rho_cap = n;
  }
  const size_t need = pairs * 6;
  if (pack_cap < need) {
    double *next = (double *)realloc(pack_buf, need * sizeof(double));
    if (next == 0) {
      return 1;
    }
    pack_buf = next;
    pack_cap = need;
  }
  return 0;
}

static double *rgpot_gupta_rho(void) { return rho_buf; }

static double *rgpot_gupta_pack(void) { return pack_buf; }

static void gupta_scalar(const double *x, size_t n, const unsigned char *slot,
                         size_t n_species, const double *a, const double *xi, const double *p,
                         const double *q, const double *r0, double *energy, double *gradient) {
#include "gupta_body.inc"
}

#ifdef RGPOT_AVX2_CLONE
RGPOT_TARGET_AVX2
static void gupta_avx2(const double *x, size_t n, const unsigned char *slot, size_t n_species,
                       const double *a, const double *xi, const double *p, const double *q,
                       const double *r0, double *energy, double *gradient) {
#include "gupta_body.inc"
}

static int have_avx2_fma(void) {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

void rgpot_gupta(const double *x, size_t n, const unsigned char *slot, size_t n_species,
                 const double *a, const double *xi, const double *p, const double *q,
                 const double *r0, double *energy, double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (have_avx2_fma()) {
    gupta_avx2(x, n, slot, n_species, a, xi, p, q, r0, energy, gradient);
    return;
  }
#endif
  gupta_scalar(x, n, slot, n_species, a, xi, p, q, r0, energy, gradient);
}
