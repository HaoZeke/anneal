#include <stddef.h>

/* Dispatched clones with no allowlist entry. The auditor proves each one
   from the linked object: a static target("avx2,fma") body reached only
   behind __builtin_cpu_supports, a static helper reached only from it, a
   body behind an out-of-line probe, and the clones of a target_clones
   function reached only from its resolver. */

__attribute__((target("avx2,fma"), noinline)) static double
clone_tail(const double *x, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; i++)
    s += x[i] * x[i];
  return s;
}

__attribute__((target("avx2,fma"), noinline)) static double
clone_avx2(double *y, const double *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    y[i] = y[i] * x[i] + x[i];
  return clone_tail(y, n);
}

__attribute__((noinline)) static double
clone_scalar(double *y, const double *x, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; i++) {
    y[i] = y[i] * x[i] + x[i];
    s += y[i] * y[i];
  }
  return s;
}

static int have_avx2_fma(void) {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

double dispatch_entry(double *y, const double *x, size_t n) {
  if (have_avx2_fma())
    return clone_avx2(y, x, n);
  return clone_scalar(y, x, n);
}

/* A probe kept out of line, as have_avx2_fma is at -O0. */
__attribute__((noinline)) static int probe_avx2(void) {
  return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2"), noinline)) static void
probed_avx2(double *y, const double *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    y[i] = x[i] - y[i];
}

void probed_entry(double *y, const double *x, size_t n) {
  if (probe_avx2())
    probed_avx2(y, x, n);
}

/* Clang gives the clones global symbols, so any caller could reach them
   and the auditor cannot prove them. GCC makes them local. */
#if !defined(__clang__)
__attribute__((target_clones("default", "avx2", "avx512f"))) void
tc_scale(double *y, const double *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    y[i] = 2.0 * x[i] + y[i];
}
#endif
