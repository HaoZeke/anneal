#include <stddef.h>

/* AVX2 bodies that look dispatched and are not. Each name must be a
   violation with no allowlist entry. */

#define AVX2 __attribute__((target("avx2,fma"), noinline))
#define BODY                                                                   \
  for (size_t i = 0; i < n; i++)                                               \
    y[i] = y[i] * x[i] + x[i];

AVX2 static void unguarded_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void wrong_guard_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void inverted_guard_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void escaped_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void partial_guard_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 void exported_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void lying_probe_avx2(double *y, const double *x, size_t n) { BODY }
AVX2 static void weak_probe_avx2(double *y, const double *x, size_t n) { BODY }

__attribute__((noinline)) static int lying_probe(void) {
  return !__builtin_cpu_supports("avx2");
}

__attribute__((noinline)) static int weak_probe(int fast) {
  return fast || __builtin_cpu_supports("avx2");
}

void (*volatile bad_hook)(double *, const double *, size_t) = escaped_avx2;

void bad_unguarded(double *y, const double *x, size_t n) { unguarded_avx2(y, x, n); }

void bad_wrong_guard(double *y, const double *x, size_t n) {
  if (__builtin_cpu_supports("sse4.2"))
    wrong_guard_avx2(y, x, n);
}

void bad_inverted_guard(double *y, const double *x, size_t n) {
  if (!__builtin_cpu_supports("avx2"))
    inverted_guard_avx2(y, x, n);
}

void bad_escaped(double *y, const double *x, size_t n) {
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    escaped_avx2(y, x, n);
}

void bad_exported(double *y, const double *x, size_t n) {
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    exported_avx2(y, x, n);
}

void bad_partial_guard(double *y, const double *x, size_t n, int fast) {
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    partial_guard_avx2(y, x, n);
  if (fast)
    partial_guard_avx2(y, x, n);
}

void bad_lying_probe(double *y, const double *x, size_t n) {
  if (lying_probe())
    lying_probe_avx2(y, x, n);
}

void bad_weak_probe(double *y, const double *x, size_t n, int fast) {
  if (weak_probe(fast))
    weak_probe_avx2(y, x, n);
}
