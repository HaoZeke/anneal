#pragma once
// Pair and reciprocal kernels shared by the host fallback and nvcc.
// A translation unit that is not compiled by nvcc defines the
// annotations as empty so the same formulas run on the CPU.

#ifdef __CUDACC__
#define RGPOT_HD __host__ __device__
#else
#define RGPOT_HD
#endif

#include <cmath>

namespace rgpot::lr::cuda {

RGPOT_HD inline double hd_erfc(double x) {
#ifdef __CUDA_ARCH__
  return erfc(x);
#else
  return std::erfc(x);
#endif
}

RGPOT_HD inline double hd_exp(double x) {
#ifdef __CUDA_ARCH__
  return exp(x);
#else
  return std::exp(x);
#endif
}

RGPOT_HD inline double hd_sqrt(double x) {
#ifdef __CUDA_ARCH__
  return sqrt(x);
#else
  return std::sqrt(x);
#endif
}

RGPOT_HD inline double real_pair(double ke, double alpha, double qi, double qj, double r,
                                 double *dedr) {
  const double ar = alpha * r;
  const double er = hd_erfc(ar);
  const double phi = er / r;
  const double pi = 3.14159265358979323846;
  const double dphi = -er / (r * r) - (2.0 * alpha / hd_sqrt(pi)) * hd_exp(-ar * ar) / r;
  *dedr = ke * qi * qj * dphi;
  return ke * qi * qj * phi;
}

RGPOT_HD inline double recip_term(double pref, double alpha, double k2, double s2) {
  const double f = hd_exp(-k2 / (4.0 * alpha * alpha)) / k2;
  return pref * f * s2;
}

RGPOT_HD inline double bj_pair(double c6, double r0, double a1, double a2, double r) {
  const double r6 = r * r * r * r * r * r;
  const double damp = a1 * r0 + a2;
  const double rr = r / damp;
  const double rr6 = rr * rr * rr * rr * rr * rr;
  const double f = 1.0 / (1.0 + 6.0 * rr6);
  return -c6 * f / r6;
}

} // namespace rgpot::lr::cuda
