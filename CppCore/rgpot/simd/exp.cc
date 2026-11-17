// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exp.hpp"
#include "rgpot/simd/clones.hpp"
#include "rgpot/simd/pair_loops.hpp"

#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace rgpot::simd {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
double
exp(double x) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
  return exp_fast<true>(x);
#elif defined(__x86_64__)
  if (detail::fma_hw_active()) {
    double y = 0;
    exp_hw_n(&x, &y, 1);
    return y;
  }
  return exp_fast<false>(x);
#else
  return exp_fast<false>(x);
#endif
}

#if defined(__x86_64__)

// target("fma") also emits AVX encodings. rgpot/isa-allow.toml names these
// functions. Every call site is behind detail::fma_hw_active(), which
// follows glibc's hwcaps mask. noinline keeps that body out of the baseline
// caller, which stays on SSE2 or the Dekker product.
__attribute__((target("fma"), flatten, noinline)) double lj_hw(const PairView &v,
                                                      const LjParams &p,
                                                      double *F,
                                                      double strain[6]) {
  return lj_scalar<true>(v, p, F, strain);
}

__attribute__((target("fma"), flatten, noinline)) double
lj_hw_span(const PairView &v, const LjParams &p, double *F, double strain[6],
           std::size_t atom_begin, std::size_t atom_end) {
  return lj_scalar<true>(v, p, F, strain, atom_begin, atom_end);
}

__attribute__((target("fma"), flatten, noinline)) double morse_hw(const PairView &v,
                                                         const MorseParams &p,
                                                         double *F,
                                                         double strain[6]) {
  return morse_scalar<true>(v, p, F, strain);
}

__attribute__((target("fma"), flatten, noinline)) void eam_hw(long n, const double *R,
                                                     const EamAtom *atoms,
                                                     const double *box,
                                                     double *F, double *U) {
  eam_scalar<true>(n, R, atoms, box, F, U);
}

__attribute__((target("fma"), noinline)) double exact_fma_hw(double a, double b,
                                                             double c) {
  return __builtin_fma(a, b, c);
}

__attribute__((target("fma"), noinline)) double r2_hw(double dx, double dy,
                                                      double dz) {
  return __builtin_fma(dz, dz, __builtin_fma(dy, dy, dx * dx));
}

__attribute__((target("fma"), noinline)) void r2_hw_n(const double *dx,
                                                      const double *dy,
                                                      const double *dz,
                                                      double *out,
                                                      std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    const __m128d x = _mm_loadu_pd(dx + i);
    const __m128d y = _mm_loadu_pd(dy + i);
    const __m128d z = _mm_loadu_pd(dz + i);
    const __m128d r2 =
        _mm_fmadd_pd(z, z, _mm_fmadd_pd(y, y, _mm_mul_pd(x, x)));
    _mm_storeu_pd(out + i, r2);
  }
  for (; i < n; ++i) {
    out[i] = __builtin_fma(dz[i], dz[i], __builtin_fma(dy[i], dy[i], dx[i] * dx[i]));
  }
}

// One pass: the squared distance is fused while dx, dy, dz are still in
// registers. Storing every candidate and calling r2_hw_n reloads them.
__attribute__((target("fma"), noinline)) std::size_t
accept_row_hw(const PairView &v, std::size_t i, double *dxb, double *dyb,
              double *dzb, double *r2b, std::int32_t *jb) {
  const std::int32_t p0 = v.rows[i];
  const std::int32_t p1 = v.rows[i + 1];
  const double xi = v.R[3 * i];
  const double yi = v.R[3 * i + 1];
  const double zi = v.R[3 * i + 2];
  const int fold = v.fold;
  std::size_t m = 0;
  for (std::int32_t p = p0; p < p1; ++p) {
    const auto up = static_cast<std::size_t>(p);
    const auto j = static_cast<std::size_t>(v.nbr[up]);
    double dx = xi - v.R[3 * j];
    double dy = yi - v.R[3 * j + 1];
    double dz = zi - v.R[3 * j + 2];
    if (fold == 1) {
      dx = dx - v.w[0] * visit_round(dx * v.inv[0]);
      dy = dy - v.w[1] * visit_round(dy * v.inv[1]);
      dz = dz - v.w[2] * visit_round(dz * v.inv[2]);
    } else if (fold == 2) {
      const double *sh = v.shift + 3 * static_cast<std::size_t>(v.code[up]);
      dx = dx - sh[0];
      dy = dy - sh[1];
      dz = dz - sh[2];
    }
    const double r2 = __builtin_fma(dz, dz, __builtin_fma(dy, dy, dx * dx));
    if (!(r2 <= v.cutoff2)) {
      continue;
    }
    dxb[m] = dx;
    dyb[m] = dy;
    dzb[m] = dz;
    r2b[m] = r2;
    jb[m] = static_cast<std::int32_t>(j);
    ++m;
  }
  return m;
}

#include "rgpot/simd/sse_soft.hpp"

__attribute__((target("fma"), always_inline)) static inline __m128d exp2_hw_finite(__m128d x_in) {
  const __m128i sign = _mm_and_si128(_mm_castpd_si128(x_in),
                                     _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
  const __m128d half = _mm_or_pd(_mm_set1_pd(0.5), _mm_castsi128_pd(sign));
  const __m128d scaled = _mm_fmadd_pd(_mm_set1_pd(kInvLn2), x_in, half);
  const __m128i k32 = _mm_cvttpd_epi32(scaled);
  const __m128d tk = _mm_cvtepi32_pd(k32);
  const __m128i k64 = _mm_unpacklo_epi32(k32, _mm_srai_epi32(k32, 31));
  const __m128d hi = _mm_sub_pd(x_in, _mm_mul_pd(tk, _mm_set1_pd(kLn2Hi)));
  const __m128d lo = _mm_mul_pd(tk, _mm_set1_pd(kLn2Lo));
  const __m128d x = _mm_sub_pd(hi, lo);
  const __m128d t = _mm_mul_pd(x, x);
  __m128d p = _mm_fmadd_pd(t, _mm_set1_pd(kP5), _mm_set1_pd(kP4));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP3));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP2));
  p = _mm_fmadd_pd(t, p, _mm_set1_pd(kP1));
  const __m128d c = _mm_sub_pd(x, _mm_mul_pd(t, p));
  const __m128d quot = _mm_div_pd(_mm_mul_pd(x, c), _mm_sub_pd(_mm_set1_pd(2.0), c));
  const __m128d y = _mm_sub_pd(_mm_set1_pd(1.0), _mm_sub_pd(_mm_sub_pd(lo, quot), hi));
  const __m128i yb = _mm_castpd_si128(y);
  const __m128i yexp = _mm_and_si128(_mm_srli_epi64(yb, 52), _mm_set1_epi64x(0x7ff));
  const __m128i ne = _mm_add_epi64(yexp, k64);
  const __m128i packed =
      _mm_or_si128(_mm_and_si128(yb, _mm_set1_epi64x(static_cast<long long>(0x800fffffffffffffULL))),
                   _mm_slli_epi64(ne, 52));
  return _mm_castsi128_pd(packed);
}

__attribute__((target("fma"), noinline)) __m128d exp2_hw(__m128d x_in) {
  const __m128d ax = _mm_andnot_pd(_mm_set1_pd(-0.0), x_in);
  // hx >= 0x40862e42, the fdlibm overflow/underflow/non-finite bin.
  const __m128d lim = _mm_castsi128_pd(_mm_set1_epi64x(0x40862e4200000000LL));
  const __m128d bad = _mm_or_pd(_mm_cmpge_pd(ax, lim), _mm_cmpunord_pd(x_in, x_in));
  if (_mm_movemask_pd(bad) == 0) {
    return exp2_hw_finite(x_in);
  }
  alignas(16) double in[2], out[2];
  _mm_store_pd(in, x_in);
  out[0] = exp(in[0]);
  out[1] = exp(in[1]);
  const int mask = _mm_movemask_pd(bad);
  if (mask != 3) {
    const __m128d fast = exp2_hw_finite(x_in);
    alignas(16) double ff[2];
    _mm_store_pd(ff, fast);
    if ((mask & 1) == 0) {
      out[0] = ff[0];
    }
    if ((mask & 2) == 0) {
      out[1] = ff[1];
    }
  }
  return _mm_load_pd(out);
}


__attribute__((target("fma"), always_inline)) static inline void exp8_hw(const double *in, double *out) {
  __m128d x0 = _mm_loadu_pd(in);
  __m128d x1 = _mm_loadu_pd(in + 2);
  __m128d x2 = _mm_loadu_pd(in + 4);
  __m128d x3 = _mm_loadu_pd(in + 6);
  const __m128d lim = _mm_castsi128_pd(_mm_set1_epi64x(0x40862e4200000000LL));
  auto bad_of = [&](__m128d x) {
    const __m128d ax = _mm_andnot_pd(_mm_set1_pd(-0.0), x);
    return _mm_or_pd(_mm_cmpge_pd(ax, lim), _mm_cmpunord_pd(x, x));
  };
  if ((_mm_movemask_pd(bad_of(x0)) | _mm_movemask_pd(bad_of(x1)) |
       _mm_movemask_pd(bad_of(x2)) | _mm_movemask_pd(bad_of(x3))) != 0) {
    for (int i = 0; i < 8; ++i) {
      out[i] = exp(in[i]);
    }
    return;
  }
  auto half_of = [](__m128d x) {
    const __m128i sign = _mm_and_si128(
        _mm_castpd_si128(x), _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
    return _mm_or_pd(_mm_set1_pd(0.5), _mm_castsi128_pd(sign));
  };
  const __m128d inv = _mm_set1_pd(kInvLn2);
  __m128d s0 = _mm_fmadd_pd(inv, x0, half_of(x0));
  __m128d s1 = _mm_fmadd_pd(inv, x1, half_of(x1));
  __m128d s2 = _mm_fmadd_pd(inv, x2, half_of(x2));
  __m128d s3 = _mm_fmadd_pd(inv, x3, half_of(x3));
  __m128i k32_0 = _mm_cvttpd_epi32(s0);
  __m128i k32_1 = _mm_cvttpd_epi32(s1);
  __m128i k32_2 = _mm_cvttpd_epi32(s2);
  __m128i k32_3 = _mm_cvttpd_epi32(s3);
  __m128d tk0 = _mm_cvtepi32_pd(k32_0);
  __m128d tk1 = _mm_cvtepi32_pd(k32_1);
  __m128d tk2 = _mm_cvtepi32_pd(k32_2);
  __m128d tk3 = _mm_cvtepi32_pd(k32_3);
  __m128i k0 = _mm_unpacklo_epi32(k32_0, _mm_srai_epi32(k32_0, 31));
  __m128i k1 = _mm_unpacklo_epi32(k32_1, _mm_srai_epi32(k32_1, 31));
  __m128i k2 = _mm_unpacklo_epi32(k32_2, _mm_srai_epi32(k32_2, 31));
  __m128i k3 = _mm_unpacklo_epi32(k32_3, _mm_srai_epi32(k32_3, 31));
  const __m128d ln_hi = _mm_set1_pd(kLn2Hi);
  const __m128d ln_lo = _mm_set1_pd(kLn2Lo);
  __m128d hi0 = _mm_sub_pd(x0, _mm_mul_pd(tk0, ln_hi));
  __m128d hi1 = _mm_sub_pd(x1, _mm_mul_pd(tk1, ln_hi));
  __m128d hi2 = _mm_sub_pd(x2, _mm_mul_pd(tk2, ln_hi));
  __m128d hi3 = _mm_sub_pd(x3, _mm_mul_pd(tk3, ln_hi));
  __m128d lo0 = _mm_mul_pd(tk0, ln_lo);
  __m128d lo1 = _mm_mul_pd(tk1, ln_lo);
  __m128d lo2 = _mm_mul_pd(tk2, ln_lo);
  __m128d lo3 = _mm_mul_pd(tk3, ln_lo);
  __m128d r0 = _mm_sub_pd(hi0, lo0);
  __m128d r1 = _mm_sub_pd(hi1, lo1);
  __m128d r2 = _mm_sub_pd(hi2, lo2);
  __m128d r3 = _mm_sub_pd(hi3, lo3);
  __m128d t0 = _mm_mul_pd(r0, r0);
  __m128d t1 = _mm_mul_pd(r1, r1);
  __m128d t2 = _mm_mul_pd(r2, r2);
  __m128d t3 = _mm_mul_pd(r3, r3);
  const __m128d p5 = _mm_set1_pd(kP5);
  const __m128d p4c = _mm_set1_pd(kP4);
  const __m128d p3c = _mm_set1_pd(kP3);
  const __m128d p2c = _mm_set1_pd(kP2);
  const __m128d p1c = _mm_set1_pd(kP1);
  __m128d p0 = _mm_fmadd_pd(t0, p5, p4c);
  __m128d p1 = _mm_fmadd_pd(t1, p5, p4c);
  __m128d p2 = _mm_fmadd_pd(t2, p5, p4c);
  __m128d p3 = _mm_fmadd_pd(t3, p5, p4c);
  p0 = _mm_fmadd_pd(t0, p0, p3c);
  p1 = _mm_fmadd_pd(t1, p1, p3c);
  p2 = _mm_fmadd_pd(t2, p2, p3c);
  p3 = _mm_fmadd_pd(t3, p3, p3c);
  p0 = _mm_fmadd_pd(t0, p0, p2c);
  p1 = _mm_fmadd_pd(t1, p1, p2c);
  p2 = _mm_fmadd_pd(t2, p2, p2c);
  p3 = _mm_fmadd_pd(t3, p3, p2c);
  p0 = _mm_fmadd_pd(t0, p0, p1c);
  p1 = _mm_fmadd_pd(t1, p1, p1c);
  p2 = _mm_fmadd_pd(t2, p2, p1c);
  p3 = _mm_fmadd_pd(t3, p3, p1c);
  auto finish = [](__m128d x, __m128d t, __m128d p, __m128d hi, __m128d lo, __m128i k) {
    const __m128d c = _mm_sub_pd(x, _mm_mul_pd(t, p));
    const __m128d quot = _mm_div_pd(_mm_mul_pd(x, c), _mm_sub_pd(_mm_set1_pd(2.0), c));
    const __m128d y = _mm_sub_pd(_mm_set1_pd(1.0), _mm_sub_pd(_mm_sub_pd(lo, quot), hi));
    const __m128i yb = _mm_castpd_si128(y);
    const __m128i yexp = _mm_and_si128(_mm_srli_epi64(yb, 52), _mm_set1_epi64x(0x7ff));
    const __m128i ne = _mm_add_epi64(yexp, k);
    const __m128i packed = _mm_or_si128(
        _mm_and_si128(yb, _mm_set1_epi64x(static_cast<long long>(0x800fffffffffffffULL))),
        _mm_slli_epi64(ne, 52));
    return _mm_castsi128_pd(packed);
  };
  _mm_storeu_pd(out, finish(r0, t0, p0, hi0, lo0, k0));
  _mm_storeu_pd(out + 2, finish(r1, t1, p1, hi1, lo1, k1));
  _mm_storeu_pd(out + 4, finish(r2, t2, p2, hi2, lo2, k2));
  _mm_storeu_pd(out + 6, finish(r3, t3, p3, hi3, lo3, k3));
}

__attribute__((target("fma"), noinline)) void morse8_hw(const double *r2,
                                                       const MorseParams &p,
                                                       double *energy,
                                                       double *fscale) {
  if (p.use_switch != 0) {
    for (int i = 0; i < 8; ++i) {
      morse_term<true>(r2[i], p, energy[i], fscale[i]);
    }
    return;
  }
  alignas(16) double arg[8], ex[8];
  alignas(16) __m128d rr[4];
  const __m128d req = _mm_set1_pd(p.r_eq);
  const __m128d nrange = _mm_set1_pd(-p.range);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    rr[i] = _mm_sqrt_pd(_mm_loadu_pd(r2 + off));
    const __m128d dr = _mm_sub_pd(rr[i], req);
    _mm_store_pd(arg + off, _mm_mul_pd(nrange, dr));
  }
  exp8_hw(arg, ex);
  const __m128d one = _mm_set1_pd(1.0);
  const __m128d depth = _mm_set1_pd(p.depth);
  const __m128d ndepth = _mm_set1_pd(-p.depth);
  const __m128d shift = _mm_set1_pd(p.shift);
  const __m128d two = _mm_set1_pd(p.two_de_a);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    const __m128d exv = _mm_load_pd(ex + off);
    const __m128d md = _mm_sub_pd(one, exv);
    const __m128d dd = _mm_mul_pd(depth, md);
    const __m128d v = _mm_fmadd_pd(dd, md, ndepth);
    _mm_store_pd(energy + off, _mm_sub_pd(v, shift));
    const __m128d d_m1 = _mm_sub_pd(md, one);
    const __m128d num = _mm_mul_pd(_mm_mul_pd(two, md), d_m1);
    _mm_store_pd(fscale + off, _mm_div_pd(num, rr[i]));
  }
}

__attribute__((target("sse2"), noinline)) void morse8_soft(const double *r2,
                                                        const MorseParams &p,
                                                        double *energy,
                                                        double *fscale) {
  if (p.use_switch != 0) {
    for (int i = 0; i < 8; ++i) {
      morse_term<false>(r2[i], p, energy[i], fscale[i]);
    }
    return;
  }
  alignas(16) double arg[8], ex[8];
  alignas(16) __m128d rr[4];
  const __m128d req = _mm_set1_pd(p.r_eq);
  const __m128d nrange = _mm_set1_pd(-p.range);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    rr[i] = _mm_sqrt_pd(_mm_loadu_pd(r2 + off));
    const __m128d dr = _mm_sub_pd(rr[i], req);
    _mm_store_pd(arg + off, _mm_mul_pd(nrange, dr));
  }
  exp8(arg, ex);
  const __m128d one = _mm_set1_pd(1.0);
  const __m128d depth = _mm_set1_pd(p.depth);
  const __m128d ndepth = _mm_set1_pd(-p.depth);
  const __m128d shift = _mm_set1_pd(p.shift);
  const __m128d two = _mm_set1_pd(p.two_de_a);
  for (int i = 0; i < 4; ++i) {
    const int off = i * 2;
    const __m128d exv = _mm_load_pd(ex + off);
    const __m128d md = _mm_sub_pd(one, exv);
    const __m128d dd = _mm_mul_pd(depth, md);
    const __m128d v = fma2_trunc(dd, md, ndepth);
    _mm_store_pd(energy + off, _mm_sub_pd(v, shift));
    const __m128d d_m1 = _mm_sub_pd(md, one);
    const __m128d num = _mm_mul_pd(_mm_mul_pd(two, md), d_m1);
    _mm_store_pd(fscale + off, _mm_div_pd(num, rr[i]));
  }
}

__attribute__((target("fma"), noinline)) static void eam_fma(long n, const double *R,
                                                             const EamAtom *atoms,
                                                             const double *box, double *F,
                                                             double *U) {
#define RGPOT_EAM_HW true
#include "rgpot/simd/eam_body.inc"
#undef RGPOT_EAM_HW
}

__attribute__((target("sse2"), noinline)) void eam_sse(long n, const double *R,
                                                       const EamAtom *atoms,
                                                       const double *box, double *F,
                                                       double *U) {
  if (detail::fma_hw_active()) {
    eam_fma(n, R, atoms, box, F, U);
    return;
  }
#define RGPOT_EAM_HW false
#include "rgpot/simd/eam_body.inc"
#undef RGPOT_EAM_HW
}

__attribute__((target("fma"), noinline)) void exp_hw_n(const double *in,
                                                       double *out,
                                                       std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    const double a = exp_fast<true>(in[i]);
    const double b = exp_fast<true>(in[i + 1]);
    out[i] = a;
    out[i + 1] = b;
  }
  for (; i < n; ++i) {
    out[i] = exp_fast<true>(in[i]);
  }
}

#else

void exp_hw_n(const double *in, double *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = exp(in[i]);
  }
}

#endif

double exp_fixed(double x) { return detail::exp_fixed_lane(x); }

RGPOT_SIMD_CLONES void exp_fixed_n(const double *x, double *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = detail::exp_fixed_lane(x[i]);
  }
}

} // namespace rgpot::simd
