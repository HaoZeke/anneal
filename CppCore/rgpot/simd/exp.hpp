#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/exact_fma.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rgpot::simd {

// fdlibm e_exp constants. The scalar body and every vector lane use these
// values in the same order.
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;
inline constexpr double kInvLn2 = 1.44269504088896338700e+00;
inline constexpr double kP1 = 1.66666666666666019037e-01;
inline constexpr double kP2 = -2.77777777770155933842e-03;
inline constexpr double kP3 = 6.61375632143793436117e-05;
inline constexpr double kP4 = -1.65339022054652515390e-06;
inline constexpr double kP5 = 4.13813679705723846039e-08;
inline constexpr double kOverflow = 7.09782712893383973096e+02;
inline constexpr double kUnderflow = -7.45133219101941108420e+02;

template <bool Hw> inline double fad(double a, double b, double c) {
  if constexpr (Hw) {
    // MSVC has no __builtin_fma. The hardware bodies that pass Hw are
    // GCC and Clang, compiled with target("fma").
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_fma(a, b, c);
#else
    return mul_add(a, b, c);
#endif
  } else {
    return fma_quick(a, b, c);
  }
}

template <bool Hw> inline double r2_fast(double dx, double dy, double dz) {
  return fad<Hw>(dz, dz, fad<Hw>(dy, dy, dx * dx));
}

// Inlined exp so a hardware-FMA caller fuses the polynomial instead of
// calling libm once per product-sum. Hw selects the instruction; the
// operations match exp().
template <bool Hw> inline double exp_fast(double x) {
  std::uint64_t ux = 0;
  std::memcpy(&ux, &x, sizeof(ux));
  std::uint32_t hx = static_cast<std::uint32_t>(ux >> 32);
  const int xsb = static_cast<int>((hx >> 31) & 1u);
  hx &= 0x7fffffffu;
  const std::uint32_t lx = static_cast<std::uint32_t>(ux);

  if (hx >= 0x40862e42u) {
    if (hx >= 0x7ff00000u) {
      if (((hx & 0xfffffu) | lx) != 0u) {
        return x + x;
      }
      return xsb == 0 ? x : 0.0;
    }
    if (x > kOverflow) {
      const double huge = 1.0e300;
      return huge * huge;
    }
    if (x < kUnderflow) {
      const double tiny = 9.33263618503218878990e-302;
      return tiny * tiny;
    }
  }

  int k = 0;
  double hi = 0.0;
  double lo = 0.0;
  if (hx > 0x3fd62e42u) {
    if (hx < 0x3ff0a2b2u) {
      const double ln2_hi = xsb ? -kLn2Hi : kLn2Hi;
      const double ln2_lo = xsb ? -kLn2Lo : kLn2Lo;
      hi = x - ln2_hi;
      lo = ln2_lo;
      k = 1 - xsb - xsb;
    } else {
      const double half = xsb ? -0.5 : 0.5;
      double scaled = 0;
      if constexpr (Hw) {
#if defined(__GNUC__) || defined(__clang__)
        scaled = __builtin_fma(kInvLn2, x, half);
#else
        scaled = fma_trunc(kInvLn2, x, half);
#endif
      } else {
        scaled = fma_trunc(kInvLn2, x, half);
      }
      k = static_cast<int>(scaled);
      const double tk = static_cast<double>(k);
      hi = x - tk * kLn2Hi;
      lo = tk * kLn2Lo;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000u) {
    const double huge = 1.0e300;
    if (huge + x > 1.0) {
      return 1.0 + x;
    }
  }

  const double t = x * x;
  double p4 = 0;
  double p3 = 0;
  double p2 = 0;
  double p1 = 0;
  if constexpr (Hw) {
#if defined(__GNUC__) || defined(__clang__)
    p4 = __builtin_fma(t, kP5, kP4);
    p3 = __builtin_fma(t, p4, kP3);
    p2 = __builtin_fma(t, p3, kP2);
    p1 = __builtin_fma(t, p2, kP1);
#else
    p4 = fma_trunc(t, kP5, kP4);
    p3 = fma_trunc(t, p4, kP3);
    p2 = fma_trunc(t, p3, kP2);
    p1 = fma_trunc(t, p2, kP1);
#endif
  } else {
    p4 = fma_trunc(t, kP5, kP4);
    p3 = fma_trunc(t, p4, kP3);
    p2 = fma_trunc(t, p3, kP2);
    p1 = fma_trunc(t, p2, kP1);
  }
  const double c = x - t * p1;
  if (k == 0) {
    const double xc = x * c;
    const double den = c - 2.0;
    const double quot = xc / den;
    return 1.0 - (quot - x);
  }
  const double xc = x * c;
  const double den = 2.0 - c;
  const double quot = xc / den;
  const double y = 1.0 - ((lo - quot) - hi);
  std::uint64_t yb = 0;
  std::memcpy(&yb, &y, sizeof(yb));
  const int ye = static_cast<int>((yb >> 52) & 0x7ff);
  const int ne = ye + k;
  yb = (yb & 0x800fffffffffffffULL) | (static_cast<std::uint64_t>(ne) << 52);
  double out = 0;
  std::memcpy(&out, &yb, sizeof(out));
  return out;
}

/// Scalar exp. Cody-Waite reduction by a split ln(2), then the fdlibm
/// degree-5 polynomial. The large-argument scale and each polynomial
/// product-sum are one rounding. Overflow returns +inf, underflow returns 0,
/// a NaN returns a NaN, and exp(-0) is +1. Vector targets replay these
/// operations per lane.
double exp(double x);

/// Same bits as exp(), for n values. Declared on every target. On x86-64
/// GCC and Clang the body is compiled with a hardware FMA, so call it only
/// where detail::fma_hw_active() holds. Elsewhere it calls exp().
void exp_hw_n(const double *in, double *out, std::size_t n);

#if defined(__x86_64__)
struct PairView;
struct LjParams;
struct MorseParams;
struct EamAtom;
double lj_sse2_entry(const PairView &view, const LjParams &params, double *F,
                     double strain[6]);
double morse_sse2_entry(const PairView &view, const MorseParams &params, double *F,
                        double strain[6]);
void eam_sse2_entry(long n, const double *R, const EamAtom *atoms, const double *box,
                    double *F, double *U);
/// EAM with an SSE2 Dekker product. Same bits as the scalar kernel.
void eam_sse(long n, const double *R, const EamAtom *atoms, const double *box, double *F,
             double *U);
#endif


namespace detail {

// Cody and Waite reduction x = k ln2 + r. kLn2Hi has 32 trailing zero bits,
// so k * kLn2Hi is exact for |k| < 2^21 (fdlibm's split).
inline constexpr double kExpInvLn2 = 0x1.71547652b82fep0;
inline constexpr double kExpLn2Hi = 0x1.62e42feep-1;
inline constexpr double kExpLn2Lo = 0x1.a39ef35793c76p-33;
// Adding 1.5 * 2^52 rounds a double below 2^51 in magnitude to an integer
// held in the low mantissa bits.
inline constexpr double kExpShift = 0x1.8p52;
// Inputs are clamped to [kExpMin, kExpMax]. exp(kExpMax) overflows and
// exp(kExpMin) rounds to zero, so the clamp keeps both limits.
inline constexpr double kExpMax = 710.0;
inline constexpr double kExpMin = -746.0;

// exp(r) ~ 1 + r + r^2 (c2 + c3 r + ... + c11 r^9) on |r| <= 0.3466.
// Sollya fpminimax with double coefficients and the constrained part 1 + r;
// the relative approximation error is below 3.8e-18 (2^-57.8), from
// sollya's supnorm. See docs/orgmode/reference/simd_exp.org.
inline constexpr double kExpC2 = 0x1.000000000000ap-1;
inline constexpr double kExpC3 = 0x1.55555555554fep-3;
inline constexpr double kExpC4 = 0x1.55555555506bcp-5;
inline constexpr double kExpC5 = 0x1.1111111126b36p-7;
inline constexpr double kExpC6 = 0x1.6c16c18477083p-10;
inline constexpr double kExpC7 = 0x1.a01a012fee99cp-13;
inline constexpr double kExpC8 = 0x1.a019995efb00fp-16;
inline constexpr double kExpC9 = 0x1.71df18ef76144p-19;
inline constexpr double kExpC10 = 0x1.28adfccb26cd8p-22;
inline constexpr double kExpC11 = 0x1.ad92e899366cbp-26;

inline std::uint64_t exp_bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

inline double exp_from_bits(std::uint64_t u) {
  double x = 0;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

/// 2^m for an integer-valued m in [-1022, 1023], built in the exponent
/// field. m + 1023 lands in the low bits of the shifted sum.
inline double exp_pow2(double m) {
  return exp_from_bits(exp_bits(m + (kExpShift + 1023.0)) << 52);
}

/// The body every exp in rgpot runs, scalar or as a vector lane. Branch
/// free: GCC and clang vectorise a loop over it. NaN propagates, +inf gives
/// +inf and -inf gives +0.
inline double exp_fixed_lane(double x) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  x = x < kExpMin ? kExpMin : x;
  x = x > kExpMax ? kExpMax : x;
  const double kd = (x * kExpInvLn2 + kExpShift) - kExpShift;
  // x - kd * kExpLn2Hi is exact (Sterbenz) and |r| <= 0.3466.
  const double r = (x - kd * kExpLn2Hi) - kd * kExpLn2Lo;
  double q = kExpC11;
  q = q * r + kExpC10;
  q = q * r + kExpC9;
  q = q * r + kExpC8;
  q = q * r + kExpC7;
  q = q * r + kExpC6;
  q = q * r + kExpC5;
  q = q * r + kExpC4;
  q = q * r + kExpC3;
  q = q * r + kExpC2;
  const double p = 1.0 + (r + (r * r) * q);
  // 2^k as two factors: both stay normal for k in [-1076, 1024], so only
  // the last product rounds, into the subnormal range or to infinity.
  const double k1 = (kd * 0.5 + kExpShift) - kExpShift;
  const double k2 = kd - k1;
  return (p * exp_pow2(k1)) * exp_pow2(k2);
}

} // namespace detail

/// e^x with the same bits on every instruction set and C library. Within
/// 1.24 ulp for normal results (see docs/orgmode/reference/simd_exp.org).
double exp_fixed(double x);

/// out[i] = exp_fixed(x[i]) for i < n. Vectorised and dispatched on the CPU
/// at run time; every variant returns the bits of exp_fixed.
void exp_fixed_n(const double *x, double *out, std::size_t n);

} // namespace rgpot::simd
