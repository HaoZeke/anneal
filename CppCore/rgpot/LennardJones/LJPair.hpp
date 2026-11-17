#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared 12-6 Lennard-Jones pair term and parameter packing used by
 * LJPot and LJClusterPot.
 */

#include "rgpot/PairSwitch.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exact_fma.hpp"

#include <cmath>

namespace rgpot {

inline simd::LjParams makeLjParams(double u0, double psi, double shift,
                                   double switch_width,
                                   const QuinticSwitch &sw) noexcept {
  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = shift;
  params.use_switch = 0;
  if (switch_width != 0.0) {
    params.use_switch = 1;
    params.r_on = sw.r_on;
    params.inv_width = sw.inv_width;
  }
  return params;
}

inline void evalLjPair(double r2, const simd::LjParams &p, double &energy,
                       double &fscale) noexcept {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = p.psi2 * inv_r2;
  const double sr4 = sr2 * sr2;
  const double a = sr4 * sr2;
  const double b = p.four_u0 * a;
  const double a_m1 = a - 1.0;
  const double v = b * a_m1;
  const double two_a = 2.0 * a;
  const double two_a_m1 = two_a - 1.0;
  const double six_b = 6.0 * b;
  const double f_un = (six_b * inv_r2) * two_a_m1;
  if (p.use_switch == 0) {
    energy = v - p.shift;
    fscale = f_un;
    return;
  }
  const double r = std::sqrt(r2);
  const double t_raw = (r - p.r_on) * p.inv_width;
  const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double six_t = 6.0 * t;
  const double inner = 15.0 - six_t;
  const double poly = simd::fma_quick(t, inner, -10.0);
  const double s = simd::fma_quick(t3, poly, 1.0);
  const double omt = 1.0 - t;
  const double ds2 = (-30.0 * t2) * omt * omt;
  const double dsdr = ds2 * p.inv_width;
  energy = v * s;
  fscale = f_un * s - (v * dsdr) / r;
}

} // namespace rgpot
