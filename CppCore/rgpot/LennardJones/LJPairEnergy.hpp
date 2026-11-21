#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file LJPairEnergy.hpp
 * @brief The pair energy of LJPot as a function of the squared distance,
 * over a scalar template.
 *
 * With @f$a = (\psi^2 / s)^3@f$ the bare term is @f$4 u_0 a (a - 1)@f$.
 * Unswitched, the constant shift is subtracted; switched, the term is
 * multiplied by the quintic of PairSwitch.hpp at @f$r = \sqrt{s}@f$. The
 * parameters are the @c simd::LjParams the force kernel runs on.
 *
 * With @c double this is the energy @c forceImpl sums for one pair. With
 * @c ad::Dual2 seeded by @c ad::seed2(s) it carries @f$g_s@f$ and
 * @f$g_{ss}@f$ as well, which PairHessian.hpp turns into the pair's
 * Hessian block. The switch is clamped on the value, so outside
 * @f$[r_\mathrm{on}, r_c]@f$ the derivatives are those of the constant
 * factor, as for the closed form.
 */

#include <cmath>

#include "rgpot/Dual.hpp"
#include "rgpot/simd/pods.hpp"

namespace rgpot::lj {

template <typename S> S pairEnergy(const S &r2, const simd::LjParams &p) {
  using std::sqrt;
  const S inv_r2 = 1.0 / r2;
  const S sr2 = p.psi2 * inv_r2;
  const S sr4 = sr2 * sr2;
  const S a = sr4 * sr2;
  const S b = p.four_u0 * a;
  const S v = b * (a - 1.0);
  if (p.use_switch == 0) {
    return v - p.shift;
  }
  const S t = (sqrt(r2) - p.r_on) * p.inv_width;
  if (ad::value(t) <= 0.0) {
    return v;
  }
  if (ad::value(t) >= 1.0) {
    return S(0.0);
  }
  const S s = 1.0 + t * t * t * (-10.0 + t * (15.0 - 6.0 * t));
  return v * s;
}

} // namespace rgpot::lj
