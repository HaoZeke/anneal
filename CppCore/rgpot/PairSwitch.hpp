#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief C^2 quintic switch that takes a pair term smoothly to zero at its
 * cutoff.
 *
 * With @f$t = (r - r_\mathrm{on}) / (r_c - r_\mathrm{on})@f$ clamped to
 * @f$[0, 1]@f$,
 * @f[
 *   S(t) = 1 - 10 t^3 + 15 t^4 - 6 t^5,\qquad
 *   S'(t) = -30 t^2 (1 - t)^2 .
 * @f]
 * The coefficients are the unique quintic with @f$S(0) = 1@f$,
 * @f$S'(0) = S''(0) = 0@f$ and @f$S(1) = S'(1) = S''(1) = 0@f$, so
 * @f$V(r) S(t(r))@f$ and its first two derivatives are continuous at
 * @f$r_\mathrm{on}@f$ and vanish at @f$r_c@f$ for any smooth @f$V@f$.
 * scripts/validation/pair_kernels.py derives them and checks the C^2
 * conditions symbolically; scripts/validation/quintic_switch.sollya bounds
 * the binary64 evaluation error of the Horner forms used here.
 *
 * The clamp is a min/max pair, so evaluation carries no branch: below
 * @f$r_\mathrm{on}@f$ it returns @f$S = 1@f$, @f$dS/dr = 0@f$ exactly.
 */

#include <algorithm>
#include <stdexcept>

namespace rgpot {

struct QuinticSwitch {
  double r_on{0.0};      //!< Where the switch starts (Angstrom).
  double inv_width{0.0}; //!< 1 / (cutoff - r_on).

  /// Switch over the last @p width Angstrom below @p cutoff. Throws
  /// std::invalid_argument unless 0 < width <= cutoff.
  static QuinticSwitch endingAt(double cutoff, double width) {
    if (!(width > 0.0) || !(width <= cutoff)) {
      throw std::invalid_argument(
          "QuinticSwitch: switch width must satisfy 0 < width <= cutoff");
    }
    return {cutoff - width, 1.0 / width};
  }

  struct Value {
    double s;    //!< S(t(r)).
    double dsdr; //!< dS/dr.
  };

  [[nodiscard]] Value operator()(double r) const noexcept {
    const double t = std::min(std::max((r - r_on) * inv_width, 0.0), 1.0);
    const double t2 = t * t;
    const double omt = 1.0 - t;
    return {1.0 + t2 * t * (-10.0 + t * (15.0 - 6.0 * t)),
            -30.0 * t2 * omt * omt * inv_width};
  }

  /// @f$d^2S/dr^2 = -60 t (1 - t) (1 - 2t) / w^2@f$, 0 outside the switch.
  [[nodiscard]] double d2sdr2(double r) const noexcept {
    const double t = std::min(std::max((r - r_on) * inv_width, 0.0), 1.0);
    return -60.0 * t * (1.0 - t) * (1.0 - 2.0 * t) * inv_width * inv_width;
  }
};

} // namespace rgpot
