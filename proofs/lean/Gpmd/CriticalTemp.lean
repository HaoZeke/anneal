/-!
# GPMD (T1): critical temperature factor algebra

Phase III / D6.1 factor in the pairing integrand. **No Mathlib.**

The pairing identity reduces the sign of `G(c,θ)` to the sign of
`e^{-u/2} - e^{-u/θ}`, which for `u>0` equals the sign of
`1/θ - 1/2 = (2-θ)/(2θ)`.

Machine-checked here:

1. Algebraic identity `1/θ − 1/2 = (2−θ)/(2θ)` for `θ ≠ 0`
2. Sign samples at the operating point, critical point, and supercritical point
3. `θ⋆ = 1/2 ∈ (0,2)`
4. Dimensionless check of operating law (A1)
-/

namespace Gpmd

/-- Factor identity used in the T1 pairing argument. -/
theorem inv_theta_half_factor (theta : Rat) (_hθ : theta ≠ 0) :
    (1 : Rat) / theta - 1 / 2 = (2 - theta) / (2 * theta) := by
  grind

/-- At the shipped operating point `θ⋆ = 1/2` the factor is positive. -/
theorem factor_pos_at_half :
    (0 : Rat) < (2 - 1 / 2) / (2 * (1 / 2)) := by
  grind

/-- At the critical temperature `θ = 2` the factor vanishes. -/
theorem factor_zero_at_two : ((2 - 2 : Rat) / (2 * 2) = 0) := by
  grind

/-- Supercritical sample: `θ = 4` makes the factor negative. -/
theorem factor_neg_at_four : (2 - 4 : Rat) / (2 * 4) < 0 := by
  grind

/-- Operating constant is strictly inside the descent window `(0,2)`. -/
theorem theta_star_in_window : (0 : Rat) < (1 / 2) ∧ (1 : Rat) / 2 < 2 := by
  grind

/-- Dimensionless inversion of A1: `T = θ⋆ f / d` with `θ⋆ = 1/2`
    recovers `T d / f = 1/2` when `f, d ≠ 0`. -/
theorem a1_dimensionless (f d : Rat) (_hf : f ≠ 0) (_hd : d ≠ 0) :
    ((1 / 2 : Rat) * f / d) * d / f = 1 / 2 := by
  grind

end Gpmd
