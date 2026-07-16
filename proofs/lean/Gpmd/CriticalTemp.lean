/-!
# GPMD design claim (T1): critical temperature window

**This is the load-bearing Lean claim for the shipped temperature law.**

Under the pairing model (D6.7 / GPMD Phase III), the state-gain integrand
has the sign of

  `1/θ − 1/2 = (2−θ)/(2θ)`

for `θ > 0`. Hence:

* `G > 0` when `0 < θ < 2`  (local descent window)
* `G = 0` when `θ = 2`
* `G < 0` when `θ > 2`

The shipped operating point is `θ⋆ = 1/2 ∈ (0,2)`, which inverts to

  `T = (1/2) · gap / d`

If this factor/window claim were false, the gap-proportional law would be
unjustified. Elementary density-ratio algebra lives in `DensityRatio.lean`
as a supporting lemma for the pairing model, not as the design claim.

No Mathlib; Lean 4 core `grind`.
-/

namespace Gpmd

/-- Design identity: the T1 pairing factor rewrites as (2−θ)/(2θ). -/
theorem inv_theta_half_factor (theta : Rat) (_hθ : theta ≠ 0) :
    (1 : Rat) / theta - 1 / 2 = (2 - theta) / (2 * theta) := by
  grind

/-- Interior of the descent window: factor positive at θ⋆ = 1/2. -/
theorem factor_pos_at_half :
    (0 : Rat) < (2 - 1 / 2) / (2 * (1 / 2)) := by
  grind

/-- Critical temperature: factor vanishes at θ = 2. -/
theorem factor_zero_at_two : ((2 - 2 : Rat) / (2 * 2) = 0) := by
  grind

/-- Outside the window: factor negative at θ = 4. -/
theorem factor_neg_at_four : (2 - 4 : Rat) / (2 * 4) < 0 := by
  grind

/-- Concrete subcritical sample used by the design law: θ = 1 < 2. -/
theorem factor_pos_at_one :
    (0 : Rat) < (2 - 1) / (2 * 1) := by
  grind

/-- Concrete supercritical sample: θ = 3 > 2. -/
theorem factor_neg_at_three :
    (2 - 3 : Rat) / (2 * 3) < 0 := by
  grind

/-- Shipped θ⋆ = 1/2 lies strictly inside the descent window (0,2). -/
theorem theta_star_in_window : (0 : Rat) < (1 / 2) ∧ (1 : Rat) / 2 < 2 := by
  grind

/-- Trichotomy samples for the design factor (sub / critical / super). -/
theorem factor_window_trichotomy_samples :
    (0 : Rat) < (2 - 1 / 2) / (2 * (1 / 2))
      ∧ (2 - 2 : Rat) / (2 * 2) = 0
      ∧ (2 - 4 : Rat) / (2 * 4) < 0 := by
  grind

/-- A1 inversion: T = θ⋆ · f / d recovers dimensionless θ⋆ when f,d ≠ 0. -/
theorem a1_dimensionless (f d : Rat) (_hf : f ≠ 0) (_hd : d ≠ 0) :
    ((1 / 2 : Rat) * f / d) * d / f = 1 / 2 := by
  grind

end Gpmd
