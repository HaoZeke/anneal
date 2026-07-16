/-!
# GPMD supporting algebra (I1): Gaussian density-ratio expansion

Supporting lemmas for the pairing model used by the **design claim** in
`CriticalTemp.lean` (T1 critical window). These are not the design claim
themselves: if only these held without the sign-factor window, the
temperature law would not be justified.

No Mathlib; Lean 4 core `grind`.
-/

namespace Gpmd

theorem sq_sub_sq (a b : Int) : a * a - b * b = (a - b) * (a + b) := by
  grind

theorem gauss_num_expand (u μ : Int) :
    (-u - μ) * (-u - μ) - (u - μ) * (u - μ) = 4 * u * μ := by
  grind

theorem gauss_num_expand_rat (u μ : Rat) :
    (-u - μ) * (-u - μ) - (u - μ) * (u - μ) = 4 * u * μ := by
  grind

theorem log_ratio_coeff (μ σ2 : Rat) :
    -(4 * μ) / (2 * σ2) = -(2 * μ) / σ2 := by
  grind

/-- ES specialization: under σ² = 4μ, log-ratio coefficient is −1/2. -/
theorem es_coeff (μ : Rat) (_hμ : μ ≠ 0) :
    -(2 * μ) / (4 * μ) = (-1 : Rat) / 2 := by
  grind

theorem density_ratio_coefficient (μ : Rat) (_hμ : μ ≠ 0) :
    -(4 * μ) / (2 * (4 * μ)) = (-1 : Rat) / 2 := by
  grind

end Gpmd
