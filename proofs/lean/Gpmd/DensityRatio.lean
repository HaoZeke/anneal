/-!
# GPMD (I1): Gaussian log-density ratio algebra

Phase III / D6.6 algebraic core. **No Mathlib.**

Machine-checked steps matching `docs/derivations/gpmd_algorithm.org` (I1):

1. Difference of squares: `a² − b² = (a−b)(a+b)`
2. Gaussian numerator: `(-u−μ)² − (u−μ)² = 4 u μ`
3. General log-ratio coefficient after canceling the `4u`
4. ES specialization: when `σ² = 4 μ` and `μ ≠ 0`, `-2 μ / σ² = -1/2`

Proofs use Lean 4 core `grind` (CommRing / field arithmetic), not
`native_decide` on a pre-simplified rational constant.
-/

namespace Gpmd

/-- Difference of squares over `Int`. -/
theorem sq_sub_sq (a b : Int) : a * a - b * b = (a - b) * (a + b) := by
  grind

/-- Numerator identity in the Gaussian log-density ratio:
    `(-u-μ)² - (u-μ)² = 4 u μ`. -/
theorem gauss_num_expand (u μ : Int) :
    (-u - μ) * (-u - μ) - (u - μ) * (u - μ) = 4 * u * μ := by
  grind

/-- Same identity over `Rat` (used for the coefficient specialization). -/
theorem gauss_num_expand_rat (u μ : Rat) :
    (-u - μ) * (-u - μ) - (u - μ) * (u - μ) = 4 * u * μ := by
  grind

/-- After dividing by `2 σ²`, the log-ratio coefficient of `u` is `-2 μ / σ²`. -/
theorem log_ratio_coeff (μ σ2 : Rat) :
    -(4 * μ) / (2 * σ2) = -(2 * μ) / σ2 := by
  grind

/-- ES limit specialization: `μ = c²`, `σ² = 4 c² = 4 μ`.
    Coefficient of `u` in the log-ratio is exactly `-1/2`. -/
theorem es_coeff (μ : Rat) (_hμ : μ ≠ 0) :
    -(2 * μ) / (4 * μ) = (-1 : Rat) / 2 := by
  grind

/-- Combined I1 coefficient under ES law `σ² = 4 μ` (μ ≠ 0). -/
theorem density_ratio_coefficient (μ : Rat) (_hμ : μ ≠ 0) :
    -(4 * μ) / (2 * (4 * μ)) = (-1 : Rat) / 2 := by
  grind

end Gpmd
