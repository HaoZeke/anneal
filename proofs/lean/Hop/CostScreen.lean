/-!
# Cost-asymmetric screen threshold

After an `S`-step screen, finishing the quench costs `Q = R - S` extra
evaluations. A discarded winner costs one full hop `R`. Net value of
quenching at improvement probability `p` is

  `p R - (1 - p) Q`

which is positive iff `p > Q / (Q + R) = (R - S) / (2 R - S)`.

For the measured hop `S = 25`, `R = 200` this is `7/15`.

No Mathlib; Lean 4 core `grind`.
-/

namespace Hop

/-- Algebraic identity: `Q/(Q+R)` with `Q = R-S` is `(R-S)/(2R-S)`. -/
theorem threshold_identity (S R : Rat) :
    (R - S) / ((R - S) + R) = (R - S) / (2 * R - S) := by
  grind

/-- The measured hop (`S = 25`, `R = 200`) is `7/15`. -/
theorem measured_hop_seven_fifteenths :
    ((200 : Rat) - 25) / (2 * 200 - 25) = (7 : Rat) / 15 := by
  grind

/-- Denominator exceeds numerator by `R`, so the threshold is below one
when `R > 0`. -/
theorem denom_minus_num (S R : Rat) :
    (2 * R - S) - (R - S) = R := by
  grind

/-- Positive extra-quench cost at the measured hop. -/
theorem extra_positive_measured : ((200 : Rat) - 25) > 0 := by
  grind

/-- Threshold in `(0,1)` at the measured hop. -/
theorem measured_in_unit_interval :
    (0 : Rat) < ((200 : Rat) - 25) / (2 * 200 - 25)
      ∧ ((200 : Rat) - 25) / (2 * 200 - 25) < 1 := by
  grind

end Hop
