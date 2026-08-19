/-!
# Occupancy packing Good--Turing

Leftover-SOAP unseen mass is `p0 = n1 / n`. A newly hatched leftover
well increments both counts, so

  `p0' - p0 = (n - n1) / (n (n + 1))`.

Packing Good--Turing uses the same estimator on leftover-well arrivals
credited to DECAF families. `n1 = 0` forces `p0 = 0`.

Retirement is mixing or catalog-saturated, and packing saturation, and
the rematched family floor. Mixing alone does not retire.

No Mathlib; Lean 4 core `grind`.
-/

namespace Hop

/-- A newly hatched leftover type raises unseen mass by
`(n - n1) / (n (n + 1))`. -/
theorem leftover_hatch (n n1 : Rat) (hn : n ≠ 0) (hs : n + 1 ≠ 0) :
    (n1 + 1) / (n + 1) - n1 / n = (n - n1) / (n * (n + 1)) := by
  grind

/-- Packing with no singletons has unseen mass zero. -/
theorem packing_n1_zero (n : Rat) : (0 : Rat) / n = 0 := by
  grind

/-- Algebraic nick: `n = 21`, `n1 = 4` is below `1/5`; the next hatch
is above. -/
theorem leftover_nick_then_rise :
    (4 : Rat) / 21 < 1 / 5 ∧ (5 : Rat) / 22 > 1 / 5 := by
  grind

/-- Occupancy retire: mixing certified and packing saturation and the
family floor. CatalogSaturated does not retire. -/
def retire (mixing packingSat floorMet : Bool) : Bool :=
  mixing && packingSat && floorMet

/-- Mixing names a putative. It does not retire without packing
saturation. -/
theorem mixing_alone_does_not_retire (floorMet : Bool) :
    retire true false floorMet = false := by
  cases floorMet <;> rfl

/-- Paper floor: packing saturation with one rematched family does not
retire when `F = 2`. -/
theorem paper_floor_blocks_one_family :
    retire false true false = false := by
  rfl

/-- Packing saturation with the family floor does not retire. -/
theorem packing_sat_and_floor_does_not_retire :
    retire false true true = false := by
  rfl

/-- Mixing plus packing saturation plus the floor retires. -/
theorem mixing_and_packing_sat_retires :
    retire true true true = true := by
  rfl

end Hop
