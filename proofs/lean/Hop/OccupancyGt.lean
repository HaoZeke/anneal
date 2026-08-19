/-!
# Occupancy packing Good--Turing

Leftover-SOAP unseen mass is `p0 = n1 / n`. A newly hatched leftover
well increments both counts, so

  `p0' - p0 = (n - n1) / (n (n + 1))`.

Packing Good--Turing uses the same estimator on leftover-well arrivals
credited to DECAF families. `n1 = 0` forces `p0 = 0`.

Retirement is mixing and packing saturation and leftover dwell and
FunnelModel EI exhausted and the rematched family floor.
CatalogSaturated does not retire. A one-shot leftover nick is not
a dwell. Mixing alone does not retire.

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

/-- Occupancy retire: mixing certified and packing saturation and
leftover dwell and FunnelModel EI exhausted and the family floor.
CatalogSaturated does not retire. -/
def retire (mixing packingSat leftoverDwell eiExhausted floorMet : Bool) : Bool :=
  mixing && packingSat && leftoverDwell && eiExhausted && floorMet

/-- Mixing names a putative. It does not retire without packing
saturation. -/
theorem mixing_alone_does_not_retire (leftoverDwellMet eiMet floorMet : Bool) :
    retire true false leftoverDwellMet eiMet floorMet = false := by
  cases leftoverDwellMet <;> cases eiMet <;> cases floorMet <;> rfl

/-- Fiedler-and-DECAF floor: packing saturation with one rematched
family does not retire when the hop-graph seam names two packing
families. -/
theorem paper_floor_blocks_one_family :
    retire false true true true false = false := by
  rfl

/-- Packing saturation with the family floor, even with leftover dwell
and EI exhausted, does not retire. -/
theorem packing_sat_and_floor_does_not_retire :
    retire false true true true true = false := by
  rfl

/-- Mixing and packing sat and the floor do not retire while leftover
has not dwelt under the ceiling. -/
theorem leftover_unsaturated_does_not_retire :
    retire true true false true true = false := by
  rfl

/-- Mixing and packing sat and leftover dwell do not retire while
FunnelModel EI on seen packings is not exhausted. -/
theorem ei_remaining_does_not_retire :
    retire true true true false true = false := by
  rfl

/-- Mixing plus packing saturation plus leftover dwell plus EI
exhausted plus the floor retires. -/
theorem mixing_and_packing_sat_retires :
    retire true true true true true = true := by
  rfl

end Hop
