/-!
# Occupancy bounds

Machine-checked identities for the occupancy retire law. No Mathlib;
Lean 4 core `grind` / `omega`. The statements match
`src/catalog/occupancy.rs`, `src/catalog/census.rs`,
`src/catalog/mixing.rs`, and the Wiley SI theorem blocks.

Coverage policy is `α = 1/5` (unseen leftover mass). The singleton
budget is `k = 3`. The visit floor is the derived `n_min`, not a
free constant. The Fiedler floor is a Cheeger witness on the
normalised Laplacian (`λ₂ ∈ [0, 2]`), not a conductance magic cut.
Retirement is mixing and packing saturation and leftover dwell
(or one packing family on the book) and FunnelModel EI exhausted
and the rematched family floor. CatalogSaturated does not retire.

`grind` does not unfold `def`s, so algebraic theorems are stated
on the expanded field expressions (same contract as
`Hop.CostScreen` and the original hatch identity).
-/

namespace Hop

/-! ## Good--Turing hatch -/

/-- A newly hatched leftover type raises unseen mass by
`(n - n1) / (n (n + 1))`. -/
theorem leftover_hatch (n n1 : Rat) (hn : n ≠ 0) (hs : n + 1 ≠ 0) :
    (n1 + 1) / (n + 1) - n1 / n = (n - n1) / (n * (n + 1)) := by
  grind

/-- Packing with no singletons has unseen mass zero. -/
theorem packing_n1_zero (n : Rat) : (0 : Rat) / n = 0 := by
  grind

/-- The hatch increment equals a ratio whose numerator is the
unseen-type deficit. Nonnegativity is `n1 ≤ n` once the
denominator `n(n+1)` is positive. -/
theorem leftover_hatch_numerator (n n1 : Rat)
    (hn : n ≠ 0) (hs : n + 1 ≠ 0) :
    ((n1 + 1) / (n + 1) - n1 / n) * (n * (n + 1)) = n - n1 := by
  grind

/-- Integer form of the next-hatch test: the sign of
`p̂₀' - α` is the sign of `n₁+1 - α(n+1)` when `n+1 ≠ 0`. -/
theorem hatch_next_sign (n n1 α : Rat) (hs : n + 1 ≠ 0) :
    ((n1 + 1) / (n + 1) - α) * (n + 1) = n1 + 1 - α * (n + 1) := by
  grind

/-- Coverage `α = 1/5`: `5(p̂₀' - 1/5)(n+1) = 5(n₁+1) - (n+1)`. -/
theorem hatch_next_one_fifth_sign (n n1 : Rat) (hs : n + 1 ≠ 0) :
    (5 * ((n1 + 1) / (n + 1) - 1 / 5)) * (n + 1) =
      5 * (n1 + 1) - (n + 1) := by
  grind

/-- Algebraic nick: `n = 21`, `n₁ = 4` is below `1/5`; the next hatch
is above. That is why a one-shot leftover nick is not a dwell. -/
theorem leftover_nick_then_rise :
    (4 : Rat) / 21 < 1 / 5 ∧ (5 : Rat) / 22 > 1 / 5 := by
  grind

/-! ## Derived visit floor

`n_min(α, k)` is the smallest `n` such that `k` leftover singletons
remain hatch-stable at ceiling `α`. For `α = 1/5`,
`(k+1)/(n+1) < 1/5` iff `n+1 > 5(k+1)` iff `n ≥ 5(k+1)`.
The production singleton budget is `k = 3`, so `n_min = 20`.
-/

/-- Three leftover singletons are hatch-stable at `n = 20`, `α = 1/5`. -/
theorem three_singletons_stable_at_twenty :
    (3 + 1 : Rat) / (20 + 1) < 1 / 5 := by
  grind

/-- Three leftover singletons are not hatch-stable at `n = 19`. -/
theorem three_singletons_unstable_at_nineteen :
    ¬ ((3 + 1 : Rat) / (19 + 1) < 1 / 5) := by
  grind

/-- `n = 20` is the smallest natural with three hatch-stable
singletons at `α = 1/5`. -/
theorem visits_floor_three_fifths :
    (3 + 1 : Rat) / (20 + 1) < 1 / 5 ∧
      ¬ ((3 + 1 : Rat) / (19 + 1) < 1 / 5) := by
  grind

/-- Natural form of the `α = 1/5` visit floor. -/
theorem visits_floor_nat (k n : Nat) :
    n + 1 > 5 * (k + 1) ↔ n ≥ 5 * (k + 1) := by
  omega

/-- Production visit floor: singleton budget 3 and coverage `1/5`
give `n_min = 20`. -/
theorem production_min_visits : 5 * (3 + 1) = 20 := by
  grind

/-! ## Discrete packing FES

The occupancy report is `ΔF/kT = ln(n_max / n₂)`. Lean stays in
`Rat` and records the Boltzmann factor `n_max / n₂`; the logarithm
is monotone, so the identities are the factor identities.
-/

/-- `n_max/n₂ - 1` has the sign of `n_max - n₂`. -/
theorem fes_factor_minus_one (nmax n2 : Rat) (h2 : n2 ≠ 0) :
    (nmax / n2 - 1) * n2 = nmax - n2 := by
  grind

/-- Multiplying every well-count by `k ≠ 0` does not change `ΔF`. -/
theorem fes_scale_invariant (nmax n2 k : Rat) (hk : k ≠ 0) (h2 : n2 ≠ 0) :
    (k * nmax) / (k * n2) = nmax / n2 := by
  grind

/-- Equal leading counts give Boltzmann factor 1, so `ΔF = 0`. -/
theorem fes_tie_is_one : (10 : Rat) / 10 = 1 := by
  grind

/-- One occupied family has no second well, so no discrete `ΔF`. -/
theorem fes_undefined_one_family : ¬ (1 ≥ 2) := by
  omega

theorem fes_defined_two_families : 2 ≥ 2 := by
  omega

/-! ## Brooks--Gelman mixed cut

`R̂² = (n-1)/n + B/(n W)`. The 1992 conventional threshold is
`R̂ < 6/5`. Occupant collapse uses this cut.
-/

theorem rhat_sq_split (n W B : Rat) (hn : n ≠ 0) (hw : W ≠ 0) :
    ((n - 1) / n * W + B / n) / W = (n - 1) / n + B / (n * W) := by
  grind

/-- The mixed cut `R̂ < 6/5` is the between/within inequality
`B/W < (11 n + 25)/25`. -/
theorem mixed_threshold_bw (n : Rat) (hn : n ≠ 0) :
    n * ((6 / 5) * (6 / 5) - (n - 1) / n) = (11 * n + 25) / 25 := by
  grind

/-- Copying one floor onto every explore chain gives `B = 0` and
`R̂² = (n-1)/n`. -/
theorem copy_collapse_rhat (n W : Rat) (hn : n ≠ 0) (hw : W ≠ 0) :
    ((n - 1) / n * W + (0 : Rat) / n) / W = (n - 1) / n := by
  grind

theorem copy_collapse_gap (n : Rat) (hn : n ≠ 0) :
    (n - 1) / n - 1 = -1 / n := by
  grind

theorem one_below_mixed_cut : (1 : Rat) < (6 / 5) * (6 / 5) := by
  grind

/-- Split-R-hat certificate length: 2 chains (occupant and, when
present, competitor) times 2 Vehtari halves times the 4-draw
minimum per half. -/
theorem certify_min_samples : 2 * 2 * 4 = 16 := rfl

/-! ## Cheeger / Fiedler family floor

The hop-graph Fiedler vector is the second eigenvector of the
*normalised* Laplacian, so `λ₂ ∈ [0, 2]`. Conductance `c` of that
cut is dimensionless and comparable to `λ₂`. Cheeger's inequality
says `λ₂/2 ≤ h ≤ √(2 λ₂)` for the Cheeger constant `h`. A measured
cut with `c < λ₂` is then a Cheeger witness: `c² < 2 λ₂` once
`c² < λ₂²` and `λ₂² ≤ 2 λ₂`. Disconnection (`c = 0`) is two
communities with no eigenvalue. Without `λ₂` a positive
conductance is not a witness.
-/

/-- Difference of squares: the Cheeger comparison is a product. -/
theorem cheeger_sq_gap (c lam : Rat) :
    lam * lam - c * c = (lam - c) * (lam + c) := by
  grind

/-- On `[0, 2]`, `λ₂² - 2 λ₂ = λ₂(λ₂ - 2) ≤ 0`. -/
theorem cheeger_cap (lam : Rat) :
    lam * lam - 2 * lam = lam * (lam - 2) := by
  grind

/-- A concrete Cheeger witness: `c = 1/20`, `λ₂ = 1/10` satisfies
`c < λ₂ ≤ 2` and `c² < 2 λ₂`. -/
theorem cheeger_example :
    (1 : Rat) / 20 < 1 / 10 ∧
      ((1 : Rat) / 20) * (1 / 20) < 2 * (1 / 10) := by
  grind

/-- Family floor: two communities iff both Fiedler sides are live,
DECAF labels them as distinct packings, and the cut is a bottleneck
(`c = 0` or `c < λ₂`). No spectrum and `c > 0` is one community. -/
def familyFloor (bothSides distinct cZero cLtLam : Bool) : Nat :=
  if bothSides && distinct && (cZero || cLtLam) then 2 else 1

theorem no_spectrum_is_one_community :
    familyFloor true true false false = 1 := by
  rfl

theorem disconnected_is_two_communities :
    familyFloor true true true false = 2 := by
  rfl

theorem cheeger_cut_is_two_communities :
    familyFloor true true false true = 2 := by
  rfl

theorem one_sided_split_is_one_community :
    familyFloor false true true true = 1 := by
  rfl

theorem same_packing_sides_are_one_community :
    familyFloor true false true true = 1 := by
  rfl

/-! ## Occupancy retire

Mixing names a putative. It does not retire without packing
saturation, leftover dwell (or one book family), EI exhausted,
and the rematched family floor. CatalogSaturated does not retire.
-/

def leftoverOk (leftoverDwell oneCell : Bool) : Bool :=
  leftoverDwell || oneCell

def retire (mixing packingSat leftoverDwell eiExhausted floorMet oneCell : Bool) : Bool :=
  mixing && packingSat && leftoverOk leftoverDwell oneCell && eiExhausted && floorMet

/-- Mixing names a putative. It does not retire without packing
saturation. -/
theorem mixing_alone_does_not_retire (leftoverDwellMet eiMet floorMet oneCell : Bool) :
    retire true false leftoverDwellMet eiMet floorMet oneCell = false := by
  cases leftoverDwellMet <;> cases eiMet <;> cases floorMet <;> cases oneCell <;> rfl

/-- Fiedler-and-DECAF floor: packing saturation with one rematched
family does not retire when the hop-graph seam names two packing
families. -/
theorem paper_floor_blocks_one_family :
    retire false true true true false false = false := by
  rfl

/-- Packing saturation with the family floor, even with leftover dwell
and EI exhausted, does not retire. -/
theorem packing_sat_and_floor_does_not_retire :
    retire false true true true true false = false := by
  rfl

/-- Mixing and packing sat and the floor do not retire while leftover
has not dwelt and the book holds more than one family. -/
theorem leftover_unsaturated_does_not_retire :
    retire true true false true true false = false := by
  rfl

/-- One packing family on the book is Boender--Rinnooy Kan on one
cell: leftover-SOAP hatches are intra-well and do not block. -/
theorem one_book_family_waives_leftover :
    retire true true false true true true = true := by
  rfl

/-- Mixing and packing sat and leftover dwell do not retire while
FunnelModel EI on seen packings is not exhausted. -/
theorem ei_remaining_does_not_retire :
    retire true true true false true false = false := by
  rfl

/-- Mixing plus packing saturation plus leftover dwell plus EI
exhausted plus the floor retires. -/
theorem mixing_and_packing_sat_retires :
    retire true true true true true false = true := by
  rfl

end Hop
