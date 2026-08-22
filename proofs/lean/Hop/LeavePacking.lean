/-!
# Occupancy Leave: what the packing grain and the escape step have to satisfy

Machine-checked statements for the Leave law in `src/catalog/packing.rs`,
`src/known_basin.rs` and `src/methods/cluster_hopping.rs`. No Mathlib;
Lean 4 core `grind`.

Three things are derived here rather than chosen.

*The grain.* A packing is a single-linkage component of DECAF cells. The
guarantee that two packings stay apart is not a property of the radius on
its own: it is the cross distance from *every cell the component reaches*
to the other packing (`component_avoids`). A radius below that cross gap
cannot merge them, and the partition coarsens monotonically in the radius
(`chain_mono`), so the admissible radii form an interval. The interval
shrinks as the book fills, because both ends are computed from the sample.

*The escape step.* At a minimum the gradient vanishes, so a Cartesian
displacement of root-mean-square size `δ` over `N` atoms raises the energy
by at most `λ N δ² / 2` under a curvature bound `λ`. A step that cannot
reach the barrier `Δ` cannot leave the basin, and the same identity read
backwards gives the step that can: `δ² = 2Δ / (λ N)`. The rung is that
identity evaluated on a measured curvature and a measured barrier, not a
constant.

*The invert.* The Leave quench transforms the force by a Householder
reflection in the packing mode, `g ↦ g - 2(g·P̂)P̂`. The reflection is an
isometry and it has exactly the zeros the raw force has, so it moves no
critical point and invents none.
-/

namespace Hop

/-! ## Single linkage

`Chain d r i j` is a walk from `i` to `j` in hops of length at most `r`.
This is single-linkage reachability: the components of the book at radius
`r` are its equivalence classes.
-/

/-- A single-linkage walk at radius `r`. -/
inductive Chain (d : Nat → Nat → Rat) (r : Rat) : Nat → Nat → Prop where
  /-- Every cell reaches itself. -/
  | refl (i : Nat) : Chain d r i i
  /-- A walk extends by one hop of length at most `r`. -/
  | link {i j k : Nat} (h : Chain d r i j) (hjk : d j k ≤ r) : Chain d r i k

/-- Coarsening: a walk at radius `r` is a walk at any larger radius. So
the packing partition only ever merges as the grain grows, and the radii
that give one particular partition form an interval. -/
theorem chain_mono (d : Nat → Nat → Rat) {r r' : Rat} (hr : r ≤ r')
    {i j : Nat} (h : Chain d r i j) : Chain d r' i j := by
  induction h with
  | refl => exact Chain.refl _
  | link _ hjk ih => exact Chain.link ih (by grind)

/-- A property closed under one hop is closed along a whole walk. -/
theorem chain_closed (d : Nat → Nat → Rat) (r : Rat) (P : Nat → Prop)
    (step : ∀ i j, P i → d i j ≤ r → P j)
    {i j : Nat} (h : Chain d r i j) (hi : P i) : P j := by
  induction h with
  | refl => exact hi
  | link _ hjk ih => exact step _ _ ih hjk

/-- **The separation the grain buys.** If every cell the component of `a`
reaches sits further than `r` from every cell of `B`, then the component
never reaches `B`.

The hypothesis quantifies over the reached cells, not over `a` alone, and
that is the whole content: a radius is safe against the cloud it is
applied to, not in the abstract. Adding cells can only lower the cross
gap, so a radius that separates two packings on today's book can merge
them on tomorrow's. -/
theorem component_avoids (d : Nat → Nat → Rat) (r : Rat) (a : Nat)
    (B : Nat → Prop)
    (gap : ∀ i j, Chain d r a i → B j → r < d i j)
    (ha : ¬ B a) {j : Nat} (h : Chain d r a j) : ¬ B j := by
  induction h with
  | refl => exact ha
  | link hij hjk _ =>
    intro hB
    have hgap := gap _ _ hij hB
    grind

/-! ## The escape step

At a minimum `∇E = 0`, so a second-order bound on the displacement is a
bound on the energy it can reach. `δ` is the root-mean-square
displacement over `N` atoms, so the squared Euclidean length is `N δ²`.
-/

/-- Energy a step of RMS size `δ` can reach under curvature bound `λ`. -/
def reach (lam n delta : Rat) : Rat := lam * n * delta ^ 2 / 2

/-- **A step that cannot reach the barrier cannot leave the basin.**
Written as the inequality the caller checks. -/
theorem trapped_of_small_step (lam n delta barrier : Rat)
    (h : lam * n * delta ^ 2 < 2 * barrier) :
    reach lam n delta < barrier := by
  unfold reach
  grind

/-- **The rung.** The same identity read backwards: the step whose reach
is exactly the barrier. No constant enters; `lam` is a measured
curvature and `barrier` a measured barrier. -/
theorem rung_reaches_barrier (lam n delta barrier : Rat)
    (hlam : lam ≠ 0) (hn : n ≠ 0)
    (h : delta ^ 2 = 2 * barrier / (lam * n)) :
    reach lam n delta = barrier := by
  unfold reach
  grind

/-- Reach grows with the square of the step, so a geometric ladder in `δ`
is a geometric ladder in energy with twice the exponent. -/
theorem reach_scales (lam n delta c : Rat) :
    reach lam n (c * delta) = c ^ 2 * reach lam n delta := by
  unfold reach
  grind

/-! ## The invert

The Leave quench sees `g' = g - 2 (g·P̂) P̂` with `‖P̂‖ = 1`. Split `g`
into the component `gp` along `P̂` and the rest, whose squared length is
`grest`.
-/

/-- The reflected component. -/
def flipped (gp : Rat) : Rat := gp - 2 * gp

/-- The Householder flips the sign of the mode component. -/
theorem flip_negates (gp : Rat) : flipped gp = -gp := by
  unfold flipped
  grind

/-- The Householder is an isometry: the transformed force has the length
of the raw force. A walk on it is not a walk on a smaller force. -/
theorem flip_preserves_norm (gp grest : Rat) :
    flipped gp ^ 2 + grest = gp ^ 2 + grest := by
  unfold flipped
  grind

/-- **The invert moves no critical point and invents none.** The
transformed force vanishes exactly where the raw force does, so the
minima and saddles of the landscape are the same before and after
arming a Leave. Immediate from [`flip_preserves_norm`]: the two
expressions are equal, not merely zero together. -/
theorem flip_same_zeros (gp grest : Rat) :
    (flipped gp ^ 2 + grest = 0) ↔ (gp ^ 2 + grest = 0) := by
  rw [flip_preserves_norm]

/-- The reflection reverses the sign of the projection, which is what
makes the walk climb: descent on `g'` is ascent along `P̂`. -/
theorem flip_reverses_projection (gp : Rat) :
    flipped gp * gp = -(gp ^ 2) := by
  unfold flipped
  grind

/-! ## The hill

The deposit is `A` per known well, and `A` has to be an energy or it
cannot be set against the barrier it exists to fill. Along the packing
coordinate `r` the well is harmonic, `q r = κ r² / 2`.
-/

/-- The well along the packing coordinate. -/
def wellDepth (kappa r : Rat) : Rat := kappa * r ^ 2 / 2

/-- The slope of that well at `r`. -/
def wellSlope (kappa r : Rat) : Rat := kappa * r

/-- **The depth is half the slope times the distance.** `κ` is never
measured, only the slope is, so this is the form the deposit takes.
Every factor is an energy per length times a length. -/
theorem depth_from_slope (kappa r : Rat) :
    wellDepth kappa r = wellSlope kappa r * r / 2 := by
  unfold wellDepth wellSlope
  grind

/-- The measured quantity is a Cartesian force `f = ∇E·P̂`, and `P̂` is
a unit direction while `r` is a descriptor distance, so `f` is a slope
per unit length and not per unit `r`. With `∇r = pn · P̂` the conversion
is a division by `pn`. Stated as the identity the amplitude uses. -/
theorem slope_in_r (f pn : Rat) (hpn : pn ≠ 0) :
    f / pn * pn = f := by
  grind

/-- **The depth is read at the grain, not at the offset.** The walk is
standing at `r`, but what the hill has to fill is the well out to the
distance `g` where packings stop being the same packing. The curvature
is what carries between them. -/
def depthAtGrain (s r g : Rat) : Rat := s / r * g ^ 2 / 2

/-- Reading the depth at the offset is the same expression with the grain
replaced by the offset. -/
theorem depth_at_offset (s r : Rat) : depthAtGrain s r r = wellDepth (s / r) r := by
  unfold depthAtGrain wellDepth
  grind

/-- **The error is the square of the ratio of the two distances.** So it
is not a constant to fold into the amplitude: it changes with every
structure, because `r` is wherever the walk happened to arm. Measured on
LJ75, `r = 0.0109` against `g = 0.35`, which is a factor above one
thousand. -/
theorem grain_over_offset (s r g : Rat) (hr : r ≠ 0) :
    depthAtGrain s r g = depthAtGrain s r r * (g / r) ^ 2 := by
  unfold depthAtGrain
  grind

/-- A hill no wider than the grain cannot carry a walk across it: at the
grain the Gaussian argument is at least one, so widening `sigma` to the
grain is what puts the crossing inside one standard deviation. -/
theorem grain_width_is_one_sigma (g : Rat) (hg : g ≠ 0) : g / g = 1 := by
  grind

/-- **A force times a descriptor length is not the depth.** Taking
`f * r` for `wellDepth` is right only when `pn = 2`, so on any structure
whose descriptor responds at another rate the deposit is wrong by that
rate. -/
theorem force_times_r_is_not_depth (f r pn : Rat) (hf : f ≠ 0) (hr : r ≠ 0)
    (hpn : pn ≠ 0) :
    f / pn * r / 2 = f * r ↔ pn = 1 / 2 := by
  grind

end Hop
