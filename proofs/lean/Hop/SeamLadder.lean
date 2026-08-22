/-! # The seam ladder

Measured on LJ75: there is no single-shot exit from the icosahedral
packing. Twenty-four raw quenches dropped along a transformed trajectory
reaching 4.4 times the DECAF road to Marks all return to the floor, and
every displacement Leave is 0 of 16 at any deposit height. What does
reach Marks is plain hopping, 2 of 16 seeds, over about ten thousand
ordinary accepted moves: the crossing is a staged rare event, passed as
a sequence, never as a jump.

So model the road as `k` stages with per-attempt success probability
`p`, and ask what an ensemble of `n` chains buys.

Chains that walk alone: the expected mass reaching stage `k` is
`n * p^k`, exponentially small in the number of stages. That is why the
ensembles never beat the serial baseline: `n` multiplies a vanishing
number.

Chains that reseed from the frontier: when a stuck chain restarts from
the furthest structure any chain has banked, each occupied stage is
restored to population `n` before the next attempt, and the expected
mass arriving at the frontier is `n * p` at every stage -- constant in
`k`, not decaying. The gain over walking alone is `p^(1-k)`, geometric
in the depth of the ladder. This is Grassberger's go-with-the-winners
and the splitting step of forward-flux sampling, keyed here on the DECAF
packing distance from the occupied floor.

And the reseeding does not touch `p`. It changes only where restarts
begin; geometry, quench and Metropolis stay raw. A community-wide
penalty does touch `p`: it multiplies every stage's acceptance by a
factor `s <= 1`, and `road_priced` below is the two-line reason the
paving lost 0 of 16 against 2 of 16 on paired seeds -- pricing the road
prices the crossing.
-/

namespace Hop

/-- Expected frontier mass after `k` stages when every chain must cross
the whole ladder alone. -/
def alone (n p : Rat) : Nat → Rat
  | 0 => n
  | k + 1 => alone n p k * p

/-- Expected arrivals at the frontier when reseeding restores each
occupied stage to population `n` before the attempt. The recursion does
not depend on the depth: that is the whole point. -/
def cloned (n p : Rat) : Nat → Rat
  | 0 => n
  | _ + 1 => n * p

/-- Walking alone is exponential decay in the number of stages: one
factor of `p` per stage, nothing else. Stated on the recursion itself,
which is the form every later bound uses. -/
theorem alone_step (n p : Rat) (k : Nat) :
    alone n p (k + 1) = alone n p k * p := rfl

/-- Reseeding holds the frontier arrival rate constant in the depth. -/
theorem cloned_const (n p : Rat) (k : Nat) : cloned n p (k + 1) = n * p := rfl

theorem alone_nonneg (n p : Rat) (hn : 0 ≤ n) (hp : 0 ≤ p) :
    ∀ k, 0 ≤ alone n p k
  | 0 => by grind [alone]
  | k + 1 => by
    have ih := alone_nonneg n p hn hp k
    have := Rat.mul_nonneg ih hp
    grind [alone]

/-- Mass never grows while walking alone. -/
theorem alone_le_start (n p : Rat) (hn : 0 ≤ n) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    ∀ k, alone n p k ≤ n
  | 0 => by grind [alone]
  | k + 1 => by
    have ih := alone_le_start n p hn hp0 hp1 k
    have hpos := alone_nonneg n p hn hp0 k
    have := Rat.mul_le_mul_of_nonneg_left hp1 hpos
    grind [alone]

/-- **Reseeding dominates at every depth.** One stage in, the two agree;
past that, the lone walker pays `p` again per stage and the reseeded
ensemble does not. -/
theorem cloning_dominates (n p : Rat) (hn : 0 ≤ n) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (k : Nat) : alone n p (k + 1) ≤ cloned n p (k + 1) := by
  have hle := alone_le_start n p hn hp0 hp1 k
  have := Rat.mul_le_mul_of_nonneg_right hle hp0
  grind [alone, cloned]

/-- **Pricing the road prices the crossing.** A penalty standing on every
stage multiplies each attempt's success by `s ≤ 1`, and the mass that
survives the ladder can only fall. Reseeding is not of this form: it
leaves `p` alone. -/
theorem road_priced (n p s : Rat) (hn : 0 ≤ n) (hp : 0 ≤ p) (hs0 : 0 ≤ s)
    (hs1 : s ≤ 1) : ∀ k, alone n (p * s) k ≤ alone n p k
  | 0 => by grind [alone]
  | k + 1 => by
    have ih := road_priced n p s hn hp hs0 hs1 k
    have hps : 0 ≤ p * s := Rat.mul_nonneg hp hs0
    have hup := alone_nonneg n p hn hp k
    -- alone (p*s) k * (p*s) <= alone p k * (p*s) <= alone p k * p
    have hstep := Rat.mul_le_mul_of_nonneg_right ih hps
    have hshrink : p * s ≤ p * 1 := Rat.mul_le_mul_of_nonneg_left hs1 hp
    have := Rat.mul_le_mul_of_nonneg_left hshrink hup
    grind [alone]

end Hop
