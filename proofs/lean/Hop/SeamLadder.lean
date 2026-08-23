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

/-! ## The methodology, derived

The lemmas above say why the failed designs failed. The four below are
the design itself: each theorem's hypotheses are a rule the
implementation must keep, and the conclusion is what the search is owed
in return. The whole scheme is: keep a monotone bank of the furthest
structures reached, restart exhausted chains from it, mix the restart
band so one poisoned representative cannot stall the ladder, and share
the bank when there is more than one chain.

Expectations are the mean-field recursions, in `Rat`, as everywhere in
`Hop`: no measure theory is needed to compare two budgets. -/

/-- Expected depth of one episode that must start from the floor: band
one with probability `p`, band two with `p^2`, and so on out to a ladder
of depth `k`. -/
def aloneDepth (p : Rat) : Nat → Rat
  | 0 => 0
  | k + 1 => aloneDepth p k + alone 1 p (k + 1)

/-- **Restarting from the floor is depth-bounded.** However deep the
ladder and however large the budget, an episode from the floor is worth
at most `p / (1 - p)` bands in expectation -- stated multiplied out so
no division enters. Every restart the search already had (symmetrise on
stall, tabu, random reseed) is of this form, which is why none of them
scaled: the budget bought more episodes, and each episode was worth the
same bounded depth. -/
theorem alone_depth_bounded (p : Rat) (hp0 : 0 ≤ p) :
    ∀ k, aloneDepth p k * (1 - p) + alone 1 p (k + 1) ≤ p
  | 0 => by
    show aloneDepth p 0 * (1 - p) + alone 1 p 1 ≤ p
    rw [show aloneDepth p 0 = (0 : Rat) from rfl,
      show alone 1 p 1 = 1 * p from rfl]
    grind
  | k + 1 => by
    have ih := alone_depth_bounded p hp0 k
    have hstep : alone 1 p (k + 2) = alone 1 p (k + 1) * p := rfl
    have hd : aloneDepth p (k + 1) = aloneDepth p k + alone 1 p (k + 1) := rfl
    -- The two nonlinear terms the step introduces cancel exactly:
    -- (D + A)(1 - p) + A p = D(1 - p) + A.
    have key : aloneDepth p (k + 1) * (1 - p) + alone 1 p (k + 2)
        = aloneDepth p k * (1 - p) + alone 1 p (k + 1) := by
      rw [hd, hstep]; grind
    grind

/-- Expected frontier depth of the ratchet: a monotone bank never gives
a band back, so every episode adds its advance probability `q` and
nothing subtracts. -/
def ratchet (q : Rat) : Nat → Rat
  | 0 => 0
  | m + 1 => ratchet q m + q

theorem ratchet_monotone (q : Rat) (hq : 0 ≤ q) (m : Nat) :
    ratchet q m ≤ ratchet q (m + 1) := by
  grind [ratchet]

/-- **The crossover.** Once the episode count `m` satisfies
`p ≤ ratchet q m * (1 - p)` -- that is, `m * q * (1 - p) ≥ p` -- the
monotone bank's expected depth passes the floor-restart bound at every
ladder depth, and it keeps growing linearly while the bound stands
still. The hypotheses are the budget rule: episodes short enough that
`m` is large, long enough that `q` stays positive. -/
theorem crossover (p q : Rat) (k m : Nat) (hp0 : 0 ≤ p) (hp1 : p < 1)
    (halone : aloneDepth p k * (1 - p) + alone 1 p (k + 1) ≤ p)
    (hm : p ≤ ratchet q m * (1 - p)) :
    aloneDepth p k ≤ ratchet q m := by
  have hone : (0 : Rat) < 1 - p := by grind
  have hpos := alone_nonneg 1 p (by grind) hp0 (k + 1)
  have hchain : aloneDepth p k * (1 - p) ≤ ratchet q m * (1 - p) := by grind
  exact Rat.le_of_mul_le_mul_right hchain hone

/-- **A poisoned frontier cannot stall a mixed restart.** Reseeding only
ever from the top band trusts one structure: if its forward probability
is zero the ladder halts however many episodes remain. Splitting the
restart, weight `1 - e` on the frontier and `e` on another occupied
band, keeps the advance probability positive whenever any banked band
can still move -- the frontier term can contribute nothing and the mix
still advances. -/
theorem eps_greedy_positive (e qt qa : Rat) (he0 : 0 < e) (he1 : e ≤ 1)
    (hqt : 0 ≤ qt) (hqa : 0 < qa) :
    0 < (1 - e) * qt + e * qa := by
  have hleft : 0 ≤ (1 - e) * qt := Rat.mul_nonneg (by grind) hqt
  have hright : 0 < e * qa := Rat.mul_pos he0 hqa
  grind

/-- Advance probability of one round in which `n` chains each attempt
the frontier of one shared bank: the round advances unless every
attempt fails. -/
def sharedRound (q : Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => sharedRound q n + q * (1 - sharedRound q n)

theorem sharedRound_le_one (q : Rat) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    ∀ n, sharedRound q n ≤ 1
  | 0 => by grind [sharedRound]
  | n + 1 => by
    have ih := sharedRound_le_one q hq0 hq1 n
    have hgap : 0 ≤ 1 - sharedRound q n := by grind
    have := Rat.mul_le_mul_of_nonneg_right hq1 hgap
    grind [sharedRound]

/-- **Chains help exactly when the bank is shared.** With every chain
offering into one bank, a round of `n + 1` chains advances at least as
often as a round of `n`, and never less often than a single chain:
`monotone` and `at least one` together are the corrected form of "more
chains should reach Marks sooner". Without the shared bank each chain
is its own `aloneDepth`, and `alone_depth_bounded` caps them all
independently -- which is the ensemble that kept losing. -/
theorem sharing_monotone (q : Rat) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (n : Nat) :
    sharedRound q n ≤ sharedRound q (n + 1) := by
  have hle := sharedRound_le_one q hq0 hq1 n
  have hgap : 0 ≤ 1 - sharedRound q n := by grind
  have := Rat.mul_nonneg hq0 hgap
  grind [sharedRound]

theorem sharing_at_least_one (q : Rat) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    ∀ n, q ≤ sharedRound q (n + 1)
  | 0 => by grind [sharedRound]
  | n + 1 => by
    have ih := sharing_at_least_one q hq0 hq1 n
    have hle := sharedRound_le_one q hq0 hq1 (n + 1)
    have hgap : 0 ≤ 1 - sharedRound q (n + 1) := by grind
    have := Rat.mul_nonneg hq0 hgap
    grind [sharedRound]

/-! ## The excursion exponent

The measured crossing is an uninterrupted excursion of about
thirty-five accepted steps, so its rate is a product of per-step
acceptances: `alone 1 q k`. Products split, and the split settles two
questions this campaign was about to spend runs on.

Interruptions multiply each step's survival by `1 - r`, so a
checkpoint reset rate `r` suppresses the crossing by
`alone 1 (1 - r) k` exactly: at one reset per five hundred hops and a
thirty-five step excursion that factor is about 0.93, and no
interruption at that slice can produce the tenfold suppression the
coordinated runs show.

A per-step shave of the acceptance itself is another matter: `q' ≤ c q`
loses `c^k` of the crossing rate, and six percent per step at
thirty-five steps is an order of magnitude. The crossing rate is
hypersensitive to the acceptance law and insensitive to interruptions,
so the only coordination that is safe by construction is one that
leaves the acceptance law of the serial baseline byte-identical -- and
the ensemble path forces `budget_window`, an adaptive temperature the
serial baseline runs without. -/

/-- Products of rates split across an excursion. -/
theorem alone_mul_split (q s : Rat) : ∀ k, alone 1 (q * s) k = alone 1 q k * alone 1 s k
  | 0 => by grind [alone]
  | k + 1 => by
    have ih := alone_mul_split q s k
    grind [alone]

/-- The crossing rate is monotone in the per-step acceptance. -/
theorem alone_mono (a b : Rat) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a ≤ b) :
    ∀ k, alone 1 a k ≤ alone 1 b k
  | 0 => by grind [alone]
  | k + 1 => by
    have ih := alone_mono a b ha hb hab k
    have hbn := alone_nonneg 1 b (by grind) hb k
    have h1 := Rat.mul_le_mul_of_nonneg_right ih ha
    have h2 := Rat.mul_le_mul_of_nonneg_left hab hbn
    grind [alone]

/-- **A per-step shave exponentiates.** Any mechanism that leaves the
walker only `c q` of its per-step acceptance loses `alone 1 c k` of the
crossing rate -- the whole factor, at every excursion length. -/
theorem per_step_amplifies (q c : Rat) (hq : 0 ≤ q) (hc : 0 ≤ c) (k : Nat) :
    alone 1 (c * q) k = alone 1 c k * alone 1 q k :=
  alone_mul_split c q k

/-- The suppressed rate never exceeds the shaved bound: the conviction
direction, for a mechanism only known to satisfy `q' ≤ c q`. -/
theorem shave_convicts (q q' c : Rat) (hq : 0 ≤ q) (hq' : 0 ≤ q') (hc : 0 ≤ c)
    (hle : q' ≤ c * q) (k : Nat) :
    alone 1 q' k ≤ alone 1 c k * alone 1 q k := by
  have hcq : 0 ≤ c * q := Rat.mul_nonneg hc hq
  have hmono := alone_mono q' (c * q) hq' hcq hle k
  have hsplit := alone_mul_split c q k
  grind

end Hop
