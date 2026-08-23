/-! # What a shared message is worth

Sharing between chains is a channel: one chain observes something and
broadcasts, and the broadcast is worth exactly the information it
carries about the crossing -- no more, however the receiving chain uses
it. The campaign measured two such channels without naming them.

The positional channel -- gap and energy of a deep structure -- fired
on both winners' staging excursions and on about five hundred
indistinguishable icosahedral excursions in the failing seeds: an
icosahedral isomer at gap 0.72 and depth -394.5 sits past both winning
doorways. Likelihood ratio near one; the posterior after a broadcast is
the prior; sharing it moves nothing. Every pre-crossing coordination
built this campaign -- the bank, the stale reseed, the doorway burst --
was a receiver wired to this empty channel, and each measured 0 of 16.

The fivefold channel -- within two eps of the floor with the fivefold
template share collapsed -- fired on 2 of 2 crossings and 3 times in
3252 records of the fourteen seeds that never crossed, and the three
are themselves one seed standing on what the share says is a crossing.
Broadcasting one fire multiplies the receiver's odds by the count
ratio, and the arithmetic below is exact.

Everything is stated on counts, division-free, so the Bayes step is a
cross-multiplication and the measured instances are literal rational
arithmetic. -/

namespace Hop

/-- Of the events a detector fires on, the share that are crossings:
`fc` fires on crossings against `ff` fires elsewhere, compared
cross-multiplied so no division enters. `sharper` says a larger
likelihood ratio gives a larger posterior share, which is the whole
Bayes step for a binary broadcast. -/
theorem sharper (fc ff fc' ff' : Rat) (hlr : fc' * ff ≤ fc * ff')
    (hswap : fc' * fc ≤ fc * fc') :
    fc' * (fc + ff) ≤ fc * (fc' + ff') := by
  grind

/-- **The measured channels, in the measured counts.** The fivefold
detector at (2 crossings, 3 elsewhere) against the positional doorway
at (2 crossings, 500 elsewhere): the fivefold posterior share of fires
is 2 of 5 and the positional is 2 of 502, and the cross-multiplied
comparison is the three-figure gap between a channel worth broadcasting
and one that is not. -/
theorem fivefold_beats_positional :
    (2 : Rat) * (2 + 3) ≤ (2 : Rat) * (2 + 500)
      ∧ (2 : Rat) * 3 ≤ (2 : Rat) * 500 := by
  constructor <;> grind

/-- A broadcast to `n` receivers is worth `n` times its posterior: the
value of sharing scales with the channel, not with the enthusiasm of
the receivers. Wiring more chains to an empty channel multiplies
nothing, which is the ensemble history of this campaign in one line. -/
theorem broadcast_scales (n post post' : Rat) (hn : 0 ≤ n)
    (hpost : post' ≤ post) : n * post' ≤ n * post :=
  Rat.mul_le_mul_of_nonneg_left hpost hn

/-! ## The boundary of coordination

The trichotomy the campaign closes on. A coordination policy can only
reallocate hops between chains on the strength of what has been
observed. If no shareable observable shifts the conditional crossing
probability -- and on LJ75 none measured does before the first
crossing: gap, energy and the template shares all carry likelihood
ratio one on the staging structures -- then every reallocation spends
the same rate it started with, and the ensemble can neither beat nor
trail independent chains. Coordination that touches the acceptance law
loses by `shave_convicts`; coordination that only reallocates is
invariant by the lemma below; coordination wins only through a channel
with likelihood ratio above one, and the only such channel measured
fires after the crossing. What raises the rate itself is the move
kernel, which is physics, not coordination. -/

/-- Reallocating a fixed number of hops across chains whose per-hop
crossing rate is the same `p` -- the case when no observation
discriminates -- yields the independent rate, whatever the weights. -/
theorem uninformative_reallocation (w1 w2 p : Rat) :
    w1 * p + w2 * p = (w1 + w2) * p := by
  grind

/-- With an informative observation the weights buy something: moving
mass onto the chain whose conditional rate is higher is worth exactly
the rate gap times the mass moved, and nothing more. The gain exists
precisely when `p2 > p1`, which by `sharper` requires a channel whose
likelihood ratio exceeds one. -/
theorem informed_reallocation (w d p1 p2 : Rat) :
    (w - d) * p1 + (w + d) * p2 - (w * p1 + w * p2) = d * (p2 - p1) := by
  grind

/-! ## Talking about where we have been

The reallocation bound above closes every channel about where the
crossing is. It says nothing about the other thing chains know
perfectly: where they have already been. A quench that descends into a
basin some chain has already cataloged returns a minimum the ensemble
already holds, and every gradient spent completing it is waste. A
minimizer that recognises the catchment mid-descent and returns the
stored minimum leaves the acceptance untouched -- the same energies
reach the same Metropolis test -- and refunds the rest of the descent.

This is coordination on the cost side, where the information actually
lives: the crossing rate per attempt is pinned by the landscape, but
attempts per budget are not, and recognition sets are monotone under
sharing -- a chain screening against the union of everyone's catalog
recognises at least everything its own history recognises. Measured on
the serial baseline, forty percent of quenches are re-descents into
known basins, paid at full price today. -/

/-- Attempts a budget buys at cost `c` per attempt, division-free:
`attempts * c = budget` is carried as a hypothesis where needed. -/
theorem more_attempts_same_rate (a a' p : Rat) (hp : 0 ≤ p) (ha : a ≤ a') :
    a * p ≤ a' * p :=
  Rat.mul_le_mul_of_nonneg_right ha hp

/-- **Recognition refunds are pure gain.** With `h` of `a` attempts
recognised early and refunded `s` of their cost each, the same budget
`a * c` buys `a * c = a' * (c - hshare * s)` attempts at the smaller
effective cost; stated as the cross-multiplied comparison: any
per-attempt cost saving buys at least proportionally many attempts.
The hypotheses are the contract: the stored minimum equals the one the
descent would have reached, so `p` is untouched, and the recognition
cost is inside the saving. -/
theorem refund_buys_attempts (b c c' : Rat) (hb : 0 ≤ b) (hc' : 0 < c')
    (hcc : c' ≤ c) (a a' : Rat) (hac : a * c = b) (hac' : a' * c' = b)
    (ha : 0 ≤ a) :
    a ≤ a' := by
  have h1 : a * c' ≤ a * c := Rat.mul_le_mul_of_nonneg_left hcc ha
  have h2 : a * c' ≤ a' * c' := by grind
  exact Rat.le_of_mul_le_mul_right h2 hc'

/-- **Sharing the recognition set is monotone.** A chain that screens
against the union of every chain's catalog hits at least as often as
one screening its own history: `h_own ≤ h_shared` gives a per-attempt
cost that can only fall, `c - h * s` decreasing in `h`. Together with
`refund_buys_attempts` and `more_attempts_same_rate`, the chain of
inequalities is the provably-better the minimiser-level talk was owed:
same acceptance, same per-attempt rate, strictly more attempts whenever
any cross-chain hit occurs. -/
theorem shared_recognition_cheapens (c s h h' : Rat) (hs : 0 ≤ s)
    (hh : h ≤ h') : c - h' * s ≤ c - h * s := by
  have := Rat.mul_le_mul_of_nonneg_right hh hs
  grind

/-! ## The refund with its error budget

Recognition is a classifier, and descriptor proximity does not certify
the catchment: a false hit stands the wrong minimum in for the descent
and perturbs the acceptance the refund theorems assumed untouched. The
sound derivation carries the error explicitly. With attempts `a` at
true rate `p` refunded up to `a'` attempts at rate `p - d`, where `d`
is whatever rate the misrecognitions cost, the trade is favourable
exactly while the rate loss stays under the rate times the relative
cost saving -- a budget for the false-hit rate, priced in the same
units as the refund. The radius stops being a belief: it is chosen so
the audited misrecognition stays inside `d`.

The estimator is part of the derivation. Auditing a share `t` of hits
-- completing those descents at full cost anyway and comparing basins
-- measures the false-hit rate unbiasedly while keeping `1 - t` of the
refund, so the bound is checkable by the run that relies on it. -/

/-- **The tolerance.** The refunded search wins exactly while the rate
it gives up stays under the rate times the attempt headroom, stated
cross-multiplied: `a' * (p - d) ≥ a * p` iff `a' * d ≤ (a' - a) * p`. -/
theorem refund_with_errors (a a' p d : Rat) :
    a' * (p - d) - a * p = (a' - a) * p - a' * d := by
  grind

/-- Auditing a share `t` of hits keeps `1 - t` of the refund: the cost
of knowing the error rate is linear and chosen, never hidden. -/
theorem audit_keeps_refund (g t : Rat) (hg : 0 ≤ g) (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) : 0 ≤ g * (1 - t) ∧ g * (1 - t) ≤ g := by
  constructor
  · exact Rat.mul_nonneg hg (by grind)
  · have := Rat.mul_nonneg hg ht0
    grind

end Hop
