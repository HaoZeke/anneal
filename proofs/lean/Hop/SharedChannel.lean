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

end Hop
