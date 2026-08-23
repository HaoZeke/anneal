/-! # The temperature ladder

The seam-ladder theorems are conditional, and the LJ75 measurements void
their hypotheses: the crossing has no metastable middle for a bank to
hold (`SeamLadder`, refuted at 0 of 16), and nothing before the crossing
is worth sharing. What the traces do show is the crossing's true shape:
an accepted excursion three to four eps above the floor -- the winners'
staging segments run -391 to -393 against floors of -395 to -396 --
followed by one quench down the far side. Under Metropolis such an
excursion is accepted with probability `exp(-delta/T)`: the crossing is
a thermally activated event, and the axis a coordination can actually
buy is temperature, not position.

That is the mechanism of replica exchange, and the adaptive-exchange
Monte Carlo of Mandelshtam, Frantsuzov and Calvo is what solved LJ74-78
(J. Phys. Chem. A 2006, 110, 5326) -- the same paper this campaign cited
for the entropic term, carrying its method unread.

The theorems here are budget-fair: the ladder never gets more hops than
the ensemble it replaces, it only distributes the same hops over rungs.
`p_k` is the per-hop probability that rung `k` produces a crossing the
exchange can hand down; the physical hypothesis `p_cold <= p_hot` is
Metropolis monotonicity of `exp(-delta/T)` in `T`, which holds below the
melt and fails above it -- the ladder's top is a knob with a ceiling,
not a free win. -/

namespace Hop

/-- Expected crossings from `n` chains all at the cold temperature, each
spending `b` hops: the ensemble that kept losing. -/
def coldPair (b p : Rat) : Rat := b * p + b * p

/-- Expected crossings from the same two chains' budget spent as one
cold rung and one hot rung with exchange: a hot crossing handed down
counts, because the quench below it identifies the funnel either way. -/
def ladderPair (b pc ph : Rat) : Rat := b * pc + b * ph

/-- **The ladder pays exactly when the crossing is thermally activated.**
Same total budget on both sides; the ladder dominates the cold pair
precisely when the hot rung crosses at least as often per hop as the
cold one, which is the Metropolis statement that an uphill excursion is
accepted more readily at higher temperature. If the top rung is past the
melt, `ph < pc` and the same identity says the ladder loses: the
hypothesis is the design constraint, not a formality. -/
theorem ladder_pays (b pc ph : Rat) (hb : 0 ≤ b) (h : pc ≤ ph) :
    coldPair b pc ≤ ladderPair b pc ph := by
  have := Rat.mul_le_mul_of_nonneg_left h hb
  grind [coldPair, ladderPair]

/-- The gain is the whole thermal excess, linear in budget: what the
ladder buys grows with every hop the hot rung spends, where the seam
bank's stale restarts bought nothing at any budget. -/
theorem ladder_gain (b pc ph : Rat) :
    ladderPair b pc ph - coldPair b pc = b * (ph - pc) := by
  grind [coldPair, ladderPair]

end Hop
