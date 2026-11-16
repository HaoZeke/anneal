# GFN2 (NH3)8

Same protocol as the tetramer. The search is `recommended_molecular` with a
budget of 4000 charged calls. A dissociated monomer is refused before xtb. CREST is
native xtb, no wall. Seeds 17000–17005, serial, hostname `bench`. A hit is
within 1e-6 Eh of the lowest intact energy either arm found,
−35.470769324714 Eh. Every geometry is intact.

| seed | anneal E / Eh | anneal wall / s | CREST E / Eh | CREST wall / s | anneal − CREST / Eh |
| --- | --- | --- | --- | --- | --- |
| 17000 | −35.469671251735 | 36.394 | −35.470769125262 | 2176.419 | 1.098e-3 |
| 17001 | −35.468680645441 | 39.398 | −35.470769313264 | 2051.465 | 2.089e-3 |
| 17002 | −35.468966682688 | 26.941 | −35.470769289051 | 2952.690 | 1.803e-3 |
| 17003 | −35.470769324714 | 39.388 | −35.470769221036 | 2275.531 | −1.037e-7 |
| 17004 | −35.470395813907 | 39.778 | −35.470769267569 | 1209.778 | 3.735e-4 |
| 17005 | −35.469218070447 | 39.270 | −35.470769273098 | 1774.669 | 1.551e-3 |

The `3141987e` rows are the anneal arm again after the rebase onto that
develop. CREST is unchanged. Every seed charges 4000. On the previous
tree, seeds 17000 and 17002 stopped at 4080 and 2855 calls whatever the
budget (`ladder.csv`), because a lost-bond refusal ended the quench
without a charge and the hop returned early.
On this tree the same gate leaves the ledger open. At the CREST call
count both seeds charge every call and finish 3.6e-5 Eh above CREST
(−35.470732540153 at 243868 calls, −35.470732881715 at 332675).

That remainder is a second minimum. For seed 17000 the anneal geometry
matches CREST conformer 3 (heavy-atom RMSD 0.015 Å, all-atom 0.016 Å)
and not the global minimum (heavy-atom RMSD 0.345 Å, all-atom 1.05 Å,
one hydrogen 4.54 Å away). Both have 12 hydrogen bonds. The global
minimum pairs four contacts near 1.89 Å with eight near 2.14 Å. The
anneal minimum keeps four contacts near 1.88 Å and spreads the rest
from 2.10 Å out to 2.26 Å. Four of the eight nitrogens sit 0.29–0.52 Å
from the global-minimum cage after the heavy atoms are aligned. Seed
17002 is the same kind of miss: heavy-atom RMSD 0.33 Å to the global
minimum, 3.64e-5 Eh above it. Starting the search on the global minimum
records that energy, so the gap is not a loose quench of the same well.

The `cb248286` rows are the same starts at 16000 calls with the TRIC
quench develop selects for this preset. Every seed still finishes above
CREST. Seeds 17000 and 17002 remain 3.5e-5 and 3.6e-5 Eh high.

`ladder.csv` repeats the same starts at 16000, 64000 and the CREST call
count. Seed 17001 is 2.09e-3 Eh high at 4000, 4.27e-4 at 64000, and
3.54e-5 at 221579 calls. A longer budget is a different trajectory, so
17003, already at CREST at 4000, is 1.15e-3 high at 241500 calls.
