# GFN2 (H2O)8

Same protocol as the ammonia tetramer. `recommended_molecular`, 4000
charged calls, dissociated monomers refused before xtb, native CREST,
no wall. Seeds 17000–17005, serial, hostname `bench`. A hit is within
1e-6 Eh of the lowest intact energy either arm found,
−40.681336013150 Eh. Every geometry is intact.

Anneal, 4000 calls on every seed:

| seed | E / Eh | wall / s |
| --- | --- | --- |
| 17000 | −40.676850890277 | 27.403 |
| 17001 | −40.681336013150 | 27.595 |
| 17002 | −40.681245643233 | 27.565 |
| 17003 | −40.670669626989 | 27.415 |
| 17004 | −40.675349096335 | 28.421 |
| 17005 | −40.670670055537 | 27.368 |

CREST, each seed at the length of that run:

| seed | E / Eh | wall / s |
| --- | --- | --- |
| 17000 | −40.681335999128 | 902.675 |
| 17001 | −40.681335990637 | 924.739 |
| 17002 | −40.681335997000 | 909.631 |
| 17003 | −40.681335987654 | 873.555 |
| 17004 | −40.681335997331 | 1080.735 |
| 17005 | −40.681335977771 | 873.894 |

Anneal is 4000 calls on every seed. CREST's call counts on these seeds are its own run lengths, 127141 and 151738 among them. The stops differ, so there is no paired test. CREST's energy is the same on every
seed. Anneal's energy matches that value on 17001 and is within 0.1 meH on 17002.

`ladder.csv` is the same starts at larger budgets. Seed 17000 is
4.49e-3 Eh high at 4000 calls and at the CREST energy from 16000 calls
through the 127141 calls CREST used. On the rebased tree seed 17001 is
9.03e-5 Eh high at 16000 calls and at the CREST energy at 64000. Seeds
17002 and 17003 are at the CREST energy at 16000 calls. Seed 17004 stays
9.04e-5 Eh high at 16000, 64000 and the 151738 calls CREST used on the
Cartesian quench. The `cb248286` rows are 16000 calls with the TRIC
quench. Seeds 17000, 17001, 17002, 17004 and 17005 reach the CREST
energy. Seed 17003 holds the minimum 9.03e-5 Eh above it.

These arms are not the (H2O)8 row of the manuscript.
The anneal file of that row records no target, tolerance, budget, or stop.
This 4000-call comparison is not that row.
