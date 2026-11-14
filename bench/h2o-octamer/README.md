# GFN2 (H2O)8

Same protocol as the ammonia tetramer. `recommended_molecular`, 4000
charged calls, dissociated monomers refused before xtb, native CREST,
no wall. Seeds 17000–17005, serial, hostname `bench`. A hit is within
1e-6 Eh of the lowest intact energy either arm found,
−40.681336013150 Eh. Every geometry is intact.

| seed | anneal E / Eh | anneal wall / s | CREST E / Eh | CREST wall / s | anneal − CREST / Eh |
| --- | --- | --- | --- | --- | --- |
| 17000 | −40.676850890277 | 27.403 | −40.681335999128 | 902.675 | 4.485e-3 |
| 17001 | −40.681336013150 | 27.595 | −40.681335990637 | 924.739 | −2.3e-8 |
| 17002 | −40.681245643233 | 27.565 | −40.681335997000 | 909.631 | 9.0e-5 |
| 17003 | −40.670669626989 | 27.415 | −40.681335987654 | 873.555 | 1.067e-2 |
| 17004 | −40.675349096335 | 28.421 | −40.681335997331 | 1080.735 | 5.987e-3 |
| 17005 | −40.670670055537 | 27.368 | −40.681335977771 | 873.894 | 1.067e-2 |

Anneal is 4000 calls on every seed. CREST finds the same energy on every
seed. Anneal reaches it on 17001 and is within 0.1 meH on 17002.

`ladder.csv` is the same starts at larger budgets. Seed 17000 is
4.49e-3 Eh high at 4000 calls and at the CREST energy from 16000 calls
through the 127141 calls CREST used. On the rebased tree seed 17001 is
9.03e-5 Eh high at 16000 calls and at the CREST energy at 64000. Seeds
17002 and 17003 are at the CREST energy at 16000 calls. Seed 17004 stays
9.04e-5 Eh high at 16000, 64000 and the 151738 calls CREST used.
