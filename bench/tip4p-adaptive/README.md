Copyright (c) anneal developers

# TIP4P adaptive-population diagnostic

Same-session fixed-call comparison on host `cursor`, seeds 0--5. The baseline
is fetched `origin/develop` at
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`; the fix is
`Config::recommended_adaptive_tip4p` at
`7554f6724c2854259bb4fc943d6998f853d6c007`. Every run spends its complete
budget: 500,000 calls for (H2O)15, 2,000,000 for (H2O)20, and 400,000 for the
LJ38 seed-2000 sanity check. A TIP4P hit is within 0.004184 kJ/mol of the
Wales--Hodges reference; an LJ38 hit is within 0.001 of -173.928427.

| system | baseline hits | adaptive hits | baseline deepest | adaptive deepest |
| --- | ---: | ---: | ---: | ---: |
| (H2O)15 | 1/6 | 2/6 | -628.3722600410 | -628.3722600409 |
| (H2O)20 | 0/6 | 1/6 | -868.6103569245 | -872.9887651423 |
| LJ38 seed 2000 | 1/1 | 1/1 | -173.9284265906 | -173.9284265906 |

This is a small diagnostic panel, not a rate claim. It is a positive but
statistically neutral result.

## Failure profile

The baseline is one live chain, not a population. Across the six runs, only
0.93% of classified (H2O)15 visits and 0.19% of (H2O)20 visits were new.
Collective rigid kicks, translations, and rotations were accepted at 99.9--100%;
single-body relocations were accepted at 8.4% for n=15 and 2.5% for n=20.
The existing large-rigid step controller scales every amplitude-bearing rigid
move from aggregate acceptance; full reorientations, relocations, and cycle
reversals have no tunable step. The existing rigid Metropolis override also
read the configured temperature directly, bypassing the computed live
temperature.

The adaptive arm runs two replicas under one unchanged total call budget.
Every 50 proposals, each replica adjusts its general proposal multiplier from
return and non-return acceptance rates and adjusts its temperature multiplier
from non-return acceptance. Both sizes settled at the allowed 1.6 step
multiplier and 0.65 temperature multiplier on both replicas. At n=20,
single-body relocation acceptance rose from 2.5% to 8.6%. The two final rungs
did not collapse to identical populations: their basin-count ranges were
426--583 and 799--1503, and 441 of 494 swaps were accepted. The low fraction of
new visits (0.22%) nevertheless shows that duplicate-basin traffic remains the
dominant limitation.

`rows.csv` is append-only evidence with the code revision and host on every
row. `step0`/`temp_scale0` and `step1`/`temp_scale1` are the final adaptive
controller multipliers; blank second-replica fields denote the one-chain
baseline.
