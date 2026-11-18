Copyright (c) anneal developers

# TIP4P adaptive-population diagnostic

Same-session fixed-call comparison on host `cursor`, seeds 0--5. The baseline
is fetched `origin/develop` at
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`; the fix is
`Config::recommended_adaptive_tip4p` at
`aa55e8aaa01b49c0849a31459effdbb78b2e476a`. Every run spends its complete
budget: 500,000 calls for (H2O)15, 2,000,000 for (H2O)20, and 400,000 for the
LJ38 seed-2000 sanity check. A TIP4P hit is within 0.004184 kJ/mol of the
Wales--Hodges reference; an LJ38 hit is within 0.001 of -173.928427.

| system | baseline hits | adaptive hits | baseline deepest | adaptive deepest |
| --- | ---: | ---: | ---: | ---: |
| (H2O)15 | 1/6 | 2/6 | -628.3722600410 | -628.3722600374 |
| (H2O)20 | 0/6 | 1/6 | -868.6103569245 | -872.9887651402 |
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
only when at least ten non-returning trials supply an acceptance estimate.
Empty crossing windows are held rather than misread as zero acceptance. The
final proposal multipliers were 1.44--1.60; temperature multipliers were
0.99--1.50 on TIP4P. Single-body relocation acceptance rose from 8.4% to 17.4%
at n=15 and from 2.5% to 23.7% at n=20.

The two final rungs did not collapse to identical populations: at n=20 the
cold rung held 1054--1180 basins while every hot rung reached the 2048-seat
index cap, and 369 of 505 swaps were accepted. The low fraction of new visits
(0.21%) nevertheless shows that duplicate-basin traffic remains the dominant
limitation.

`rows.csv` is append-only evidence with the code revision and host on every
row. `step0`/`temp_scale0` and `step1`/`temp_scale1` are the final adaptive
controller multipliers; blank second-replica fields denote the one-chain
baseline.
