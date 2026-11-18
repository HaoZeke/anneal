Copyright (c) anneal developers

# TIP4P adaptive-population diagnostic

Same-session fixed-call comparison on host `cursor`, canonical seeds
2000--2005. The baseline is fetched `origin/develop` at
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`; the fix is
`Config::recommended_adaptive_tip4p` at
`107ea4294d9766051ec9e63a95a49175688885b7`. Every water run spends 2,000,000
calls; the LJ38 seed-2000 sanity check spends 400,000. A TIP4P hit is within
0.001 kJ/mol of the Cambridge reference; an LJ38 hit is within 0.001 of
-173.928427.

| system | baseline hits | adaptive hits | baseline deepest | adaptive deepest |
| --- | ---: | ---: | ---: | ---: |
| (H2O)15 | 4/6 | 3/6 | -628.3722600476 | -628.3722600519 |
| (H2O)20 | 3/6 | 0/6 | -872.9887651731 | -868.3348050446 |
| LJ38 seed 2000 | 1/1 | 1/1 | -173.9284265906 | -173.9284265906 |

This is a small diagnostic panel, not a rate claim. The adaptive arm is
negative on these seeds: one fewer n=15 hit and three fewer n=20 hits. It
remains opt-in, and the unchanged recommended preset retains the baseline
behavior. LJ38 shows no regression.

## Failure profile

The baseline is one live chain, not a population. Across the six runs, only
0.16% of classified (H2O)15 visits and 0.15% of (H2O)20 visits were new.
Collective rigid kicks, translations, and rotations were accepted at 99.9--100%;
single-body relocations were accepted at 7.6% for n=15 and 1.5% for n=20.
The existing large-rigid step controller scales every amplitude-bearing rigid
move from aggregate acceptance; full reorientations, relocations, and cycle
reversals have no tunable step. The existing rigid Metropolis override also
read the configured temperature directly, bypassing the computed live
temperature.

The adaptive arm runs two replicas under one unchanged total call budget.
Every 50 proposals, each replica adjusts its general proposal multiplier from
return and non-return acceptance rates and adjusts its temperature multiplier
only when at least ten non-returning trials supply an acceptance estimate.
Empty crossing windows are held rather than misread as zero acceptance.
Replica exchange uses those adapted temperatures in its swap ratio. The final
proposal multipliers were 1.60; temperature multipliers were 0.86--1.50 on
TIP4P. Single-body relocation acceptance rose from 7.6% to 27.9% at n=15 and
from 1.5% to 23.0% at n=20. The higher crossing acceptance did not translate
to better global-minimum hit rates.

The two final rungs did not collapse to identical populations: at n=20 the
cold rung held 672--1153 basins while the hot rung held 1117--2048, and 332 of
424 swaps were accepted. The low fraction of new visits (0.24%) nevertheless
shows that duplicate-basin traffic remains the dominant limitation.

`rows.csv` is append-only evidence with the code revision and host on every
row. `step0`/`temp_scale0` and `step1`/`temp_scale1` are the final adaptive
controller multipliers; blank second-replica fields denote the one-chain
baseline.
`rungs.csv` records both populations and supports the collapse diagnostic.
Rows outside seeds 2000--2005, and rows from `7554f672...` and `aa55e8aa...`,
preserve earlier exploratory panels; the comparison above uses the canonical
seeds at the final `107ea429...` implementation.
