Copyright (c) anneal developers

LJ55, seeds 2000-2011, budget 1,000,000, host bench, one core.
The reference minimum is -279.248470.
Anneal is Config::for_cluster. ASE basin hopping and minima hopping
use the ASE 3.29 constructor defaults.
The rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625
or 3654866310baf4ef63853130608ab65392429ff6. Both commits hold the
same search. host is bench.
The wall cap is 4.424556 s, three times the slowest anneal wall,
1.474852 s.
Basin hopping reached the cap on 11 of 12 seeds. Seed 2002 hit at
75325 calls, best -279.24756501, wall 1.323068 s.
Minima hopping reached the cap on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -279.248470 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 1 | 72 | 1 | -279.24847046 | -0.00000046 | 0.823500 | 1.000000 | 1.000000 | 0.562777 | 0.260722 |
| basin hopping | 1/12 | 9.61483e-06 | 135 | 0.000147956 | -271.21690949 | 8.03156051 | 4.424591 | 5.391291 | 0.186119 | 1.738571 | 2.681977 |
| minima hopping | 0/12 | 7.39602e-07 | 144 | 1.02689e-05 | -266.05409411 | 13.19437588 | 4.488125 | 5.817187 | 0.174737 | 0.183773 | 4.307671 |

Misses are censored at 1,000,000 for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
