Copyright (c) anneal developers

LJ104, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -582.086642.
Anneal is Config::for_cluster. ASE basin hopping and minima hopping
use the ASE 3.29 constructor defaults.
code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625. host is bench.
The wall cap is 280.907064 s, three times the slowest anneal wall,
93.635688 s.
Every arm missed. Basin hopping spent the call budget on 12 of 12 seeds.
Minima hopping reached the cap on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -582.086642 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -580.80432897 | 1.28231303 | 88.960797 | 1.000000 | 1.000000 | 63.958464 | 25.016078 |
| basin hopping | 0/12 | 1 | 72 | 1 | -575.36385980 | 6.72278220 | 127.172397 | 1.427901 | 0.700332 | 83.116758 | 43.981369 |
| minima hopping | 0/12 | 1 | 72 | 1 | -570.25110190 | 11.83554010 | 287.850950 | 3.233731 | 0.309246 | 1.361099 | 286.357127 |

Misses are censored at 4,000,000, so the three call lists tie and the
rank test is U = 72, p = 1.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
