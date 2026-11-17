Copyright (c) anneal developers

LJ150, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -893.310258.
Anneal is Config::for_cluster. ASE basin hopping and minima hopping
use the ASE 3.29 constructor defaults.
code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625. host is bench.
The wall cap is 544.418154 s, three times the slowest anneal wall,
181.472718 s.
Every arm missed. Basin hopping spent the call budget on 12 of 12 seeds.
Minima hopping reached the cap on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -893.310258 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -884.65296483 | 8.65729317 | 177.839157 | 1.000000 | 1.000000 | 132.630578 | 45.213123 |
| basin hopping | 0/12 | 1 | 72 | 1 | -868.96761602 | 24.34264198 | 213.520749 | 1.199954 | 0.833366 | 167.078759 | 46.468266 |
| minima hopping | 0/12 | 1 | 72 | 1 | -862.56043539 | 30.74982261 | 571.325791 | 3.216914 | 0.310859 | 2.452872 | 567.757919 |

Misses are censored at 4,000,000, so the three call lists tie and the
rank test is U = 72, p = 1.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
