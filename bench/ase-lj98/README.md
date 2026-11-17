Copyright (c) anneal developers

LJ98, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -543.665361.
Anneal is Config::for_cluster. ASE basin hopping and minima hopping
use the ASE 3.29 constructor defaults.
code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 246.546693 s, three times the slowest anneal wall,
82.182231 s.
Basin hopping spent the call budget on 12 of 12 seeds.
Minima hopping reached the cap on 12 of 12 seeds.
Anneal hit on seeds 2000, 2004, 2008 and 2011.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -543.665361 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 4/12 | 1 | 72 | 1 | -543.54672557 | 0.11863543 | 80.934059 | 1.000000 | 1.000000 | 57.146890 | 23.659054 |
| basin hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -536.02030046 | 7.64506053 | 118.307105 | 1.460599 | 0.684651 | 74.478849 | 43.673459 |
| minima hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -532.04115348 | 11.62420752 | 254.781554 | 3.158878 | 0.316569 | 1.270590 | 253.521140 |

Misses are censored at 4,000,000 for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
