Copyright (c) anneal developers

LJ38, seeds 2000-2011, budget 400,000, host bench, one core.
The reference minimum is -173.928427.
Anneal is Config::for_cluster. ASE basin hopping and minima hopping
use the ASE 3.29 constructor defaults.
code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 4.653972 s, three times the slowest anneal wall,
seed 2006 at 1.551324 s.
Basin hopping and minima hopping reached that cap on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -173.928427 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10/12 | 1 | 72 | 1 | -173.92842659 | 0.00000041 | 0.981375 | 1.000000 | 1.000000 | 0.574202 | 0.407172 |
| basin hopping | 0/12 | 6.73038e-05 | 132 | 0.000125309 | -171.10081694 | 2.82761006 | 4.654009 | 4.998332 | 0.210866 | 1.319512 | 3.334485 |
| minima hopping | 0/12 | 6.73038e-05 | 132 | 0.000125309 | -170.08494879 | 3.84347821 | 4.864397 | 5.168082 | 0.200434 | 0.138811 | 4.728948 |

Anneal misses are seeds 2006 and 2010, best -173.25237842 and -173.13431701.
Misses are censored at 400,000 for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
