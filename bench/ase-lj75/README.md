Copyright (c) anneal developers

LJ75, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -397.492331.
This record is based on d80bb1a5bc159b94b04d9cef2402c53c3315de6d,
which contains develop 24ed9551f2a574622fae77ec5f8a4077e1dfd336.
Anneal is Config::for_cluster. Its code_revision is
98dcfb3106f191baf7381fecd5624426831094fe.
ASE basin hopping and minima hopping use the ASE 3.29 constructor
defaults. The harness revision for the new ASE rows is
162d230740f078abbcb5caae9b0aba08477691bd.
The earlier basin-hopping wall was removed because its revision is not reachable. The quoting record remains on `origin/bench/ase-lj75`.
The ASE wall cap is 167.806728 s, three times the slowest anneal
wall, seed 2000 at 55.935576 s. A seed that reaches the cap is a
miss. The row keeps the calls, the wall, the best energy and the cap.
Minima hopping reached that cap on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -397.492331 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 1/12 | 1 | 72 | 1 | -396.28224883 | 1.21008217 | 50.979432 | 1.000000 | 1.000000 | 34.003937 | 17.055697 |
| basin hopping | 0/12 | 1 | 78 | 0.359317 | -393.72607097 | 3.76626003 | 89.907831 | 1.760879 | 0.567898 | 46.580477 | 43.358141 |
| minima hopping | 0/12 | 1 | 78 | 0.359317 | -391.70068193 | 5.79164907 | 172.344747 | 3.375083 | 0.296295 | 0.943898 | 171.463289 |

Anneal seed 2009 hit at 1913436 calls, best -397.49233098.
Misses are censored at 4,000,000 for the rank test.
Medians are type 7. Wall ratios are paired by seed.
The gap is the median best minus -397.492331.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.

Of 930 samples taken while the minima-hopping call rate was falling,
864 were in ComparePositions._indistinguishable_compare.
That fraction is 0.929. 0 samples were in the potential.
