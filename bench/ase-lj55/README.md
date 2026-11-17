Copyright (c) anneal developers

LJ55, seeds 2000-2011, budget 1,000,000, host bench, one core.
The reference minimum is -279.248470.
Anneal is Config::for_cluster. Recommended is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 3654866310baf4ef63853130608ab65392429ff6, 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
minima hopping code_revision 3654866310baf4ef63853130608ab65392429ff6.
The wall cap is 4.424556 s, three times the slowest for_cluster wall, seed 2005 at 1.474852 s. The recommended arm uses that stored cap.
anneal hit on 12 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 0 of 12.
recommended hit on 12 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 0 of 12.
basin hopping hit on 1 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 11 of 12.
basin hopping hits are seeds 2002.
basin hopping misses are seeds 2000, 2001, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 and 2011, best -266.69343009, -273.99219814, -271.57561032, -264.88452782, -265.76181467, -268.76444026, -270.85820867, -274.19184105, -273.94082392, -272.24591034 and -268.81672059.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -279.248470 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 1 | 72 | 1 | -279.24847046 | -0.00000046 | 0.823500 | 1.000000 | 1.000000 | 0.562777 | 0.260722 |
| recommended | 12/12 | 1 | 9 | 0.000308045 | -279.24847046 | -0.00000046 | 0.127701 | 0.201256 | 5.222613 | 0.055039 | 0.074915 |
| basin hopping | 1/12 | 9.61483e-06 | 135 | 0.000147956 | -271.21690949 | 8.03156051 | 4.424591 | 5.391291 | 0.186119 | 1.738571 | 2.681977 |
| minima hopping | 0/12 | 7.39602e-07 | 144 | 1.02689e-05 | -266.05409411 | 13.19437588 | 4.488125 | 5.817187 | 0.174737 | 0.183773 | 4.307671 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 12/12 vs 1/12 | 9.61483e-06 | 1 | 1.86753e-05 | 0.028862 | 40.684581 |
| recommended against minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0 | 1.02689e-05 | 0.026010 | 42.766681 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
