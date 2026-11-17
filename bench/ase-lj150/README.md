Copyright (c) anneal developers

LJ150, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -893.310258.
Anneal is Config::for_cluster. Recommended is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
minima hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 544.418154 s, three times the slowest for_cluster wall, seed 2007 at 181.472718 s. The recommended arm uses that stored cap.
anneal hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
recommended hit on 5 of 12 seeds, spent the call budget on 7 of 12, and reached the wall cap on 0 of 12.
recommended hits are seeds 2000, 2002, 2006, 2007 and 2009.
recommended misses are seeds 2001, 2003, 2004, 2005, 2008, 2010 and 2011, best -884.64066360, -885.28228245, -883.17507380, -884.63620281, -883.49689268, -883.49689268 and -883.11058782.
basin hopping hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -893.310258 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -884.65296483 | 8.65729317 | 177.839157 | 1.000000 | 1.000000 | 132.630578 | 45.213123 |
| recommended | 5/12 | 0.0372671 | 42 | 0.0164732 | -884.96147303 | 8.34878497 | 215.989911 | 1.208740 | 0.827400 | 131.988054 | 86.158183 |
| basin hopping | 0/12 | 1 | 72 | 1 | -868.96761602 | 24.34264198 | 213.520749 | 1.199954 | 0.833366 | 167.078759 | 46.468266 |
| minima hopping | 0/12 | 1 | 72 | 1 | -862.56043539 | 30.74982261 | 571.325791 | 3.216914 | 0.310859 | 2.452872 | 567.757919 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 5/12 vs 0/12 | 0.0372671 | 42 | 0.0164732 | 1.010353 | 0.990122 |
| recommended against minima hopping | 5/12 vs 0/12 | 0.0372671 | 42 | 0.0164732 | 0.386276 | 2.593210 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
