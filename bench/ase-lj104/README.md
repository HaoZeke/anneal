Copyright (c) anneal developers

LJ104, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -582.086642.
Anneal is Config::for_cluster. Recommended is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
minima hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 280.907064 s, three times the slowest for_cluster wall, seed 2006 at 93.635688 s. The recommended arm uses that stored cap.
anneal hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
recommended hit on 9 of 12 seeds, spent the call budget on 3 of 12, and reached the wall cap on 0 of 12.
recommended hits are seeds 2000, 2001, 2002, 2003, 2004, 2006, 2007, 2009 and 2011.
recommended misses are seeds 2005, 2008 and 2010, best -582.03842906, -581.01667626 and -581.01667626.
basin hopping hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -582.086642 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -580.80432897 | 1.28231303 | 88.960797 | 1.000000 | 1.000000 | 63.958464 | 25.016078 |
| recommended | 9/12 | 0.000336519 | 18 | 0.000383392 | -582.08663738 | 0.00000462 | 65.135958 | 0.731648 | 1.382426 | 37.029132 | 28.106825 |
| basin hopping | 0/12 | 1 | 72 | 1 | -575.36385980 | 6.72278220 | 127.172397 | 1.427901 | 0.700332 | 83.116758 | 43.981369 |
| minima hopping | 0/12 | 1 | 72 | 1 | -570.25110190 | 11.83554010 | 287.850950 | 3.233731 | 0.309246 | 1.361099 | 286.357127 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 9/12 vs 0/12 | 0.000336519 | 18 | 0.000383392 | 0.514893 | 1.964426 |
| recommended against minima hopping | 9/12 vs 0/12 | 0.000336519 | 18 | 0.000383392 | 0.221828 | 4.553814 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
