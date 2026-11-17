Copyright (c) anneal developers

LJ98, seeds 2000-2011, budget 4,000,000, host bench, one core.
The reference minimum is -543.665361.
Anneal is Config::for_cluster. Recommended is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
minima hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 246.546693 s, three times the slowest for_cluster wall, seed 2010 at 82.182231 s. The recommended arm uses that stored cap.
anneal hit on 4 of 12 seeds, spent the call budget on 8 of 12, and reached the wall cap on 0 of 12.
anneal hits are seeds 2000, 2004, 2008 and 2011.
anneal misses are seeds 2001, 2002, 2003, 2005, 2006, 2007, 2009 and 2010, best -542.89147285, -543.54672557, -541.39486717, -543.54672557, -541.47319977, -543.64295724, -541.47319977 and -541.86974818.
recommended hit on 10 of 12 seeds, spent the call budget on 2 of 12, and reached the wall cap on 0 of 12.
recommended hits are seeds 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 and 2011.
recommended misses are seeds 2000 and 2001, best -543.64295724 and -543.64295724.
basin hopping hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -543.665361 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 4/12 | 1 | 72 | 1 | -543.54672557 | 0.11863543 | 80.934059 | 1.000000 | 1.000000 | 57.146890 | 23.659054 |
| recommended | 10/12 | 0.0360748 | 26 | 0.00639975 | -543.66536077 | 0.00000023 | 37.016521 | 0.905323 | 1.104982 | 21.307591 | 15.708931 |
| basin hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -536.02030046 | 7.64506053 | 118.307105 | 1.460599 | 0.684651 | 74.478849 | 43.673459 |
| minima hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -532.04115348 | 11.62420752 | 254.781554 | 3.158878 | 0.316569 | 1.270590 | 253.521140 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 10/12 vs 0/12 | 6.73038e-05 | 12 | 0.000125309 | 0.312831 | 3.207960 |
| recommended against minima hopping | 10/12 vs 0/12 | 6.73038e-05 | 12 | 0.000125309 | 0.145297 | 6.907829 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
