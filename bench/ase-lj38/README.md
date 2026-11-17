Copyright (c) anneal developers

LJ38, seeds 2000-2011, budget 400,000, host bench, one core.
The reference minimum is -173.928427.
Anneal is Config::for_cluster. Recommended is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
minima hopping code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The wall cap is 4.653972 s, three times the slowest for_cluster wall, seed 2006 at 1.551324 s. The recommended arm uses that stored cap.
anneal hit on 10 of 12 seeds, spent the call budget on 2 of 12, and reached the wall cap on 0 of 12.
anneal hits are seeds 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008, 2009 and 2011.
anneal misses are seeds 2006 and 2010, best -173.25237842 and -173.13431701.
recommended hit on 12 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 0 of 12.
basin hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -173.928427 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10/12 | 1 | 72 | 1 | -173.92842659 | 0.00000041 | 0.981375 | 1.000000 | 1.000000 | 0.574202 | 0.407172 |
| recommended | 12/12 | 0.478261 | 6 | 0.000155282 | -173.92842659 | 0.00000041 | 0.066933 | 0.068564 | 15.393024 | 0.026147 | 0.042388 |
| basin hopping | 0/12 | 6.73038e-05 | 132 | 0.000125309 | -171.10081694 | 2.82761006 | 4.654009 | 4.998332 | 0.210866 | 1.319512 | 3.334485 |
| minima hopping | 0/12 | 6.73038e-05 | 132 | 0.000125309 | -170.08494879 | 3.84347821 | 4.864397 | 5.168082 | 0.200434 | 0.138811 | 4.728948 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0 | 1.02689e-05 | 0.014382 | 69.785574 |
| recommended against minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0 | 1.02689e-05 | 0.013796 | 72.557467 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
