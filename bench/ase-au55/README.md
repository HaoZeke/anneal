Copyright (c) anneal developers

Au55, seeds 2000-2011, budget 1,000,000, host bench, one core.
The reference minimum is -191.03085280.
Anneal is Config::for_cluster on Gupta::copper_gold(0, 55), with the pair length and well depth as the scales. Recommended is Config::recommended_with_scales at those scales. ASE uses the same Gupta through asg_eval.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The ASE rows are the committed rows and were not rerun.
anneal code_revision 33bed75ff4e49af3628e7156e31c6e53e2fde32c.
recommended code_revision b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
basin hopping code_revision 33bed75ff4e49af3628e7156e31c6e53e2fde32c.
minima hopping code_revision 33bed75ff4e49af3628e7156e31c6e53e2fde32c.
The wall cap is 96.650229 s, three times the slowest for_cluster wall, seed 2001 at 32.216743 s. The recommended arm uses that stored cap.
anneal hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
recommended hit on 0 of 12 seeds, spent the call budget on 12 of 12, and reached the wall cap on 0 of 12.
basin hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.
minima hopping hit on 0 of 12 seeds, spent the call budget on 0 of 12, and reached the wall cap on 12 of 12.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -191.03085280 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -190.81083051 | 0.22002228 | 31.896040 | 1.000000 | 1.000000 | 26.788756 | 5.087176 |
| recommended | 0/12 | 1 | 72 | 1 | -190.98331173 | 0.04754107 | 33.663735 | 1.053887 | 0.948869 | 26.379868 | 7.288144 |
| basin hopping | 0/12 | 1 | 72 | 1 | -189.78747254 | 1.24338026 | 96.650629 | 3.030182 | 0.330013 | 14.617083 | 82.033499 |
| minima hopping | 0/12 | 1 | 72 | 1 | -190.29395447 | 0.73689833 | 98.391769 | 3.092070 | 0.323408 | 2.126042 | 96.272051 |

Fisher and Mann-Whitney below compare recommended with each ASE arm. Misses are censored at the call budget.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 0/12 vs 0/12 | 1 | 72 | 1 | 0.348303 | 2.871061 |
| recommended against minima hopping | 0/12 vs 0/12 | 1 | 72 | 1 | 0.341120 | 2.931524 |

Misses are censored at the call budget for the rank test.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
Inside is time in the potential. Outside is the rest of the wall.
