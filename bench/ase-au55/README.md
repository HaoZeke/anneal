Copyright (c) anneal developers

Au55, seeds 2000-2011, budget 1,000,000, host bench, one core.
The reference minimum is -191.03085280.
Anneal is Config::for_cluster on Gupta::copper_gold(0, 55), with the
pair length and well depth as the scales. ASE basin hopping and minima
hopping use the ASE 3.29 constructor defaults and the same Gupta
through asg_eval.
code_revision 33bed75ff4e49af3628e7156e31c6e53e2fde32c. host is bench.
The wall cap is 96.650229 s, three times the slowest anneal wall,
32.216743 s.
Every arm missed. Basin hopping and minima hopping reached the cap
on 12 of 12 seeds.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to -191.03085280 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -190.81083051 | 0.22002228 | 31.896040 | 1.000000 | 1.000000 | 26.788756 | 5.087176 |
| basin hopping | 0/12 | 1 | 72 | 1 | -189.78747254 | 1.24338026 | 96.650629 | 3.030182 | 0.330013 | 14.617083 | 82.033499 |
| minima hopping | 0/12 | 1 | 72 | 1 | -190.29395447 | 0.73689833 | 98.391769 | 3.092070 | 0.323408 | 2.126042 | 96.272051 |

Misses are censored at 1,000,000, so the three call lists tie and the
rank test is U = 72, p = 1.
Medians are type 7. Wall ratios are paired by seed.
Fisher is the two-sided exact test. Mann-Whitney is two-sided with
the tie correction and the continuity correction.
