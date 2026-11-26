# LJ75 arch-chains pooled comparison

Pooled rows: 36 (12 seeds, arms comm, indep, and single). Budget 4000000 oracle calls. A miss is counted as 4000001 calls. Hits are the recorded hit field. The campaign global minimum is -397.4923309829.

## Per arm

| arm | hits | median calls_to_hit |
| --- | --- | --- |
| comm | 7/12 | 3611831.5000 |
| indep | 6/12 | 3637966.5000 |
| single | 9/12 | 2065577.5000 |

## Pairwise tests

The median-calls ratio is median(left) / median(right). Fisher's exact test is two-sided on the hit/miss table, and the odds ratio is the left arm's hit odds divided by the right arm's hit odds. The Mann-Whitney statistic is the two-sided U for the left arm from scipy.stats.mannwhitneyu, with the continuity correction and method auto. The 95% interval is the 2.5 and 97.5 percentiles of 10000 resamples of the median-calls ratio. Each contrast uses numpy Generator seed 20261123 and resamples the two arms independently with replacement.

### comm vs single

Hits comm 7/12, single 9/12. Fisher odds ratio 0.466667, two-sided p 6.668461e-01. Mann-Whitney U 86.5, two-sided p 4.102422e-01. Median-calls ratio 1.74858, 95% interval [0.495464, 2.93235]. The 95% interval includes 1.

### indep vs single

Hits indep 6/12, single 9/12. Fisher odds ratio 0.333333, two-sided p 4.003231e-01. Mann-Whitney U 89, two-sided p 3.278291e-01. Median-calls ratio 1.76123, 95% interval [0.525477, 2.93235]. The 95% interval includes 1.

### comm vs indep

Hits comm 7/12, indep 6/12. Fisher odds ratio 1.4, two-sided p 1.000000e+00. Mann-Whitney U 69, two-sided p 8.793613e-01. Median-calls ratio 0.992816, 95% interval [0.382729, 2.34693]. The 95% interval includes 1.

## Shard hit check

Reported hits for shards 2000, 2003, 2006, and 2009: single 3, 1, 3, 2; comm 2, 2, 1, 2; indep 1, 1, 2, 2.

Observed hits in that shard order: single 3, 1, 3, 2; comm 2, 2, 1, 2; indep 1, 1, 2, 2.

Observed shard hits match the reported check.
