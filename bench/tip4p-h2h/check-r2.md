# R2 check

rows compared: 120
mismatches: 0

Wall times are within-run only and cannot be compared outside this run.

(H2O)25 has no independent reference. Hits are not scored. Only best energies are reported: the median and the minimum on each arm.

All runs used one machine id and core 0.

## Mismatches

- none

## Wall time

wall_mwu_p is the exact two-sided Mann-Whitney test on raw wall time.

| molecules | n anneal | n gmin | wall-time MW method | wall median anneal | wall iqr anneal | wall max anneal | wall median gmin | wall iqr gmin | wall max gmin | wall-time MW p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 12 | 12 | exact-rank-enumeration | 0.022406 | 0.025737 | 0.061736 | 1.952360 | 13.694607 | 16.031700 | 7.39602e-07 |
| 15 | 12 | 12 | exact-rank-enumeration | 11.667445 | 11.180535 | 17.782377 | 48.015600 | 2.286625 | 51.756000 | 7.39602e-07 |
| 20 | 12 | 12 | exact-rank-enumeration | 21.541228 | 17.971949 | 29.214083 | 83.630100 | 4.869725 | 88.417400 | 7.39602e-07 |
| 21 | 12 | 12 | exact-rank-enumeration | 26.312360 | 1.031697 | 27.535068 | 81.096750 | 7.100475 | 88.136800 | 7.39602e-07 |
| 25 | 12 | 12 | exact-rank-enumeration | 35.730266 | 1.755382 | 37.045699 | 107.109000 | 3.485750 | 111.252000 | 7.39602e-07 |

At n=21 and n=25 every run on both arms used the full 2,000,000-call budget, so those wall differences measure per-call cost, not search success.

At n=8, gmin wall iqr is larger than its median; this spread is high.
At n=8, wall max is anneal 0.061736 and gmin 16.031700.

## Search

Hits and calls-to-hit are scored only where an independent reference exists. A miss is ranked worse than a hit at the same call count. The exact Mann-Whitney on calls-to-hit and Fisher on hits each use a Bonferroni correction over the 4 scored sizes. The search claim rests on those columns.

| molecules | n anneal | n gmin | hits/n anneal | hits/n gmin | calls median anneal | calls median gmin | calls-to-hit MW method | calls-to-hit MW p | calls-to-hit MW p Bonferroni | Fisher p | Fisher p Bonferroni |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 12 | 12 | 12/12 | 8/12 | 6339.0 | 240163.0 | exact-rank-enumeration | 7.39602e-07 | 2.95841e-06 | 0.0931677 | 0.372671 |
| 15 | 12 | 12 | 7/12 | 0/12 | 1481619.5 | 2000000.0 | exact-rank-enumeration | 0.00457666 | 0.0183066 | 0.00457666 | 0.0183066 |
| 20 | 12 | 12 | 6/12 | 0/12 | 1686249.5 | 2000000.0 | exact-rank-enumeration | 0.01373 | 0.0549199 | 0.01373 | 0.0549199 |
| 21 | 12 | 12 | 0/12 | 0/12 | 2000000.0 | 2000000.0 | exact-rank-enumeration | 1 | 1 | 1 | 1 |

## Best energies where hits are not scored

These sizes have no independent reference. There is no hit column, no Fisher column, and no calls-to-hit column. The table file leaves those fields empty on these rows.

| molecules | n anneal | n gmin | anneal min | anneal median | gmin min | gmin median |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | 12 | 12 | -1113.0346568571 | -1103.3447313609 | -1087.3344641800 | -1082.1265516750 |

