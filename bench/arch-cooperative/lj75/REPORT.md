# CooperativeRun LJ75 report

The arm completed 12 seeds and hit the target on
12 seeds. Misses enter calls-to-hit as
4000001. Intervals are percentile bootstrap 95% confidence
intervals from 10000 resamples with seed 20261123.

The earlier row is retained unchanged in `rows-bankrpc-only.csv`.
Its configuration had `shape_keyed` disabled, so it is excluded from
this report and is not a paper cell.

| Measure | Estimate | 95% CI |
| --- | ---: | ---: |
| Hit rate | 1.000 | [1.000, 1.000] |
| Calls to hit, median | 26901 | [10045, 310879] |
| Wall time, median seconds | 556.022 | [519.908, 625.366] |
| Ledger sum, median calls | 4000000 | [4000000, 4000000] |

Named-term, worker-total, coordinator-total, and budget mismatches:
none.


Fisher's exact test and the Mann-Whitney U test are unavailable because
the requested data contain one arm and no comparison arm.
