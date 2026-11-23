# PR 176 Paired Follow-up Protocol

## Purpose

This follow-up measures the merged change on two cells after the
larger round-two gate. It uses fresh builds and fresh rows. No earlier
rows enter the analysis.

## Arms

The arms are:

- A: base commit `e25221f3a0bf4f78a67d4b7877d38f813b002ef5`,
  tree `cc5c9c3c4062db5e4afaaf4c007cabc8965851cc`.
- A2: an independent release rebuild of A from the same commit and tree.
- B: the merge of A with change head
  `6b12b9aab1e78589f0634413aae8c12611104607`, tree
  `c9f9914d243a9b362513f49ffae749f4ef980e79`.

All three arms are release builds in separate target directories. A and
A2 are built independently. Every run is pinned to one core, and all
supported thread-count controls are set to one.

## Cells and seeds

The cells are `tip4p8` and `lj38`. Each arm runs 24 seeds per cell. Seeds
are `2000..2023`, preserving the earlier scheme and extending it without
reuse of earlier measurements.

Each six-seed block uses these arm orders:

1. A, B, A2
2. B, A2, A
3. A2, A, B
4. B, A, A2
5. A, A2, B
6. A2, B, A

The block repeats four times. This is the preregistered balanced order,
with A and B alternating in ABBA order across consecutive seed pairs.
Both cells and all arms run in one uninterrupted session. If the session
cannot finish, available rows are sealed as a partial result and are not
combined with a later session.

The configured paper budgets and recommended settings are unchanged.
No reduced mode is permitted.

## Recorded fields

Every run records:

- total wall time;
- outside time per charged call;
- charged calls;
- hit status;
- in-process `RssAnon`;
- cell, arm, seed, slot, budget, commit, tree, executable digest, and
  session identifier.

Rows are append-only. Failed or incomplete launches are retained in the
run log and are not replaced silently. Statistical analysis uses all
valid completed rows, including outliers.

## Preregistered bar

B must have no hit loss against A. Hit counts and a two-sided Fisher exact
test are reported. Charged calls are also compared with a two-sided
Mann-Whitney test after scoring each miss as `budget + 1`.

For every seed, the primary timing estimand is the B/A ratio of outside
time per charged call. A seed-paired 95% percentile interval is formed by
resampling the 24 seed ratios with replacement. A second paired 95%
interval is the Student interval on log ratios, transformed back to the
ratio scale. Both interval upper bounds are reported.

The outside-time bar requires the paired 95% upper bound for B/A to be no
more than 1.10. The result is also reported against the empirical rebuild
threshold

`max(1.10, paired 95% upper bound for A2/A)`.

The same seed-paired bootstrap and paired log interval are reported for
total wall-time B/A, but wall time is kept separate from the primary
per-call outside-time verdict.

For `lj38`, charged calls per run for B and A are reported separately in
the summary. They are not merged with either timing ratio into a single
number. Any outlier run is listed with its raw wall time, outside time per
charged call, charged calls, hit status, and `RssAnon`.

## Reporting

Per-run rows and per-arm summaries report all recorded fields. The
per-cell summary reports hits, both B/A timing comparisons, both interval
methods, the A2/A outside-time band, the fixed and rebuild-adjusted bars,
and the separate `lj38` charged-call comparison.
