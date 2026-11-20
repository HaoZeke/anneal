# TIP4P b11 gate: rows and extensions

The gate rules are in `gates/anneal-tip4p-b11-gate.md`. This file
pre-registers the extensions below before any of their rows is run. The
harness is `scripts/tip4p_h2h/gate_b11.py`.

## Rows

`rows.csv` holds every row of every campaign. Columns that label a row:

- `campaign`: `rigid`, `second`, `nonrigid`, or `probe`.
- `seed_set`: `primary` for seeds 2000 to 2011, `second` for seeds 2012
  to 2023, empty for non-rigid rows.
- `arm` and `label`: `A` develop, `A2` second develop build, `B` b11,
  `C` b6 (label `b6`), `P` a bisect probe.
- `probe`: the commit a probe arm was built from, and on the A rows of
  that probe session.
- `best` is the lowest energy the objective returned, which is the
  energy the driver scores. `search_best` is the search's own reported
  best and is kept for comparison only.

A first launch of the primary campaign was stopped by the harness after
184 rows, because its hit cross-check read `search_best` instead of the
scored energy. Those rows are kept in the logs archive and not used. The
primary campaign was rerun from its first row.

## b6 arm

Arm C is b6 at `53d98cd2`, tree `14b537fd9b73b8fe21e01721954b41a278c5d223`,
run in the same counterbalanced session as A, A2 and B on every primary
cell. Per cell the report gives A, A2, B and b6 hits, and outside time per
charged call for b6/A and B/A. Every TIP4P row has `calls == charged`;
the harness checks this at scoring and fails if any row differs.

## Second seed set

Seeds 2012 to 2023, fixed here. Arms A and B. Cells TIP4P15 to TIP4P21.
Serial, `taskset -c 0`, every thread variable 1, A then B on even seeds
and B then A on odd seeds, no warm-up run. The only stop is the
2,000,000-call budget: no target stop is armed. A hit is the lowest
returned energy at or below the Cambridge energy plus 0.001. Because the
lowest energy only decreases and a target stop does not alter the
trajectory before it fires, a hit under this protocol is the same event
as a hit under the primary protocol's target stop. The campaign runs to
completion once and is not extended.

Analysis plan:

1. Fresh set alone: per-cell hits for A and B, and the B/A ratio of the
   type-7 median outside time per charged call. Rules 1 and 2 of the gate
   apply to these cells as written.
2. Pooled with the primary seeds, n = 24 per cell: per cell, an exact
   two-sided paired McNemar test on (seed, cell) pairs. `b` counts pairs
   where A hits and B misses, `c` pairs where A misses and B hits, and
   `p = min(1, 2 * P(X <= min(b, c)))` for `X ~ Binomial(b + c, 1/2)`,
   with `p = 1` when `b + c = 0`. One more McNemar pools all 168
   TIP4P15 to TIP4P21 seed-cell pairs.

## Bisect

The direction is fixed by the primary b6 result. b6 is bad if its outside
time per charged call is over 1.10 times A on tip4p9 or on tip4p15, or if
it has fewer hits than A on tip4p17 or over TIP4P15 to TIP4P21.

- If b6 is bad, bisect the first-parent commits of `61fa67cf..53d98cd2`.
- Otherwise, bisect the first-parent commits of `53d98cd2..b58dd009`, the
  tip of `fix/tip4p-hits-b9`.

Each probe builds one commit as arm P and runs it against A in its own
session: cells tip4p9, tip4p15 and tip4p17, seeds 2000 to 2011, A then P
on even seeds and P then A on odd seeds, serial, with the primary
protocol's target stop. A probe commit is bad if the type-7 median outside
time per charged call of P is over 1.10 times A on tip4p9 or on tip4p15,
or if P has fewer tip4p17 hits than A. The range's start is taken as good.
Its end is probed first unless it is b6, whose primary rows decide it. If
the end is good, that is reported and no first bad commit is named.
Otherwise probes take the midpoint of the remaining first-parent range
until one commit is left, which is reported with its subject, diffstat
and the per-probe table.
