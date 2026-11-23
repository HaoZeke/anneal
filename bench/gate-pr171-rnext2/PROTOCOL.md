# PR 171 r-next2 same-session gate protocol

The immutable base A is develop
`364679f060386949f805a72dea9d7d63b7e9ea4f`, tree
`79610c980d39f19a2d383ffe3c53e4505ddb9b6c`.  The candidate B is the
twelve commits `61fa67cf..d87ca43e` (the four PR commits through
`633d5ce5` and the eight r-next2 commits) applied to A with
`git am -3`, tree `c3a236ac7e28f07bb8ed29fd1436295ba44d1745`.  The
harness checks both trees and records the candidate commit it ran.
Develop moved from `cb0cfbfa` to `364679f0` without touching `src`,
`examples` or the Cargo files, so the base drivers match the previous
gate's base.

## Arms and schedule

`D` is one release build of A.  `D2` is an independent release build
of A.  `G` and `G2` are repeated runs of one release build of B.  Even
seed blocks run `D,G,G2,D2`; odd seed blocks run `D2,G2,G,D`.

Every run is serial and pinned to core 0.  Every environment variable
whose name ends in `_NUM_THREADS` is set to one.  Dynamic thread counts
are disabled.  There is no quick mode, warm-up, retuning, rescue,
hand-picking, or relabeling.  Budgets and stop levels are identical in
every arm.

The preregistered cells are:

* `m38r14d`: `h2h_seed`, Morse 38 at rho 14, budget 400000, stop
  level -144.320054, seeds 2000 through 2011.
* `lj75`: `h2h_gmin_proof` panel cell, budget 4000000, stop level
  -397.491331, seeds 2000 through 2011.
* `lj98`: `h2h_gmin_proof` panel cell, budget 4000000, stop level
  -543.664361, seeds 2000 through 2011.
* `tip4p6`: `h2h_seed` TIP4P water control, budget 400000, stop level
  -197.7795260034, seeds 2000 through 2023.

A hit requires `seen.lowest` and `out.best` at or below the stop
level.

## Measurements

Each row records the hit, calls, charged calls, `seen.lowest`,
`out.best`, the energy of the returned geometry, outside-process
seconds from `getrusage`, and the peak `RssAnon` from
`/proc/<pid>/status`.  It records the observed-basis probe offers,
hits, and charged probe calls from build-only counters.

For `h2h_seed` rows, `stop_run` is one when the run exited cleanly,
`seen.lowest` is at or below the stop level, and calls are below the
budget.  `best_is_lowest` is one when the printed `out.best`,
`seen.lowest`, and returned-geometry energy are equal.  The panel
driver prints one best value, so these fields are empty there.

The target requires a nonzero G probe counter.  The molecular control
requires zero G offers.  Rows are append-only.  A failed run is
retained and makes that cell inconclusive; there are no replacement
runs.

## Preregistered statistics and verdict

All tests compare the first G run with D.  G2 is the repeat arm.
D2/D is the A/A band for every metric.  Random procedures use seed
1712026 and 10000 seed-level bootstrap draws.

For each cell:

1. Fisher's exact test compares the hit tables.  A one-sided
   Mann-Whitney test compares calls with every miss scored as
   `budget + 1`.  G fails only when it is worse and `p < 0.05`.
2. The paired geometric-mean outside-time ratio G/D is bootstrapped by
   seed.  Its 95 percent interval must have upper bound at most 1.10.
   The D2/D interval is reported beside it.
3. The paired geometric-mean peak-RssAnon ratio G/D is bootstrapped by
   seed.  Memory fails only when its interval lower bound is greater
   than `max(1.10, upper end of the D2/D RssAnon interval)`.
4. On `m38r14d`, every G and G2 stop run must have
   `best_is_lowest` equal to one.  D and D2 splits are reported.

Lower G time or memory never fails.  PASS requires complete rows, both
path-counter requirements, and every cell to satisfy every rule.
