# PR 171 restack same-session gate protocol

The immutable base A is develop
`9b65fda3972fd8c014bce886a3ea5c05956d64af`, tree
`a49a66d1422cfefa28012fd12e6729bc4e879d5f`.  The restacked payload
applied as one squash commit is candidate B, tree
`f172df01104c53c74daf6c74f63bf08c45d19b06`.  The harness checks both
trees and records the candidate commit it ran.

## Arms and schedule

`A` and `A2` are independent release builds of the base.  `B` and
`B2` are repeated runs of one release build of the candidate; `B2` is
not a separate build.  Even seed blocks run `A,B,B2,A2`; odd seed
blocks run `A2,B2,B,A`.

All builds and runs are serial in one session.  Runs are pinned to
core 0.  Every environment variable ending in `_NUM_THREADS` is one,
and dynamic thread counts are disabled.  There is no quick mode,
warm-up, retuning, rescue, hand-picking, or relabeling.  Budgets and
stop levels are identical in every arm.

The preregistered cells are:

* `m38r14d`: Morse 38 at rho 14, budget 400000, stop -144.320054,
  seeds 2000 through 2011.
* `lj38`: Lennard-Jones 38 panel cell, budget 400000, stop
  -173.927427, seeds 2000 through 2011.
* `lj75`: Lennard-Jones 75 panel cell, budget 4000000, stop
  -397.491331, seeds 2000 through 2011.
* `lj98`: Lennard-Jones 98 panel cell, budget 4000000, stop
  -543.664361, seeds 2000 through 2011.
* `tip4p6`: TIP4P water control, budget 400000, stop
  -197.7795260034, seeds 2000 through 2023.

A hit requires the observed and returned best energies to be at or
below the stop level.

## Measurements

Each row records the hit, calls, charged calls, outside wall seconds,
outside wall seconds per call, observed and returned energies, peak
in-process `RssAnon`, and the observed-basis probe offers, hits, and
charged calls.  Outside wall time is elapsed child wall time less the
time measured inside the potential.

The base receives a neutral counter that always writes zeros.  The
candidate counter increments an offer after
`claim_observed_basis_probe` accepts and an observed-basis candidate
exists.  It increments a hit only when that candidate is recorded,
and reports `observed_basis_spent` as probe calls.  This is a
counter-only rewrite at the moved `directed_stall_source` path and
does not change search behaviour.

For seed-driver rows, `stop_run` is one when the run exits cleanly,
reaches the stop level, and uses fewer calls than the budget.
`best_is_lowest` is one when the returned best, observed lowest, and
returned-geometry energy agree.  Panel rows have no geometry split.

The Morse target requires a nonzero B probe counter.  The molecular
control requires zero B offers.  Rows are append-only.  A failed run
is retained and makes that cell inconclusive; there are no
replacement runs.

## Preregistered statistics and verdict

Tests compare B with A.  B2 is a repeated candidate observation.
A2/A is the independent-build band.  Random procedures use seed
20261123 and 10000 seed-level bootstrap draws.

For each cell:

1. Fisher's exact test compares hit tables.  A one-sided Mann-Whitney
   test compares calls with every miss scored as `budget + 1`.  B
   fails only when it is worse and `p < 0.05`.
2. The seed-paired geometric-mean total outside-time ratio B/A is
   bootstrapped.  Its 95 percent interval must have upper bound at
   most 1.10.  The alternative ceiling
   `max(1.10, A2/A upper)` is also reported.  Per-call outside-time
   ratios are reported alongside totals.
3. The paired geometric-mean peak-RssAnon ratio B/A is bootstrapped.
   Memory fails only when its lower interval bound exceeds
   `max(1.10, A2/A upper)`.
4. On `m38r14d`, every B and B2 stop run must have
   `best_is_lowest` equal to one.  A and A2 splits are reported.

Lower B time or memory never fails.  PASS requires complete rows,
both path-counter requirements, and every cell to satisfy every rule.
