# Ledger gate for the merged payload

The measurement is fixed here, and in the runner, the guard and the
analysis, before any row is written. Rows are append-only. A failed row
is kept. There is no replacement run, no quick mode, no warm-up and no
hand edit of a result.

## Subjects

A is develop `c425635bf564a989b180adc3311ded09066f73d6`. If that branch
later moves, the arm is still this commit, and the live head is recorded
beside it. A2 is a second release build of that same tree. In the rows
its arm name is `Ap`.

B is `452fb3312edcb211530da59647723e6db9b5796d`. The gate records its
tree and the tree from `git merge-tree --write-tree` of A and B.

P is `3f4cfa63245e3d5e27158af565a26d48e4ff9b74`, fetched from origin.
It is a trajectory reference. If that object cannot be fetched, P is
skipped and the report says so.

## Arms and schedule

One serial session, pinned to core 0. Every variable whose name ends in
`_NUM_THREADS` is one. Release builds only, one toolchain for every arm.

Seeds are 2000 through 2011, twelve per arm per cell. Timed arms use
palindromic blocks of two seeds, in the order A, B, A2, A2, B, A. Each
timed arm has the same mean position. P runs after that session, cell
by cell and seed by seed, on the same core. It is not part of the
timed order.

Budgets and stop levels are the same on every arm. Molecular budgets
come from `load_budgets()` in the runner (the recorded per-seed anneal
budget, or the median of those seeds when a seed has none). The other
cells use these paper budgets:

| cell | budget | stop level |
|---|---|---|
| lj38 | 400000 | -173.928427 |
| m38r14d | 400000 | -144.321054 |
| lj75 | 4000000 | -397.492331 |
| lj98 | 4000000 | -543.665361 |
| m80r6 | 4000000 | -378.333471 |
| au55 | 1000000 | -191.03085280 |
| hexamer | 500000 | -197.7805260034 |

The stop level for those seven cells is the published minimum plus
0.001. Molecular cells use the recommended molecular preset and the
paired conformer hit rule.

## Timing and memory

A timed sample is the summed wall of one or more identical searches.
The repetition count starts at one. When that wall is below 50 ms the
same search is repeated until the sum reaches 50 ms, and at most 64
times. `wall_s` and `inside_s` in a row are the per-search means.
`sample_s` is the summed wall. Outside time is wall minus time inside
potential calls. Memory is the in-process peak `RssAnon`, in kB, taken
while the search state is alive.

## Ledger columns

On a B row, `charged` is compared with the sum of `screen`, `full`,
`check`, `closure`, `quench`, `settle`, `lattice`, `reoccupy`, `probe`
and `probe_unbacked`. An unnamed-charge error fails that row. For a
Python cell, `callbacks + internal + probe_unbacked` is compared with
`spent + split_extras`, where `internal` is lattice plus reoccupy plus
quench plus settle. `oracle_calls` is in ledger units: it equals
`spent` when the ledger is present. `call_budget` equals the search
budget passed into that run.

The guard names quench and check charges when the arm's ledger has
those names. On an arm whose ledger has no such split, the named
columns are left blank and are not tested. `probe_unbacked` on the
guard is zero, because that term is counted on the Python boundary.

## Checks

Random draws use seed 20261123 and 10000 seed-level resamples.

1. Ledger. Every B row passes the identities above. A failing row is
   listed with its values.
2. Trajectory. B against A, and B against P, per seed. Best energy
   matches when the absolute difference is at most 1e-6. Differences
   near 1e-9 are inside that window and are not a move. The hit flag
   and `calls_to_hit` match exactly. A miss has an empty
   `calls_to_hit`. Every seed that moves is listed.
3. Bar against A. No hit loss: Fisher's exact test, one-sided, for
   fewer hits on B, fails at p < 0.05. No calls-to-hit loss: a
   one-sided Mann-Whitney test on calls, with a miss scored as
   budget + 1, fails at p < 0.05 when B is larger. Outside time and
   peak `RssAnon` are seed-paired B/A geometric means with a 95 percent
   percentile interval. A regression on either ratio is real only when
   its lower end is above 1 and above the upper end of the A2/A
   interval. The upper end of B/A is also reported against 1.10.

## Verdict

A cell is FAIL when any B row fails the ledger identities, when any
seed moves against A or against P, when either hit test fails, or when
an outside-time or memory regression is real. Otherwise the cell is
PASS. The gate is PASS only when every cell is PASS. A missing arm, a
row error, or a timed sample still under 50 ms after the repetition
cap makes that cell FAIL.
