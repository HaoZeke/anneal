# Morse 38 at rho 6 after the stalled-hold lift

The median calls to a hit on Morse 38 at rho 6 rose from 100,636.5 to
115,642 between the two revisions below. This directory pairs the runs
seed by seed:

| Arm | Revision | Tree | Code |
| --- | --- | --- | --- |
| B | `7fff9f2e` | `113a3420` | the held plain arm never lifts |
| R7 | `90433050` | `788f2a13` | the hold lifts once the plain walk leaves or stalls |
| C1 | `90433050` with `resume-retry.diff` | `00d2d3c4` | R7, but a failed retry returns the chain to where it opened |

R7's `code_revision` is the commit as it was run. Applying the patch
again gives a new commit id with the same tree, so `code_tree` is the
identifier that holds. C1 is R7 with `resume-retry.diff` applied; it was
not committed.

Every run is `h2h_seed morse 38 6 400000 <seed> recommended -157.476108`,
seeds 2000 to 2023: release build, one thread per run, six runs at a time
on one 8-core x86_64 virtual machine. Every run of every arm hits.
`python3 bench/168-r8-m38r6/stats.py` recomputes the table from `rows.csv`.
Each test is exact and two-sided over all sign flips of the pairs that
differ:

| arm | differ | slower | median B | median arm | p median | W+ of 66 | p Wilcoxon | p sign | geometric mean ratio | p ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| R7 | 11 | 8 | 100636.5 | 115642.0 | 0.2500 | 54 | 0.0674 | 0.2266 | 1.1210 | 0.0820 |
| C1 | 11 | 9 | 100636.5 | 116609.0 | 0.2500 | 56 | 0.0420 | 0.0654 | 1.1151 | 0.0518 |

The 13 seeds where R7 and B agree run call for call the same. On each of
the other 11, the hold that follows the first failed restraint block
outlasted the run it began in and handed a block back to the restraint:
once on 8 seeds, twice on 2003 and 2013, three times on 2012. No seed
lifted because the plain walk left the incumbent. After the first lift
the two chains differ, so the calls to a hit are a fresh draw from a
heavy-tailed spread, and 3 of the 11 seeds got faster.

The rise in the median is not distinguishable from noise (p = 0.25). The
paired differences lean slower (Wilcoxon p = 0.067), as expected from the
restraint block each lift spends. C1 tested whether the revert that
follows a failed retry costs more than that block: with `before_gain`
spent by the first revert, R7 puts the chain back on the incumbent and
drops the plain walk's progress. Resuming that walk instead does not
lower the median, and leans slower still (Wilcoxon p = 0.042), so R7
stays as it is.
