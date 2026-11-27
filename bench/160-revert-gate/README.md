# PR #160 revert same-runner gate

This protocol is fixed before any timed row is recorded.

## Arms

- A is commit `61fa67cf380e46876b569da45e7cc091647a775b`, tree
  `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- A2 is a separate release build of the same commit and tree, in a separate
  worktree and target directory.
- B is the PR #160 head plus the one-patch revert, tree
  `0e469d0f9e49deb0083332c72316d29918626350`.

The three driver source files must have identical sha256 digests on every
arm. Each arm is built separately in release mode with the lockfile.

## Panel

TIP4P9 through TIP4P21 use seeds 2000 through 2023 and a budget of
2,000,000 charged calls. References come from `data/tip4p_ccd.csv`.

The recorded Lennard-Jones and Morse panel uses seeds 2000 through 2011:

| Cell | Budget | Reference |
| --- | ---: | ---: |
| lj38 | 400000 | -173.928427 |
| lj55 | 1000000 | -279.248470 |
| lj75 | 4000000 | -397.492331 |
| lj98 | 4000000 | -543.665361 |
| lj102 | 4000000 | -569.363652 |
| lj103 | 4000000 | -575.766131 |
| lj104 | 4000000 | -582.086642 |
| lj150 | 4000000 | -893.310258 |
| lj75-seed | 4000000 | -397.492331 |
| m55r6 | 1000000 | -250.286609 |
| m55r14 | 1000000 | -220.646208 |
| m80r6 | 4000000 | -378.333471 |
| m80r14 | 4000000 | -340.811371 |

The recorded panel already includes Morse cells, so no cells are added.

## Execution

Runs are serial and pinned to core 0. Every thread-count variable is one.
Each process receives a clean environment. The six arm permutations rotate
by seed, beginning `A,B,A2` then `A2,B,A`. There are no retries, warm-up
exclusions, replacements, or result-dependent changes.

Each row records the seed, cell, arm, tree, exit code, energy, calls, wall
time, outside time per charged call, affinity, peak `RssAnon`, and
`ru_maxrss`. Hits are re-derived from energy at reference plus 0.001.
Driver hit flags are not used. A miss has infinite time to hit.

## Registered decision

Each cell is judged independently.

1. No hit loss requires B hits to be at least A hits. The report includes
   the one-sided exact Fisher p-value for the alternative that A has the
   greater hit rate.
2. Outside cost requires the B/A ratio of median outside time per charged
   call to be at most 1.10. The report includes a deterministic paired
   bootstrap 95% percentile interval with 10,000 resamples and a one-sided,
   tie-corrected Mann-Whitney p-value for B being slower. The A2/A ratio and
   interval are reported as the same-runner band.

Peak anonymous RSS is judged by whether the B median lies within the
observed A-to-A2 median band. It is reported separately and does not alter
clauses 1 or 2. `ru_maxrss` is reported and never gates.

The overall verdict is PASS only when both registered clauses pass in
every complete cell. Failures and misses remain in the record as observed.

For TIP4P17 seeds 2000 through 2003, the report separately checks whether
B and A have bit-identical energies, equal charged calls, and identical
final-coordinate sha256 digests.
