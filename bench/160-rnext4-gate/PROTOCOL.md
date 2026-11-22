# PR #160 r-next4 default-on gate

This protocol is committed before the harness and before any result.

## Arms

- A is `cb0cfbfa0cfae21b62cc32037defc5327dc52a51`, tree
  `65972b37b798ebacd9d99d789df761099eefea1f`.
- A2 is an independent release build of A in another worktree and
  target directory.
- B is the twelve-patch r-next4 mailbox applied with three-way fallback
  to `9837358ae1027067ff654fac8b48969020853452`. Its required tree is
  `81fa4c3de8722fd35eab7311c7e365c2294e5fb6`.

The committed driver sources must be byte-identical across the arms.
The harness is a separate checkout on A. No external GMIN or ASE
program is built, launched, or rerun.

## Cells and seeds

The 28 cells are the prior 26 cells plus tip4p25 and au55-seed:

- tip4p9 through tip4p21, and tip4p25;
- lj38, lj55, lj75, lj98, lj102, lj103, lj104, lj150;
- lj75-seed;
- m55r6, m55r14, m80r6, m80r14;
- au55-seed.

Cheap cells use seeds 2000 through 2011. They are tip4p9 through
tip4p14, lj38, lj55, lj75-seed, m55r6, m80r6, and au55-seed. All
other cells, including tip4p25, use seeds 2000 through 2005.

TIP4P cells use 2,000,000 calls. au55-seed uses 1,000,000 calls and
reference -191.03085280. The other budgets and references are the
paper values fixed in the harness. TIP4P25 uses reference
-1113.0346568571.

## Execution

Every arm is built independently in release mode with the lockfile.
Runs are serial and pinned to core 0. Thread-count variables are one.
The six three-arm orders rotate by seed. Starts are shared by seed.
There are no quick runs, retries, exclusions, or replacements. Rows
are appended and synced one at a time.

The executable is unchanged. A small preloaded probe runs inside the
post-exec process. At process teardown it reads `RssAnon` from
`/proc/self/status`, reads `/proc/self/comm`, and prints both. A row is
valid only when the reported image is the expected driver. No parent
polls memory.

B must print exactly one `rigid_diag` line for every TIP4P run. A and
A2 must print none. The seven fields are recorded. Across the B TIP4P
rows, shell or recorded must be nonzero.

## Registered bars

Hits are derived from best energy at reference plus 0.001. A miss is
charged the full budget.

The hit clause passes only when B has at least as many hits as A. The
one-sided Fisher exact probability for A having more hits and every
paired seed result are reported.

The outside-cost clause passes only when the upper bound of the paired
bootstrap 95% interval for the B/A median outside time per charged call
is at most 1.10. It uses 10,000 deterministic resamples and type-7
quantiles. The A2/A interval is the same-commit noise band and does not
widen the limit.

Memory is one-sided. For each seed, B above both A and A2 counts as
more; B below both counts as less; values inside the A/A2 band are
tied. The memory clause fails only when more exceeds less and the
one-sided exact sign probability is below 0.05. Lower B memory never
fails. Medians and counts are reported.

The overall verdict passes only if all three clauses pass in every
complete cell and every rigid diagnostic requirement passes.

## Builder claims

The analysis checks these claims without changing the registered gate:

- tip4p19 has 3/6 B hits, and seed 2001 hits at 836428 calls;
- tip4p15 has 4/6 B hits;
- B has 4/12 tip4p17 hits on seeds 9200 through 9211;
- tip4p21 has paired B/A outside median 0.979 with interval
  [0.932, 0.993].

The twelve tip4p17 claim seeds are run on B after the panel, serially
on the same core with the same binary and environment.
