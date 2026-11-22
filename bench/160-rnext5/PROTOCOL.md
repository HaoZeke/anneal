# PR #160 r-next5 default-on gate

This protocol and its harness are committed before any result.

## Arms

- A is `cb0cfbfa0cfae21b62cc32037defc5327dc52a51`, tree
  `65972b37b798ebacd9d99d789df761099eefea1f`.
- A2 is an independent release build of A in another worktree and
  target directory.
- B is the fourteen-patch r-next5 mailbox applied with three-way fallback
  to `9837358ae1027067ff654fac8b48969020853452`. Its required tree is
  `a2f52f8621630de6321eb4f2700fbaad5fce0b0e`.

The committed driver sources must be byte-identical across the arms.
The harness is a separate checkout on A.

## Anneal-only source check

The `h2h_gmin_proof run` source was checked before timed runs. It constructs
`Config::recommended`, a Lennard-Jones `PairPotential`, a `Ledger`, and
`WarmLbfgs`, then calls anneal's `run_with_gradient`. The run command has no
GMIN option, executable lookup, GMIN path, or GMIN data-file read. Its only
subprocess is `git`, used to print commit metadata. The optional path
argument receives generated anneal start coordinates; it is not read.

Only the `run` and `coords` commands are used. Test-only comparison helpers
and the result summarizer are not invoked. No external GMIN executable,
data file, arm, or result row is built, read, launched, or produced.

## Cells and seeds

The 28 cells are:

- tip4p9 through tip4p21, and tip4p25;
- lj38, lj55, lj75, lj98, lj102, lj103, lj104, lj150;
- lj75-seed;
- m55r6, m55r14, m80r6, m80r14;
- au55-seed.

Cheap cells use seeds 2000 through 2011. They are tip4p9 through tip4p14,
lj38, lj55, lj75-seed, m55r6, m80r6, and au55-seed. All other cells use
seeds 2000 through 2005.

TIP4P cells use 2,000,000 calls. au55-seed uses 1,000,000 calls and
reference -191.03085280. The other budgets and references are the fixed
paper values. TIP4P25 uses reference -1113.0346568571.

## Execution

Every arm is built independently in release mode with the lockfile. Runs
are serial and pinned to core 0. Thread-count variables are one. The six
three-arm orders rotate by seed. Starts are shared by seed. There are no
quick runs, retries, exclusions, or replacements. Rows are appended and
synced one at a time.

A small preloaded probe runs inside the post-exec process. At process
teardown it reads `RssAnon` from `/proc/self/status`, reads the executable
image from `/proc/self/comm`, and prints both. A row is valid only when the
reported image is the expected driver. No parent polls memory.

B must print exactly one `rigid_diag` line for every TIP4P run. A and A2
must print none. The seven fields are recorded. Across B's TIP4P rows,
shell or recorded must be nonzero.

## Registered bars

Hits are derived from best energy at reference plus 0.001. A miss is
charged the full budget. The hit clause passes only when B has at least as
many hits as A. The one-sided Fisher exact probability and every paired
seed are reported. Any seed where A hits and B misses is listed.

For each seed, outside time per charged call is divided as B/A. The
outside-cost estimate is the arithmetic mean of those paired ratios. Its
upper bound is mean plus the two-sided 95% Student t critical value times
the sample standard error. The clause passes only when that upper bound is
at most 1.10. A2/A is analyzed identically as the same-commit noise band.
Per-arm medians are also reported.

Memory is one-sided. For each seed, B above both A and A2 counts as more;
B below both counts as less; values inside the A/A2 band are tied. The
memory clause fails only when more exceeds less and the one-sided exact
sign probability is below 0.05. Lower B memory never fails.

The overall verdict passes only if all three clauses pass in every complete
cell and every rigid diagnostic requirement passes.

## Builder claim

The analysis explicitly reports LJ150. It checks whether seed 2000 has
identical A and B best-energy bits and charged calls, and whether the
outside medians reproduce 22.455 and 23.087 microseconds per call.
