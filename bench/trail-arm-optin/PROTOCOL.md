# Trail-arm opt-in gate

This gate compares the idle trail-arm redraw change with its merge base.
It is an opt-in test: every search sets `escape_stall_patience` to one.
No production library source is committed by this benchmark.

## Frozen revisions

- A: `364679f060386949f805a72dea9d7d63b7e9ea4f`
- B: `102bb44d712242a889c45c6eae10f6b89f66d2f8`
- B tree: `dd7f1311540d512dc0e1a7f1cff9a830394e5f89`
- A2: an independent build of A

Each arm is built in a separate worktree with the release profile. The
probe source and tracing transform in this directory are copied unchanged
to all worktrees. The transform only counts stall predicates and changed
arm selections, then prints those counts after the search.

Builds and runs are serial, pinned to one core, with all recognized
`*_NUM_THREADS` controls set to one. GMIN programs and GMIN arms are
excluded. The driver uses `Config::recommended`, changes only
`escape_stall_patience` to one, and uses the normal factor of 2.0.

## Panel

| cell | budget | seeds |
| --- | ---: | --- |
| lj13-panel | 80000 | 2000 through 2011 |
| lj38-panel | 80000 | 2000 through 2011 |
| lj13-reach | 80000 | 1, 2, and 3 |

Each seed is run once on A, B, and A2. Adjacent seeds reverse the
three-arm order: `A,B,A2`, then `A2,B,A`. This places B between the two
independent control builds and is the three-arm ABBA interleave.

Rows are appended and never rewritten. A row includes the arm revision
and tree, binary digest, anonymous machine identifier, core, thread
count, hit, calls to hit, charged calls, final energy and its bits,
redraw count, stall-trigger count, externally measured wall time, and
in-process `RssAnon`. A miss has calls to hit equal to budget plus one
for analysis. The raw charged value remains unchanged.

The anonymous machine identifier is the SHA-256 digest of the stable
machine identifier. Timing uses a monotonic clock around the child
process. Memory is read by the child from its process status after the
search.

## Analysis

The analysis reports redraw totals and refuses a PASS when B has no
redraws. It compares A, B, and A2 hit counts with two-sided Fisher exact
tests and calls to hit with two-sided Mann-Whitney tests. Misses count as
budget plus one.

For each cell, paired ratios use:

`exp(mean(log(right_seed / left_seed)))`.

Outside-time and `RssAnon` confidence intervals use 10000 paired
bootstrap samples from seed 20261123 and type-7 2.5 and 97.5 percent
quantiles. Ratios are reported for B/A and A2/A.

The verdict requires no hit loss. A time regression is real only when
the B/A lower confidence limit exceeds both one and the A2/A upper
limit. Both the strict 1.10 bar and the
`max(1.10, A2/A upper)` bar are reported. The one-sided memory gate
fails only when the B/A lower confidence limit exceeds
`max(1.10, A2/A upper)`.

Trajectory differences compare hit, calls, charged work, final-energy
bits, redraws, and stall triggers by seed. Early stopping is flagged
because per-call timing can be distorted when arms stop after different
amounts of work; aggregate charged calls and wall time are reported.

The verdict is PASS when the path is exercised and all gates pass, FIX
when a gate fails, and PARTIAL when required evidence is missing.
