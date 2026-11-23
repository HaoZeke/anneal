# PR #160 focused TIP4P rerun

This protocol and its harness are committed before any timed result.

## Arms

- A is `cb0cfbfa0cfae21b62cc32037defc5327dc52a51`, tree
  `65972b37b798ebacd9d99d789df761099eefea1f`.
- A2 is an independent release build of the same commit and tree.
- B is `752b938f4c680b72c9785516a2694676e2d7c93b`, tree
  `618d1d0e981d54e2a5a46c9e5b9bb956fe30533b`.

B is the current pull-request head fetched from `pull/160/head`. Its
parent is `a74a31b3fc9a9ba8732ab62d8a7aa39a90ec7baa`.
The only difference from code tree
`a2f52f8621630de6321eb4f2700fbaad5fce0b0e` is
`docs/orgmode/methods/catalog.org`; Cargo inputs, sources, examples,
benches, tests, and the build script have no differences.

Each arm is built once in release mode in a distinct target directory.
The `h2h_seed` source is required to be byte-identical across arms.

## Cells and execution

The cells are tip4p9, tip4p10, tip4p14, tip4p16, and tip4p17. Every cell
uses the fixed paper reference, a 2,000,000-call budget, and fresh seeds
2000 through 2011. No result from an earlier gate is reused.

For even seed offsets, the four positions are A, B, A2, B. For odd seed
offsets, they are B, A2, B, A. Thus A and A2 each have one observation
per seed, while B has two observations per seed and occupies every
position equally. The two B times are averaged within seed before paired
analysis. Deterministic B outcomes must agree within seed.

Processes are fresh, serial, pinned to one core, and receive a clean
single-thread environment. Quick mode is not used. Each row records the
arm, tree, seed, position, outside time, charged calls, hit, in-process
anonymous RSS, executable identity, and all rigid-path counters.

Anonymous RSS is read from `/proc/self/status` by code loaded into the
executed process and emitted at process teardown. A row is valid only if
the reported executable and affinity match the requested run. B must
emit exactly one rigid diagnostic line per run; A and A2 must emit none.

No external competitor executable, source, data, arm, or result is
built, downloaded, read, or run.

## Registered analysis

A hit has best energy at most reference plus 0.001. Per cell, hit
counts are compared with one-sided Fisher exact tests. Charged calls
are compared with a one-sided Mann-Whitney test, with misses scored as
budget plus one.

For each seed, B outside time per charged call is the mean of its two
observations. B/A and A2/A ratios use the geometric mean across seeds.
Confidence intervals are percentile intervals from 100,000 seed
bootstrap resamples using fixed seed 1606. The A2/A interval is the
same-commit noise band. Per-seed ratios and median time by run position
are reported.

The time clause fails when the B/A confidence upper bound exceeds 1.10.
The analysis also reports whether it exceeds the adjusted comparison
limit, the maximum of 1.10 and the A2/A confidence upper bound.

Memory uses the same paired bootstrap construction on anonymous-RSS
ratios. It fails only when the B/A confidence lower bound is above the
maximum of 1.10 and the A2/A confidence upper bound. Lower B memory does
not fail.

No other verdict rules are applied.
