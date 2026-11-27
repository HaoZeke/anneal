# Lazy basin gate

This protocol is fixed before any measurement row from this gate.

## Question

Does deferring basin identity, and caching lattice radii while a
cloud grows, keep the search bit-identical to the previous head and
stay inside the pre-registered cost bars?

## Arms

One session, one machine, four release builds.

- A is develop `364679f060386949f805a72dea9d7d63b7e9ea4f`, tree
  `79610c980d39f19a2d383ffe3c53e4505ddb9b6c`.
- A2 is an independent release rebuild of that same commit and tree.
- B is that develop commit merged with lazy tree
  `0467d399182a809d06661f1ae1903553833b1e8e`. The merge tree is
  `9d40e137e4f20cd0a8401b40b0b2f06f7521716e`.
- B0 is that develop commit merged with
  `6b12b9aab1e78589f0634413aae8c12611104607`. The merge tree is
  `be6101476d3999e9232e71016b7871c17ca52a4d`. B0 is the bit-identity
  reference only. It is not part of the timed panel.

Each arm is `cargo build --release --locked` with the example feature
set used by the earlier gate. A and A2 are separate target directories
and separate instrumentation passes. Binaries are pinned to one core
for every timed launch. Every thread-count control used by the runner
is set to 1. No reduced budget and no external minimizer is used.

## Cells

Seeds start at 2000.

- `tip4p8` and `lj38` use 24 seeds (`2000..2023`).
- The other cells use 12 seeds (`2000..2011`).

The harness panel has no `lj100` cell. The LJ cells from `lj75`
through `lj150` are `lj75`, `lj98`, `lj102`, `lj103`, `lj104`, and
`lj150`. The Morse cells are `m38r14`, `m80r10`, and `m80r14`.

Paper budgets and recommended settings are unchanged. Fast cells are
run first so a stopped session can still seal finished cells.

## Order

For each cell, each seed runs the panel in this six-seed block, and
the block repeats:

1. A, B, A2
2. B, A2, A
3. A2, A, B
4. B, A, A2
5. A, A2, B
6. A2, B, A

A and B therefore alternate in ABBA order across consecutive seeds.
After that cell's panel finishes, B0 runs the same seeds. B0 does not
sit inside the ABBA block. Rows are append-only. A failed launch is
kept and is not replaced.

## Recorded fields

The row schema matches the paired follow-up: in-process RssAnon,
outside time, charged calls, the basis-walk counter, and the final
hashes. The final hashes are the exact energy bits of the reported
best, the SHA-256 of that structure's coordinate bits, the exact bits
of the final energy, and the SHA-256 of the final structure.

## Bars

A finished cell is FAIL when any bar below fails. A FAIL is not
relabeled when another comparison would pass. An unfinished cell is
PARTIAL, not PASS.

1. Hit loss. `hits(B) < hits(A)` fails the cell. The two-sided Fisher
   exact p is reported.
2. Outside time per charged call. For each seed, form the B/A ratio.
   Report the seed-paired bootstrap 95% interval of the mean ratio and
   the paired Student 95% interval on the log ratios, transformed back
   to the ratio scale. The cell fails when the larger of the two upper
   bounds is above 1.10. Also show that upper bound against
   `max(1.10, larger A2/A upper bound)`. The second number does not
   turn a FAIL into a PASS.
3. RssAnon, one sided. Form the same two intervals for the B/A ratio
   of in-process RssAnon. The cell fails only when the smaller of the
   two lower bounds is above `max(1.10, larger A2/A upper bound)`.

Charged calls are reported with a two-sided Mann-Whitney test. A miss
is scored as `budget + 1`. The test is not a bar. Median scored calls
and median raw charged calls are both reported. Total wall B/A uses
the same intervals and is not a bar. The count of rows with
`basis_walks > 0` is reported and is not a bar.

## Bit identity

On every finished seed, B and B0 must agree on hit, charged calls, best
energy bits, structure hash, final energy bits, and final structure
hash. Each disagreement is reported with its seed and field. Any
disagreement fails the bit-identity claim. A missing seed leaves the
claim incomplete; it is not called a pass.
