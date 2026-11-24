# Restack gate

This file is committed before any timed row. Later commits may append
rows and the registered summary. They may not rewrite a row that was
already written, and they may not change a bar.

## Arms

Four release builds, one session, one process at a time, pinned to one
core. Every thread-count variable is 1.

- A: develop `9b65fda3972fd8c014bce886a3ea5c05956d64af`, tree
  `a49a66d1422cfefa28012fd12e6729bc4e879d5f`.
- A2: an independent release build of that same tree.
- B: one squash of the packings series onto that develop. The tree is
  `6b5df2e6f262b86e74023f94f3ebcbadc1eb17b6`. The squash commit id is
  not an identity.
- P: `168156802e9985fba0523e440588d513e040a6ff`, tree
  `8607842fdb1551f9da6b4e302ad0324c7167ed9a`.

The bench patches apply on each arm and do not change `src/`,
`examples/`, or `scripts/`. After those patches the trees are
`5640e87329128bba10d3cbb6b09dff2cad91fcff` (A),
`88603978d74b669f75f11b770975982ac0a6e09d` (B), and
`5826c8b8a26a1342c7214c9983ca077d38d96233` (P). Binaries are built from
the pre-patch trees above.

Cluster binaries carry the measurement sampler at layout 0. On the
develop trees the walk lives in the search body and the outer function
only dispatches, so the sampler starts in the search body. A and A2
are built without the generic marker. B and P are built with it. TIP4P
binaries are separate uninstrumented builds of `h2h_seed`. The focused
runner checks the A head and the A and B trees above. It has no P arm.

No external competitor binary is built or run.

## Cells

Twelve seeds, 2000 through 2011. Paper budgets. No shortened run.

Default: `lj38` 400000, `lj75` 4000000, `lj98` 4000000, `m55r6`
1000000.

Probe: `lj55` 1000000, `m38r14` 400000, `m38r6` 400000, `sc126-38`
400000, `sc126-55` 1000000, `sc108-38` 400000, `sc96-38` 400000,
`sc96-55` 1000000.

TIP4P: `tip4p9`, `tip4p10`, `tip4p14`, `tip4p16`, `tip4p17`. Budget
2000000, recommended configuration, h2h seed `tip4p`, target equal to
the reference plus 0.001.

Journal order is the stall-heavy probe cells, then the other cluster
cells, then TIP4P:

`m38r14`, `sc126-38`, `sc126-55`, `sc108-38`, `sc96-38`, `sc96-55`,
`lj38`, `lj55`, `m38r6`, `m55r6`, `lj98`, `lj75`, then the five TIP4P
cells in the order above.

Within a cell every seed finishes before the next cell. Even seeds run
A, B, P, A2. Odd seeds run A2, P, B, A. The stock cluster driver
requires the flag order D, D2, G, P and would otherwise run that order.
This gate executes D, G, P, D2 on even seeds and the reverse on odd
seeds. D is A, D2 is A2, G is B, and P is P.

TIP4P uses the focused runner's own order. Even offsets are A, B, A2,
B. Odd offsets are B, A2, B, A. The two B times are averaged within a
seed before a ratio. Deterministic B fields must agree within a seed.

## Rows

Each cluster row records evaluations to the hit, charged calls, hit,
printed best energy, wall and potential time, outside time, outside
time per charged call, in-process RssAnon peak, and the packing offered
and probe counters. A row is valid only when that peak is nonzero. The
sampler has no separate path counter; a zero peak means the status path
did not report.

Each TIP4P row records the focused runner's fields. Its in-process
RssAnon value is the teardown reading, not a sampled peak. A zero
reading is invalid. Both B repetitions are kept.

The machine field is the sha256 of the CPU model string. It is not a
host name. Rows are append-only. A stopped session is sealed as the
rows written so far.

## Bars

A miss is scored as budget plus one for the call tests. Ratios use a
seed-paired geometric mean. Each interval draws 10000 resamples from a
generator seeded with 20261123, restarted for that interval, and takes
the 2.5 and 97.5 percentiles with Hyndman-Fan type 7.

Hit loss fails when B has fewer hits than A, or fewer than P on a
cluster cell. The report includes a two-sided Fisher exact test and a
two-sided Mann-Whitney U test on the scored calls.

Outside time fails the strict bar when the upper bound exceeds 1.10.
The adjusted comparison, upper bound against the maximum of 1.10 and
the A2/A upper bound, is reported and does not relabel a strict
failure. A real time regression is a lower bound above 1 and above the
A2/A upper bound.

Per-call outside time is reported with the same interval and is not the
time bar. When its upper bound exceeds 1.10, the total-time upper bound
does not, and B's call geometric mean is below A's, the cell is flagged
as an early-stop per-call artifact.

RssAnon fails only when its lower bound exceeds the maximum of 1.10 and
the A2/A upper bound. A lower reading does not fail. A real memory
regression uses the same lower-bound rule as time.

Packings are expected to move default and probe cells off A. Those
cells are not failed for differing from A. B is judged against P on
hits, evaluations, charged calls, calls to the hit, and best-energy
bits. The previous gate's published B anchors, from tree
`d62db74207dfea1071579bf2121c4ebd4f337baa`, are a second deterministic
reference and are reported seed by seed where that writeup named them.

TIP4P requires B to match A on hits, calls, and charged counts. Energies
within 1e-9 are agreement. A bit difference at that scale is noise and
is not a failure.

## Heated lattice exchange

Checked from the sources before timed rows are interpreted. The cold
lattice-exchange fingerprint is `0x100885ac118293e1` on both P and B.
The heated fingerprint is `0x44038697bde16fb7` on P and
`0x1a623a8e37ad1d43` on B. The lattice proposal still scales as heat
times the preset temperature. The restack changes the heated walk only.
Stall-heavy cells, `m38r14` and the `sc` rows, are where that
interaction can move B off P. A unit test on each tree is recorded with
the summary.
