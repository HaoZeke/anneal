# TIP4P b11 gate

Status: pre-registered. The rules below are fixed before any gate row is
run or read. Results are appended under "Results" after the campaign.

## Arms

| Arm | Source | Tree | Role |
| --- | --- | --- | --- |
| A | develop `61fa67cf` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | baseline |
| A2 | develop `61fa67cf`, second build in its own worktree and target directory | `cff796c507a8fa1d0bfb28888c3570b61988055e` | A/A noise band |
| B | b6 `53d98cd2` plus the three b11 patches | `75f19ed42a5f3f7311cf670c6b9af15b8eeaa68f` | candidate |
| C | b6 `53d98cd2` | `14b537fd9b73b8fe21e01721954b41a278c5d223` | claim checks only, not gated |

Stop conditions name trees only. The campaign does not start unless the
B tree is `75f19ed42a5f3f7311cf670c6b9af15b8eeaa68f` and the A and A2
trees are `cff796c507a8fa1d0bfb28888c3570b61988055e`. Each arm records
the tree it was built from, and the harness refuses a binary directory
whose recorded tree differs from the expected one.

## Measurement

Every arm is a release build with the 1.88.0 toolchain, `--locked`,
built by `scripts/tip4p_h2h/gate_b11.py build`. Every run is one process
on CPU 0 (`taskset -c 0`), one at a time. Every `*_NUM_THREADS` and
`*_MAX_THREADS` variable, including OMP, OPENBLAS, MKL, RAYON, BLIS,
NUMEXPR, VECLIB and TBB, is set to 1. No shortened budget is used and no
opening run is discarded. Both arms of a seed use the same seed and so
the same start; the harness compares the written start file of every arm
and stops if two differ.

Rigid cells are TIP4P8 through TIP4P21 and TIP4P25, which is TIP4P9 to
TIP4P21 plus the whole rigid-water panel (8, 15, 20, 21, 25). Each cell
runs seeds 2000 to 2011 through `h2h_seed tip4p <n> 0 2000000 <seed>
recommended <stop>`. For n at most 21 the stop is the Cambridge energy in
`data/tip4p_ccd.csv` plus 0.001. TIP4P25 has no Cambridge energy and no
stop.

Within each (cell, seed) block the arm order is A, B, C, A2 for even
seeds and A2, C, B, A for odd seeds. Every pair of arms therefore runs in
ABBA order over consecutive seeds. Cells run in ascending size and seeds
in ascending order. The rigid campaign runs before the non-rigid one.

The non-rigid panel is the thirteen systems of
`scripts/gmin_panel/run_panel.py` at their paper budgets, seeds 2000 to
2002, arms A and B, A first on even seeds and B first on odd seeds. Each
system runs its panel driver (`run <system> <seed> <start file>`). The
Lennard-Jones and Morse systems also run `h2h_seed` at the panel budget
with the reference plus 0.001 as stop, which writes the lowest geometry
and the trace of every new lowest energy. Au55 writes its best structure
through `H2H_BEST`.

## Definitions

- A hit on n at most 21 is `energy_hits(best, reference, 0.001)`. On
  TIP4P25 the reference is the lowest best energy over all 48 TIP4P25
  rows of the four arms.
- Outside time per call is `(wall_s - pot_s) / calls` for one run, with
  `calls` the driver's printed objective call count.
- Medians are Hyndman-Fan type 7 over the twelve seeds of a cell.

## Rules

1. Hits. A cell fails if B has fewer hits than A over its twelve seeds.
   The per-seed hit table of A and B is reported for every cell, with
   each seed where A hits and B misses listed.
2. Outside time. A cell fails if the median outside time per call of B
   divided by that of A is above 1.10. The A/A ratio (A2 over A) is
   reported beside every cell. The cap is not widened by the A/A band.
   If an A/A ratio falls outside 1/1.10 to 1.10 the band is flagged as
   too noisy for that cell, and the flag is reported with the verdict.
3. Non-rigid identity. The rule fails if any non-rigid system and seed
   differs between A and B in any of: panel best energy, hit, calls to
   hit, charged calls, start file digest, Au55 best-structure digest,
   `h2h_seed` best, lowest and geometry energies, calls, charged calls,
   hops, basins, trace digest, or geometry digest.

The gate is PASS only if all three rules pass in every cell.

## Claim checks

These are tested and reported, not gated.

1. b11 reverts the b8 chart and keeps b9 and b10: the b6-to-b11
   production diff is checked for chart code and for the paid-check
   buffer reuse only.
2. Serial B/C outside-time ratios lie in 0.976 to 1.009, pooled 0.993.
3. B and C have identical hits, and TIP4P15 to TIP4P21 total 23/84.
4. The TIP4P9 to TIP4P21 rows are identical between B and C, 156 of 156,
   on hit, calls, charged calls, best energy, basins, trace digest and
   geometry digest.

## Extensions

The b6 report, the second seed set (2012 to 2023) and the bisect are
pre-registered in `gates/anneal-tip4p-b11/README.md`. The pass rules for
B above are unchanged by them.

## Results

Verdict: FAIL. Rule 1 fails, rule 2 fails, rule 3 passes.

Rows are `gates/anneal-tip4p-b11/rows.csv`, 1,110 rows: 720 primary
rigid, 168 second-set, 150 non-rigid and 72 probe. Per-cell values are
`gates/anneal-tip4p-b11/summary.json`. Every row carries the hashed
machine id `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
and core 0. Every TIP4P row has `calls == charged`, so outside time per
call and per charged call are the same number. Starts agree across arms
on every seed. The A and A2 binaries are byte-identical, so A/A measures
run-to-run noise only.

The host stopped the machine three times during the primary campaign.
The rows recorded across those stops are TIP4P16 seed 2009 A2, TIP4P18
seed 2000 B and TIP4P19 seed 2001 C. Their `pause_gap_s` is about 2,362 s,
and each is 17 to 25 per cent above its cell's median outside time per
call. They are kept. Leaving any one out moves its cell median by at most
0.021 us per call, which changes no verdict. The only other gap above 5 s
is the first Morse row, at 38 s, where the non-rigid campaign was
restarted after a harness fix. No row was rerun, dropped or edited.

### Rule 1 and rule 2, primary seeds 2000 to 2011

Outside times are type-7 medians in microseconds per charged call. Ratios
are medians over A's median. The A hit, B miss and A miss, B hit columns
list seeds.

| Cell | A | A2 | B | b6 | A hit, B miss | A miss, B hit | A us/call | A2/A | B/A | b6/A | B/b6 | Rule 1 | Rule 2 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TIP4P8 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.2792 | 1.0116 | 0.9900 | 1.0396 | 0.9523 | pass | pass |
| TIP4P9 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.3334 | 1.0050 | 1.0027 | 1.0158 | 0.9871 | pass | pass |
| TIP4P10 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.4591 | 0.9982 | 0.9988 | 1.0075 | 0.9914 | pass | pass |
| TIP4P11 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.5703 | 0.9999 | 0.9937 | 0.9990 | 0.9947 | pass | pass |
| TIP4P12 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.7970 | 1.0001 | 0.9896 | 0.9926 | 0.9970 | pass | pass |
| TIP4P13 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 1.9088 | 1.0049 | 1.0132 | 1.0370 | 0.9770 | pass | pass |
| TIP4P14 | 12/12 | 12/12 | 12/12 | 12/12 | - | - | 2.1599 | 0.9936 | 1.0190 | 1.0170 | 1.0020 | pass | pass |
| TIP4P15 | 7/12 | 7/12 | 5/12 | 5/12 | 2007 2010 | - | 2.0700 | 0.9549 | 1.1402 | 1.1320 | 1.0073 | fail | fail |
| TIP4P16 | 2/12 | 2/12 | 1/12 | 1/12 | 2000 | - | 2.1776 | 1.0390 | 1.2233 | 1.2705 | 0.9628 | fail | fail |
| TIP4P17 | 5/12 | 5/12 | 3/12 | 3/12 | 2003 2006 | - | 2.4339 | 1.0126 | 1.2622 | 1.2703 | 0.9936 | fail | fail |
| TIP4P18 | 2/12 | 2/12 | 3/12 | 3/12 | - | 2003 | 2.3400 | 0.9498 | 1.3297 | 1.3155 | 1.0108 | pass | fail |
| TIP4P19 | 7/12 | 7/12 | 7/12 | 7/12 | 2001 | 2010 | 2.6152 | 0.9906 | 1.2136 | 1.2302 | 0.9865 | pass | fail |
| TIP4P20 | 6/12 | 6/12 | 4/12 | 4/12 | 2006 2010 | - | 2.7914 | 0.9922 | 1.1913 | 1.2108 | 0.9838 | fail | fail |
| TIP4P21 | 0/12 | 0/12 | 0/12 | 0/12 | - | - | 3.0087 | 1.0135 | 1.2653 | 1.2882 | 0.9823 | pass | fail |
| TIP4P25 | 1/12 | 1/12 | 1/12 | 1/12 | - | - | 4.1411 | 0.9927 | 1.2299 | 1.2504 | 0.9836 | pass | fail |

The TIP4P25 reference is -1113.0346568703, reached on seed 2010 by every
arm. The A/A ratio lies in 0.9498 to 1.0390, inside 1/1.10 to 1.10 in
every cell, so no cell is flagged as noisy. Every B/A ratio from TIP4P15
up, 1.1402 to 1.3297, is above both the cap and the A/A band.

Per-seed hits, seeds 2000 to 2011 left to right. A2 equals A and b6
equals B in every cell.

| Cell | A | B |
| --- | --- | --- |
| TIP4P8-14 | 111111111111 | 111111111111 |
| TIP4P15 | 110101011010 | 110101001000 |
| TIP4P16 | 100000100000 | 000000100000 |
| TIP4P17 | 001100101001 | 001000001001 |
| TIP4P18 | 100000000010 | 100100000010 |
| TIP4P19 | 110001110101 | 100001110111 |
| TIP4P20 | 011010100011 | 011010000001 |
| TIP4P21 | 000000000000 | 000000000000 |
| TIP4P25 | 000000000010 | 000000000010 |

Develop's 29/84 on TIP4P15 to TIP4P21 is stable: A and A2 have the same
hit on every seed of every cell. B and b6 have 23/84.

### Second seed set, 2012 to 2023

| Cell | A | B | A hit, B miss | A miss, B hit | A us/call | B us/call | B/A | Rule 1 | Rule 2 |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| TIP4P15 | 8/12 | 4/12 | 2015 2017 2018 2023 | - | 1.8324 | 2.3005 | 1.2555 | fail | fail |
| TIP4P16 | 5/12 | 4/12 | 2012 2015 2018 | 2019 2020 | 2.1133 | 2.5645 | 1.2135 | fail | fail |
| TIP4P17 | 4/12 | 3/12 | 2023 | - | 2.1601 | 2.8588 | 1.3235 | fail | fail |
| TIP4P18 | 3/12 | 1/12 | 2016 2017 | - | 2.4574 | 3.0598 | 1.2451 | fail | fail |
| TIP4P19 | 5/12 | 5/12 | - | - | 2.6723 | 3.2276 | 1.2078 | pass | fail |
| TIP4P20 | 0/12 | 1/12 | - | 2013 | 2.6885 | 3.4180 | 1.2713 | pass | fail |
| TIP4P21 | 1/12 | 0/12 | 2012 | - | 2.9998 | 3.7644 | 1.2549 | fail | fail |

A has 26/84 and B 18/84. These runs have no target stop, so each spends
the full budget and the outside times are not on the same call counts as
the primary set.

Pooled with the primary seeds, n = 24 per cell. `b` is A hit, B miss and
`c` is A miss, B hit. The p value is the exact two-sided McNemar test.

| Cell | n | A hits | B hits | b | c | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TIP4P15 | 24 | 15 | 9 | 6 | 0 | 0.03125 |
| TIP4P16 | 24 | 7 | 5 | 4 | 2 | 0.6875 |
| TIP4P17 | 24 | 9 | 6 | 3 | 0 | 0.25 |
| TIP4P18 | 24 | 5 | 4 | 2 | 1 | 1 |
| TIP4P19 | 24 | 12 | 12 | 1 | 1 | 1 |
| TIP4P20 | 24 | 6 | 5 | 2 | 1 | 1 |
| TIP4P21 | 24 | 1 | 0 | 1 | 0 | 1 |
| TIP4P15-21 | 168 | 55 | 41 | 19 | 5 | 0.006611 |

### Rule 3, non-rigid panel

All 13 panel systems, seeds 2000 to 2002, are identical between A and B
on every compared field: 39 panel-driver pairs and 36 `h2h_seed` pairs
on the Lennard-Jones and Morse systems. The Au55 best structures match.
Energies are compared as the drivers print them, eight decimals from the
panel drivers and ten from `h2h_seed`. Geometries are compared as the
digest of the written file.

### Bisect

b6 is bad under the pre-registered rule. Its outside time is 1.1320 of A
on TIP4P15, and it has 3/12 hits on TIP4P17 against A's 5/12 and 23/84
on TIP4P15 to TIP4P21 against 29/84. The range is therefore the
first-parent commits of `61fa67cf..53d98cd2`: `b79621a4` and `53d98cd2`.
b6 is the end of the range and its primary rows decide it, so one probe
was needed.

| Probe | Tree | TIP4P9 P/A | TIP4P15 P/A | TIP4P17 hits A, P | TIP4P9 hits A, P | TIP4P15 hits A, P | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `b79621a4` | `39391d3a3dbe4fd81959b56882429c5ef0165914` | 1.0102 | 1.1898 | 5, 3 | 12, 12 | 7, 5 | bad |
| `53d98cd2` (b6, primary rows) | `14b537fd9b73b8fe21e01721954b41a278c5d223` | 1.0158 | 1.1320 | 5, 3 | 12, 12 | 7, 5 | bad |

The probe's median outside times are 1.2985 and 1.3118 us per call on
TIP4P9 for A and P, 1.9423 and 2.3108 on TIP4P15, and 2.2216 and 2.8582
on TIP4P17. Its P rows equal the primary b6 rows, and its A rows equal
the primary A rows, on all 36 seeds in hit, calls, energy and trace.

First bad commit: `b79621a4` "feat(search): generalize stalled rigid
archive walks". It is the first commit after develop, and its child
`53d98cd2` adds only a newsfragment.

```text
 examples/globalopt_profile.rs               |   3 +-
 src/methods/cluster_hopping.rs              | 417 +++++++++++++++++++++------
 src/methods/cluster_hopping/cycles.rs       | 421 +++++++++++++++++++++++++++-
 src/methods/cluster_hopping/moves.rs        |  21 +-
 src/methods/cluster_hopping/stall_bridge.rs | 110 ++++++++
 src/potentials/gmin_tip4p.rs                |   8 +-
 src/potentials/tip4p.rs                     |   8 +-
 src/python.rs                               |   8 +-
 src/rigid_body.rs                           |  24 ++
 9 files changed, 919 insertions(+), 101 deletions(-)
```

The cost and the hit drop already appear at b6, so they predate b8, b9
and b10.

### Claim checks

1. Reverts b8 and keeps b9 and b10. Confirmed on the code. The b6-to-b11
   production diff touches only `src/methods/cluster_search.rs`, and only
   to reuse the paid-check buffers. No scaled chart code is present. The
   b8 counters (392 ns per call) and the call-2 divergence were not
   re-measured here, because there is no b8 arm.
2. Serial A/B 0.976 to 1.009, pooled 0.993. That comparison is b11 over
   b6. Measured B/b6 is 0.9523 to 1.0108 on TIP4P8 to TIP4P21, pooled
   0.9925. The pooled figure holds. The range is wider at both ends,
   0.9523 on TIP4P8 and 1.0108 on TIP4P18. Against develop, B is 1.2900
   of A pooled over all primary rigid rows, with A2/A at 1.0079.
3. Identical hits, TIP4P15 to TIP4P21 at 23/84. Confirmed against b6:
   B and b6 hits match in every cell and both have 23/84. Develop has
   29/84, so the claim does not mean no hit loss.
4. 156/156 rigid rows bit-identical. Confirmed against b6: all 156
   TIP4P9 to TIP4P21 rows match in hit, calls, charged calls, energy,
   search best, basins, trace digest and geometry digest, as do the
   TIP4P8 and TIP4P25 rows. Only 77 of the 156 match between develop and
   b6.

### Verdict by rule

| Rule | Verdict | Cells that fail |
| --- | --- | --- |
| 1, no hit loss | FAIL | primary TIP4P15, 16, 17, 20; second TIP4P15, 16, 17, 18, 21 |
| 2, outside time at most 1.10 of A | FAIL | primary TIP4P15 to TIP4P21 and TIP4P25; second TIP4P15 to TIP4P21 |
| 3, non-rigid identity | PASS | none |

b11 is trajectory- and cost-neutral against b6, but b6 carries a 13 to
33 per cent outside-time increase per call and a hit loss against
develop on TIP4P15 and up. The bisect places both at `b79621a4`.
