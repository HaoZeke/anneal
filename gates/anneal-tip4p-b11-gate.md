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

Pending.
