# Basin-bound fix gate

Three release builds, one session, one core. This file states the rule
before any timed row is recorded.

## Arms

- Arm A is tree `f2786b12dc42dab75994a0380e10bd3ff4ba0dc7`.
- Arm A2 is a separate release rebuild of that same tree, in its own
  worktree and target directory. A2 against A is the noise band.
- Arm B is develop `364679f060386949f805a72dea9d7d63b7e9ea4f`. That
  tree contains `57f4fc120376788c9afebd14dc48754ff1bd56c9`, the
  merge-radius overflow fix. No other tree may stand in for B.

Every arm is a locked release build with `rgpot-ex`, the same
compiler and the same flags. The measurement overlay is the same
source on every arm. It samples `RssAnon` in the benchmark process
and counts entries to `blocks_separate`. The count is one relaxed
atomic on that path. The driver prints it from `Sampler::finish`,
after `wall_s` has been stored, so the report is outside the timed
region. The overlay hash and every binary hash are journalled.

## Panel

The probe visits every cell of the previous basin-bound panel. Budgets
are the paper list: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and above.
TIP4P uses 2e6 for every size from 8 through 21 and stops at the
reference plus 0.001.

| Cell | Driver | Budget | Reference |
| --- | --- | ---: | ---: |
| lj38 | h2h_gmin_proof | 400000 | -173.928427 |
| lj75 | h2h_gmin_proof | 4000000 | -397.492331 |
| lj98 | h2h_gmin_proof | 4000000 | -543.665361 |
| lj102 | h2h_gmin_proof | 4000000 | -569.363652 |
| lj104 | h2h_gmin_proof | 4000000 | -582.086642 |
| blj105n38 | h2h_blj | 400000 | -177.2090846748 |
| blj105n55 | h2h_blj | 1000000 | -283.1545313810 |
| blj115n38 | h2h_blj | 400000 | -187.3676867453 |
| blj115n55 | h2h_blj | 1000000 | -289.2931853058 |
| m55r6 | h2h_morse | 1000000 | -250.286609 |
| m55r14 | h2h_morse | 1000000 | -220.646208 |
| m80r6 | h2h_morse | 4000000 | -378.333471 |
| m80r14 | h2h_morse | 4000000 | -340.811371 |
| sc126-38 | h2h_sc | 400000 | -35419.9804 |
| sc126-55 | h2h_sc | 1000000 | -52884.6806 |
| tip4p8 .. tip4p21 | h2h_seed | 2000000 | data/tip4p_ccd.csv |

## Probe and cell choice

Before any timed row, one untimed probe runs seed 2000 at budget
20000 on all 29 cells, arm A then arm B. Those runs are not gate
rows. A cell fires when arm B's `merge_radius_bound` count is
positive.

The timed cells are the firing cells, highest B count first. Then
append `lj38`, `lj75`, `m80r14` and the TIP4P size with the highest
B count, skipping a name already taken. If every TIP4P count is
zero, the TIP4P cell is `tip4p8`. If the list is longer than 10,
delete from the end any cell that is not one of those four names
until 10 remain. `cells.txt` records the choice before the first
timed row.

## Seeds

The timed gate uses twelve seeds, 2000 through 2011, when that panel
can be run serially on one core. Otherwise it uses six seeds, 2000
through 2005. It never uses fewer than six. The seed count is fixed
in the probe commit, before any timed row.

Each seed has two ABBA blocks: `(A, B, B, A)` then `(A, A2, A2, A)`.
For seed offset k the eight labels are rotated left by k modulo
eight. Cells run in `cells.txt` order. Seeds run from 2000 upward.
No quick mode and no other timed work shares the session.

The harness and every driver are pinned to core 0 with `taskset -c
0`. Every `*_NUM_THREADS` variable is `1`. The child environment is
the harness allow-list.

## Bar

A hit is re-derived from `best` against the cell reference at
tolerance 0.001. A miss, a non-finite energy or an inconclusive row
is a miss. The sample unit for the hit tests is the seed. A seed
hits on an arm only when both replicates in its ABBA block hit. The
call score of a replicate is `calls_to_hit` on a hit and `budget + 1`
on a miss. The seed score is the worse of the two replicates.

A cell is `INCONCLUSIVE` when any of its rows is inconclusive or the
seed set is short. An inconclusive cell cannot pass. A cell is
`FAIL` when any clause below fails. Otherwise it is `PASS`. A fail
is never relabelled as a pass.

(a) Hit loss of B against A. One-sided Fisher exact on the seed hit
counts, and one-sided Mann-Whitney on the seed scores (B larger is
worse). Raw observation hit counts are reported as well. A drop
fails only when its p-value is below 0.05.

(b) Outside time per call, `outside_s / max(1, charged)`. The upper
edge of the B/A bootstrap 95% interval of the ratio of type-7
medians (10,000 paired seed resamples, fixed seed `20261121`) must
be at most 1.10. The same upper edge is also reported against
`max(1.10, U)`, where `U` is the upper edge of the A2/A interval.
The pass rule is the hard 1.10 bound. Each arm contributes the
midpoint of its two replicates.

(c) In-process `RssAnon`. Each arm contributes the maximum of its two
replicates. This clause fails only when the B/A lower edge is above
`max(1.10, U)`, where `U` is the A2/A upper edge. A point ratio above
1.10 does not fail when the lower edge is not.

Per seed, the report compares each B replicate with the first A
replicate on `evals`, `charged`, `calls_to_hit` and `best_bits`. A
difference does not by itself fail the cell. It is expected only
where the overflow arithmetic changes a skip. The report names the
cells and seeds that differ. The merge-radius path count is the sum
of `merge_radius_bound` on each arm.

Rows are append-only. A run log is written as `.tmp`, synced, renamed,
and read back before its sha256 is accepted.

## Cells

The probe is seed 2000 at budget 20000, arm A then arm B. Twenty-seven
cells return status ok. On each of those, A and B record the same
merge-radius count and the same best bits. `sc126-38` and `sc126-55`
exit before the search because the Sutton-Chen pair library is not
installed, so this probe does not show whether they reach the path.
They are not timed cells.

B counts, highest first among cells that ran: tip4p10 1773, tip4p21
1566, tip4p17 1292, tip4p16 1255, tip4p13 1207, tip4p12 1193,
tip4p11 1145, tip4p14 1139, tip4p15 1043, tip4p8 923, tip4p18 870,
tip4p20 853, tip4p19 653, m55r6 625, lj38 424, m80r6 264, lj98 165,
lj104 163, lj75 150, lj102 147, tip4p9 139, m55r14 131, blj105n55 45,
m80r14 43, blj115n38 35, blj115n55 24, blj105n38 24.

The timed list, after keeping lj38, lj75, m80r14 and tip4p10 and
capping at 10, is:

1. tip4p10
2. tip4p21
3. tip4p17
4. tip4p16
5. tip4p13
6. tip4p12
7. tip4p11
8. lj38
9. lj75
10. m80r14

The seed count is six, 2000 through 2005. Twelve seeds do not fit on
one core. At eight observations, the earlier panel's median elapsed
for m80r14, lj75, tip4p21, tip4p17 and tip4p16 is already more than
five hours at twelve seeds. Six is the minimum that still runs every
chosen cell.

## Results

This section is generated from `rows.csv` by `run_gate.py summarize`.

10 cells, 6 seeds, eight observations, 480 rows, 0 inconclusive. Gate verdict: **PASS**. 10 cells pass, 0 fail and 0 are inconclusive.

Trajectory differences are listed below. A difference does not by
itself fail a cell. A hit-count or call-score drop fails only when
its one-sided p-value is below 0.05.

- Hit loss, fails on: none.
- Outside-time upper bound above 1.10, fails on: none.
- RssAnon lower bound above its limit, fails on: none.

Hits seed are seeds where both B/A replicates hit. Hits obs are the
raw B/A observation counts. Outside is the median time outside energy
calls per charged call. The hard outside limit is 1.10. The noise
limit is max(1.10, A2/A upper). RssAnon fails only when its B/A lower
bound is above max(1.10, A2/A upper). Bound counts entries to the
merge-radius block test.

| Cell | Traj B, A2 | Hits seed A, B | Hits obs A, B | Fisher p | MW p | Outside B/A [95% CI] | Noise limit | Within noise | RssAnon B/A [95% CI] | RSS limit | Bound A, A2, B | Verdict |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: | --- | --- |
| tip4p10 | 12, 12 | 6, 6 | 12, 12 | 1 | 0.532246 | 0.987408 [0.945758, 1.06183] | 1.13723 | 1 | 1.0071 [1.00236, 1.01109] | 1.1 | 76464, 38232, 38232 | PASS |
| tip4p21 | 12, 12 | 0, 0 | 0, 0 | 1 | 1 | 1.01003 [0.984167, 1.02637] | 1.1 | 1 | 1.00476 [1.00392, 1.00557] | 1.1 | 56370016, 28185008, 28185008 | PASS |
| tip4p17 | 12, 12 | 2, 2 | 4, 4 | 0.727273 | 0.538138 | 0.994997 [0.941179, 1.01825] | 1.1 | 1 | 1.00304 [1.00246, 1.0053] | 1.1 | 43951636, 21975818, 21975818 | PASS |
| tip4p16 | 12, 12 | 1, 1 | 2, 2 | 0.772727 | 0.549185 | 0.999542 [0.972082, 1.01197] | 1.1 | 1 | 1.00256 [1.00107, 1.00456] | 1.1 | 44600108, 22300054, 22300054 | PASS |
| tip4p13 | 12, 12 | 6, 6 | 12, 12 | 1 | 0.532246 | 1.01847 [0.975767, 1.04453] | 1.1 | 1 | 1.0023 [0.997801, 1.0039] | 1.1 | 1642200, 821100, 821100 | PASS |
| tip4p12 | 12, 12 | 6, 6 | 12, 12 | 1 | 0.532246 | 0.997159 [0.946935, 1.0204] | 1.1 | 1 | 1.00223 [0.99696, 1.00844] | 1.1 | 410232, 205116, 205116 | PASS |
| tip4p11 | 12, 12 | 6, 6 | 12, 12 | 1 | 0.532246 | 0.982197 [0.977895, 1.00236] | 1.1 | 1 | 0.999078 [0.99361, 1.00072] | 1.1 | 2142144, 1071072, 1071072 | PASS |
| lj38 | 12, 12 | 6, 6 | 12, 12 | 1 | 0.532246 | 0.976908 [0.961586, 1.02313] | 1.1 | 1 | 1.00304 [0.999008, 1.00725] | 1.1 | 622424, 311212, 311212 | PASS |
| lj75 | 12, 12 | 2, 2 | 4, 4 | 0.727273 | 0.538138 | 0.9763 [0.972718, 0.994951] | 1.1 | 1 | 1.00744 [1.00729, 1.00992] | 1.1 | 126734392, 63367196, 63367196 | PASS |
| m80r14 | 12, 12 | 3, 3 | 6, 6 | 0.71645 | 0.534259 | 0.951445 [0.93241, 0.97866] | 1.1 | 1 | 1.01305 [1.00953, 1.01436] | 1.1 | 263138796, 131569398, 131569398 | PASS |

### Trajectory differences

None. On all 10 cells and 6 seeds, both B observations and A2 match A.

### Inconclusive rows

None. Every run exited 0 with timing and RssAnon recorded.

### Internal and external peak disagreements

None over 2 MiB.

### What changed

No timed cell and no timed seed differs. Both B replicates match the
first A replicate on evals, charged, calls to a hit and best bits, and
A2 matches A. The merge-radius count on B equals the count on A2. The
A total is twice that, because A has four observations and B has two.
The overflow arithmetic did not move a search outcome on this panel.
The bound path ran on every arm, and these descriptors sit far below
the magnitude where a squared gap overflows.
