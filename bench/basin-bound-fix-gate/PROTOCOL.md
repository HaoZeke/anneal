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

This section is replaced from `rows.csv` by `run_gate.py summarize`.
