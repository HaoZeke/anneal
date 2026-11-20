# TIP4P b9 gate

Rules fixed before any search in this gate.

## Arms

Four release builds, each in its own directory. No search flag is changed between them.

| Arm | Source |
| --- | --- |
| D | develop `61fa67cf380e46876b569da45e7cc091647a775b`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e` |
| Dp | a second release build of that same develop tree |
| B6 | tree `14b537fd9b73b8fe21e01721954b41a278c5d223` |
| B9 | tree `40238b525d795bf99fdf22a6ee66a4e238e4782e` |

## Cells

TIP4P clusters n = 8 through 21. The budget is the 2_000_000-call ceiling named `PAPER_BUDGETS` in `scripts/tip4p_h2h/run_campaign.py` (`{15: 2_000_000, 20: 2_000_000}`) and `scripts/tip4p_h2h/run_radius.py` (`{4: 2_000_000, 8: 2_000_000, 15: 2_000_000, 20: 2_000_000}`). Every size in that range uses that ceiling. References are the energies in `data/tip4p_ccd.csv`. The driver is `h2h_seed` on the recommended rigid preset. The early stop is the reference plus 0.001. `H2H_BUDGET` is not set.

Non-rigid cells are the default-on panel declared by `bench/no-motif-soap-gate/run_gate.py` (`notes_specs` plus `repo_specs`):

- LJ, `h2h_gmin_proof`: lj38, lj55, lj75, lj98, lj103, lj104
- Morse, `h2h_morse`: m55r6, m55r14, m80r6, m80r14
- Au, `h2h_metal`: au55
- Sutton-Chen, `h2h_sc`: sc126-38, sc126-55, sc126-80, sc108-38, sc108-55, sc108-80, sc96-38, sc96-55, sc96-80
- binary LJ, `h2h_metal`, from `scripts/metal_h2h/run_darby.py`: au38, cu6au32, cu19au19, au55, cu27au28

`au55` is declared twice with the same driver, budget and reference. It is one cell. The gate config samples Sutton-Chen at six seeds. This gate uses seeds 2000-2011 on every cell, including Sutton-Chen.

A declared cell is scored only when every arm's `budget` print equals the gate config's expected budget and the printed reference is finite and the same on every arm. A mismatch is not scored. That is the gate's own unsupported rule. m75 and surface-slab are not in the command list and are not cells.

## Seeds and order

Seeds 2000-2011. One serial process, `taskset -c 0`, and every thread-count variable at 1. No warm-up run and no quick flag.

Canonical order is D, B6, B9, Dp. For offset `k = seed - 2000`, rotate that order by `k mod 4`. Offsets 4 through 7 run the rotated order in reverse. Offsets 0-3 and 8-11 keep the rotated order. Each arm occupies each of the four slots on 3 of the 12 seeds.

## Hit, calls, time

A hit is a finite best energy at or below the reference plus 0.001. The stored best token is the driver's token. Calls stored on a row are the driver's calls to the hit, or the budget when the row misses.

Calls compared by the Mann-Whitney test rank a hit at its call count and a miss at `budget + 1`, after every hit. The test is two-sided, normal approximation, with the tie correction and the continuity correction. A calls loss is p < 0.05 and the B9 sample ranked worse (higher) than D.

Outside time per call is `outside_s / charged` on each row. The cell ratio is the Hyndman-Fan type 7 median on B9 divided by the type 7 median on D. The A/A factor is the same ratio of Dp to D. Fisher's exact test on hits is two-sided.

## Pass rule

PASS requires all three of the following on every scored cell.

1. B9 has no fewer hits than D, and no fewer hits than B6.
2. B9 has no calls loss against D at Mann-Whitney p < 0.05, with misses ranked worst.
3. The B9/D outside-time-per-call ratio of medians is at most 1.10, or else it is no larger than the Dp/D ratio.

FIX if every scored row finished and any of those three fails. FAIL if a scored row is missing, an arm tree is wrong, or LJ, Morse, Sutton-Chen or Au has no scored cell.

Two further reports do not change that verdict. Non-rigid B9 rows are compared with D on the best-energy token, the calls and the charged count. A B8 build is added only when that comparison differs. TIP4P19 hit counts for B6 and B9 come from this gate's rows. Counts for b7 and b8 come from whatever committed rows already exist, and each count names its file.
