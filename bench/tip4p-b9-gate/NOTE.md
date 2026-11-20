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

## Result

Verdict: FIX.

Every scored row finished: 1776 rows, 37 cells, seeds 2000-2011, four arms. Session `20261120T094540Z`. Machine id `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`. Core 0 and affinity 0 on every row. `cu19au19` is not scored. The budget print is the driver default, not an environment override. See below. LJ, Morse, Sutton-Chen and Au each have at least one scored cell. The arm trees match the table above. The rescored hit agrees with the driver hit flag on every row.

1. Fails. B9 has fewer hits than D on tip4p15 (5 against 7), tip4p16 (1 against 2), tip4p17 (3 against 5) and tip4p20 (4 against 6). B9 has as many hits as B6 on every cell.
2. Holds. No cell is a calls loss. The smallest Mann-Whitney p of B9 against D is 0.558356 on tip4p17.
3. Fails. tip4p15 through tip4p21 have B9/D outside-time-per-call ratios 1.22221, 1.20482, 1.22867, 1.40786, 1.23591, 1.19516 and 1.26156. Each is above 1.10 and above that cell's Dp/D ratio. On those seven cells the B6/D ratio sits beside the B9/D ratio.

B9 equals B6 on hits on every cell, and the B9/D and B6/D time ratios match, so the TIP4P 15-21 regression already comes in with b6.

Hits below are counts out of 12. Calls are type 7 medians of the ranked counts, and a miss ranks at budget + 1. B9/D, B6/D and Dp/D are ratios of type 7 medians of `outside_s / charged`. The last four columns are 1 when that part of the pass rule fails.

| Cell | D | Dp | B6 | B9 | Fisher B9/D | Fisher B9/B6 | Calls D | Calls Dp | Calls B6 | Calls B9 | MWU p | B9/D | B6/D | Dp/D | Hit vs D | Hit vs B6 | Calls | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tip4p8 | 12 | 12 | 12 | 12 | 1 | 1 | 6339 | 6339 | 6339 | 6339 | 1 | 1.01876 | 1.02431 | 0.989441 | 0 | 0 | 0 | 0 |
| tip4p9 | 12 | 12 | 12 | 12 | 1 | 1 | 5839.5 | 5839.5 | 5839.5 | 5839.5 | 1 | 1.00552 | 1.00178 | 0.990543 | 0 | 0 | 0 | 0 |
| tip4p10 | 12 | 12 | 12 | 12 | 1 | 1 | 13138.5 | 13138.5 | 13138.5 | 13138.5 | 1 | 0.943558 | 0.979143 | 0.949377 | 0 | 0 | 0 | 0 |
| tip4p11 | 12 | 12 | 12 | 12 | 1 | 1 | 109288 | 109288 | 109288 | 109288 | 0.862164 | 1.03424 | 1.02557 | 0.98208 | 0 | 0 | 0 | 0 |
| tip4p12 | 12 | 12 | 12 | 12 | 1 | 1 | 41418.5 | 41418.5 | 41418.5 | 41418.5 | 1 | 1.01907 | 0.991808 | 1.03787 | 0 | 0 | 0 | 0 |
| tip4p13 | 12 | 12 | 12 | 12 | 1 | 1 | 216689.5 | 216689.5 | 256396.5 | 256396.5 | 0.839654 | 1.0685 | 1.02959 | 1.04592 | 0 | 0 | 0 | 0 |
| tip4p14 | 12 | 12 | 12 | 12 | 1 | 1 | 135853 | 135853 | 135853 | 135853 | 1 | 0.993537 | 1.00525 | 0.989019 | 0 | 0 | 0 | 0 |
| tip4p15 | 7 | 7 | 5 | 5 | 0.684273 | 1 | 1481619.5 | 1481619.5 | 2000001 | 2000001 | 0.599885 | 1.22221 | 1.22234 | 1.04402 | 1 | 0 | 0 | 1 |
| tip4p16 | 2 | 2 | 1 | 1 | 1 | 1 | 2000001 | 2000001 | 2000001 | 2000001 | 0.615303 | 1.20482 | 1.19454 | 0.973332 | 1 | 0 | 0 | 1 |
| tip4p17 | 5 | 5 | 3 | 3 | 0.666846 | 1 | 2000001 | 2000001 | 2000001 | 2000001 | 0.558356 | 1.22867 | 1.2318 | 1.00897 | 1 | 0 | 0 | 1 |
| tip4p18 | 2 | 2 | 3 | 3 | 1 | 1 | 2000001 | 2000001 | 2000001 | 2000001 | 0.597201 | 1.40786 | 1.39391 | 1.0094 | 0 | 0 | 0 | 1 |
| tip4p19 | 7 | 7 | 7 | 7 | 1 | 1 | 1060845 | 1060845 | 1081780.5 | 1081780.5 | 1 | 1.23591 | 1.23906 | 0.989404 | 0 | 0 | 0 | 1 |
| tip4p20 | 6 | 6 | 4 | 4 | 0.680172 | 1 | 1686250 | 1686250 | 2000001 | 2000001 | 0.561488 | 1.19516 | 1.19739 | 0.995358 | 1 | 0 | 0 | 1 |
| tip4p21 | 0 | 0 | 0 | 0 | 1 | 1 | 2000001 | 2000001 | 2000001 | 2000001 | 1 | 1.26156 | 1.25829 | 1.02309 | 0 | 0 | 0 | 1 |
| lj38 | 12 | 12 | 12 | 12 | 1 | 1 | 127839.5 | 127839.5 | 127839.5 | 127839.5 | 1 | 0.998996 | 1.00049 | 0.993003 | 0 | 0 | 0 | 0 |
| lj55 | 12 | 12 | 12 | 12 | 1 | 1 | 5722 | 5722 | 5722 | 5722 | 1 | 1.00868 | 1.00327 | 1.02347 | 0 | 0 | 0 | 0 |
| lj75 | 5 | 5 | 5 | 5 | 1 | 1 | 4000001 | 4000001 | 4000001 | 4000001 | 1 | 1.08183 | 1.02085 | 1.04142 | 0 | 0 | 0 | 0 |
| lj98 | 7 | 7 | 7 | 7 | 1 | 1 | 3166811.5 | 3166811.5 | 3166811.5 | 3166811.5 | 1 | 1.00939 | 1.01824 | 1.0227 | 0 | 0 | 0 | 0 |
| lj103 | 6 | 6 | 6 | 6 | 1 | 1 | 3905073 | 3905073 | 3905073 | 3905073 | 1 | 0.980287 | 0.996641 | 0.959316 | 0 | 0 | 0 | 0 |
| lj104 | 5 | 5 | 5 | 5 | 1 | 1 | 4000001 | 4000001 | 4000001 | 4000001 | 1 | 1.03248 | 1.0145 | 1.02107 | 0 | 0 | 0 | 0 |
| m55r6 | 12 | 12 | 12 | 12 | 1 | 1 | 22596 | 22596 | 22596 | 22596 | 1 | 0.98716 | 1.01494 | 1.0039 | 0 | 0 | 0 | 0 |
| m55r14 | 9 | 9 | 9 | 9 | 1 | 1 | 451561.5 | 451561.5 | 451561.5 | 451561.5 | 1 | 0.993196 | 0.999675 | 1.0243 | 0 | 0 | 0 | 0 |
| m80r6 | 12 | 12 | 12 | 12 | 1 | 1 | 62677 | 62677 | 62677 | 62677 | 1 | 1.03234 | 1.00686 | 1.00528 | 0 | 0 | 0 | 0 |
| m80r14 | 6 | 6 | 6 | 6 | 1 | 1 | 3901429.5 | 3901429.5 | 3901429.5 | 3901429.5 | 1 | 0.97122 | 0.972149 | 0.957007 | 0 | 0 | 0 | 0 |
| au55 | 4 | 4 | 4 | 4 | 1 | 1 | 1000001 | 1000001 | 1000001 | 1000001 | 1 | 0.994018 | 0.992848 | 1.00152 | 0 | 0 | 0 | 0 |
| sc126-38 | 12 | 12 | 12 | 12 | 1 | 1 | 113504.5 | 113504.5 | 113504.5 | 113504.5 | 1 | 1.00537 | 0.978801 | 0.982586 | 0 | 0 | 0 | 0 |
| sc126-55 | 12 | 12 | 12 | 12 | 1 | 1 | 18379.5 | 18379.5 | 18379.5 | 18379.5 | 1 | 1.00962 | 0.997709 | 1.00951 | 0 | 0 | 0 | 0 |
| sc126-80 | 5 | 5 | 5 | 5 | 1 | 1 | 1000001 | 1000001 | 1000001 | 1000001 | 1 | 1.01854 | 1.00161 | 0.99215 | 0 | 0 | 0 | 0 |
| sc108-38 | 7 | 7 | 7 | 7 | 1 | 1 | 361031.5 | 361031.5 | 361031.5 | 361031.5 | 1 | 0.964148 | 0.989707 | 0.971974 | 0 | 0 | 0 | 0 |
| sc108-55 | 5 | 5 | 5 | 5 | 1 | 1 | 1000001 | 1000001 | 1000001 | 1000001 | 1 | 1.0007 | 1.0003 | 0.999192 | 0 | 0 | 0 | 0 |
| sc108-80 | 3 | 3 | 3 | 3 | 1 | 1 | 1000001 | 1000001 | 1000001 | 1000001 | 1 | 1.02556 | 1.01218 | 0.997405 | 0 | 0 | 0 | 0 |
| sc96-38 | 12 | 12 | 12 | 12 | 1 | 1 | 50230 | 50230 | 50230 | 50230 | 1 | 0.987972 | 0.988954 | 1.00061 | 0 | 0 | 0 | 0 |
| sc96-55 | 12 | 12 | 12 | 12 | 1 | 1 | 23157 | 23157 | 23157 | 23157 | 1 | 0.995115 | 0.995257 | 1.0013 | 0 | 0 | 0 | 0 |
| sc96-80 | 7 | 7 | 7 | 7 | 1 | 1 | 989847.5 | 989847.5 | 989847.5 | 989847.5 | 1 | 0.990724 | 0.977017 | 0.977017 | 0 | 0 | 0 | 0 |
| au38 | 8 | 8 | 8 | 8 | 1 | 1 | 186585 | 186585 | 186585 | 186585 | 1 | 0.985337 | 0.979117 | 1.01203 | 0 | 0 | 0 | 0 |
| cu6au32 | 0 | 0 | 0 | 0 | 1 | 1 | 400001 | 400001 | 400001 | 400001 | 1 | 0.99457 | 0.985969 | 1.00689 | 0 | 0 | 0 | 0 |
| cu27au28 | 0 | 0 | 0 | 0 | 1 | 1 | 1000001 | 1000001 | 1000001 | 1000001 | 1 | 0.984189 | 0.983648 | 0.991301 | 0 | 0 | 0 | 0 |


### cu19au19

`PAPER_BUDGETS` in `scripts/metal_h2h/run_darby.py` at `61fa67cf380e46876b569da45e7cc091647a775b` sets `cu19au19` to 400000. `examples/h2h_metal.rs` names 80000 in `system_spec`. The `budget` command prints that field and the reference. It does not read `H2H_BUDGET`. `H2H_BUDGET` is read only by the `run` command, and no `H2H_*` variable was set. The cell was not run. Each arm printed:

```
80000
-122.9914839100
```

### Non-rigid digests

On all 23 non-rigid cells the B9 rows match the D rows on the best-energy token, the calls and the charged count. A B8 build was not added.

### TIP4P19

This gate's `rows.csv`:

- D and Dp: 7/12, seeds 2000, 2001, 2005, 2006, 2007, 2009, 2011
- B6: 7/12, seeds 2000, 2005, 2006, 2007, 2009, 2010, 2011
- B9: 7/12, seeds 2000, 2005, 2006, 2007, 2009, 2010, 2011

b7 is `bench/tip4p-b7-gate/rows.csv` on `origin/record/tip4p-b7-gate` at `4b8c5cbfe52778bfd2ed35dd33077f27ea61a895`. That commit is not an ancestor of develop. Arm B, tree `c75edc53ed1ec9b91b1bd76372661ddec9e33a90`: 2/12, seeds 2001 and 2007. Arm A on that file, tree `14b537fd9b73b8fe21e01721954b41a278c5d223`: 7/12, seeds 2000, 2005, 2006, 2007, 2009, 2010, 2011.

The b8 tip4p19 count of 3/12 is unverified. No fetchable ref contains `bench/tip4p-b8-vs-b6-gate/rows.csv`.
