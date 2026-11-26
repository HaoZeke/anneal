# Communicating vs Independent vs Single-Chain LJ Report

Rows recorded: 36 (seeds 2000..2011, `SEARCH_HIT_TOL = 1.0e-04`).

## LJ38 (`n = 38`, `budget = 400000`, `reference = -173.928427`)

| arm | n | hits | hit rate (95% CI) | median calls_to_hit (95% CI) | median wall_s (95% CI) | median cpu_s (95% CI) | median peak RssAnon KiB (95% CI) | mean ledger_sum | corpus appended |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `comm` | 12 | 9/12 | 0.750 [0.500, 1.000] | 300690 [91136, 394455] | 35.47 [33.70, 36.07] | 35.47 [33.70, 36.07] | 48324 [48186, 48670] | 400000 | 96 |
| `indep` | 12 | 9/12 | 0.750 [0.500, 1.000] | 183596 [93826, 331598] | 32.28 [31.55, 32.95] | 32.27 [31.54, 32.95] | 48692 [48126, 49068] | 400000 | 96 |
| `single` | 12 | 11/12 | 0.917 [0.750, 1.000] | 76085 [53038, 246040] | 23.30 [22.58, 24.52] | 23.30 [22.57, 24.52] | 34902 [34560, 35528] | 400000 | 24 |

### LJ38 Pairwise Comparisons (10,000-resample bootstrap, seed 20261123)

| comparison | hit rate diff (95% CI) | Fisher exact p | calls_to_hit median diff (95% CI) | MWU U | MWU p | wall_s median diff (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| `comm` vs `indep` | +0.000 [-0.333, +0.333] | 1 | +117094 [-144119, +275392] | 64.5 | 0.6837 | +3.20 [+1.33, +4.12] |
| `comm` vs `single` | -0.167 [-0.417, +0.083] | 0.5901 | +224605 [-37692, +324206] | 45.5 | 0.1325 | +12.17 [+10.13, +13.14] |
| `indep` vs `single` | -0.167 [-0.417, +0.083] | 0.5901 | +107512 [-87442, +255513] | 52.5 | 0.2716 | +8.97 [+7.55, +10.04] |

### LJ38 Ledger and Mechanism Means

| arm | hops | screened_out | returned | basins | history_minima | history_obs | history_new | history_s | orbits | continuous_sym |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `comm` | 5417.3 | 0.0 | 3732.6 | 261.2 | 726.9 | 1590.3 | 894.6 | 126.61 | 301.5 | 511.8 |
| `indep` | 5427.8 | 0.0 | 3733.9 | 259.1 | 740.4 | 1599.6 | 923.3 | 112.35 | 309.3 | 510.1 |
| `single` | 7886.8 | 647.7 | 5467.4 | 237.0 | 470.2 | 1001.3 | 570.8 | 19.25 | 437.9 | 744.9 |

## Remaining Systems Stopped Under 4-Hour Time Guard

Execution stopped before starting systems whose projected full-budget wall time exceeded the 4-hour (14,400 s) campaign cap (bottlenecked by `universal_descriptor_space` in `MinimumHistory::observe` at ~0.10 s/obs for `lj75` and ~0.19 s/obs for `lj98` across 36 runs per system).

| system | full budget | cal budget | comm cal wall_s | indep cal wall_s | single cal wall_s | projected per-seed (s) | projected 12-seed total (s / h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lj75` | 4000000 | 100000 | 21.80 | 21.49 | 24.10 | 2695.5 | 32346 s (8.98 h) |
| `lj98` | 4000000 | 100000 | 28.55 | 28.36 | 42.98 | 3995.5 | 47946 s (13.32 h) |
