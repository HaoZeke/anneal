# Gupta gold against GMIN

Two anneal revisions, same twelve seeds (2000–2011) and the same paper budgets. Serial, `sudo unshare --uts`, hostname `bench`, `taskset -c 0`. A hit is within 0.001 of a Cambridge energy. Wall medians are Hyndman–Fan type 7. Inside µs is `1e6 * pot_s / charged`, outside µs the same for `wall_s - pot_s`, each the median of the per-seed rates. `revision` is the tree the row was annealed on. GMIN did not change, so its rows stay on `c21f052c020f`.

`c21f052c020f` is the search before the physical sphere, the once-only common-neighbour trail, and the acceptance temperature for a steep quench. `f2a628c0157b` is the current develop. The binary that produced those rows is `e5bac3e2`, that develop plus one guard: a labelled core cut shorter than the cluster is not quenched. The two-shell Mackay body still stops at 55 sites (`n.min` of that site list) and is still offered. Without the guard, Au75 and Au98 pass a 165-coordinate geometry to a 225- or 294-coordinate potential. The guard does not change a cut that already has one site per atom.

GMIN is Wales `aac1548a8aec13b53ad12a8d6904625081685d2e`, Darby `BGUPTAT`, gold row A 0.2061, p 10.229, q 4.036, ξ 1.790, r0 2.884. Both arms of a head-to-head start from `h2h_metal coords`. Budgets are 400000 at 38 atoms and 4000000 at 75 and 98. The Cambridge Cu–Au archives publish pure gold at 38 (−130.54001708) and 55 only, so 75 and 98 have no stop.

| Revision | System | Arm | Hits | Median wall (s) | Median calls | Median best | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c21f052c020f | Au38 | anneal | 12/12 | 0.225 | 10494.5 | −130.54001708 | 14.99 | 4.93 |
| c21f052c020f | Au38 | gmin | 0/12 | 11.744 | 400000 | −129.85330626 | 27.14 | 2.21 |
| f2a628c0157b | Au38 | anneal | 12/12 | 0.256 | 12312.5 | −130.54001708 | 14.91 | 5.10 |
| c21f052c020f | Au75 | anneal | — | 332.153 | 4000000 | −263.27183577 | 57.93 | 25.30 |
| c21f052c020f | Au75 | gmin | — | 426.629 | 4000000 | −261.08629524 | 102.94 | 3.70 |
| f2a628c0157b | Au75 | anneal | — | 243.138 | 4000000 | −263.27183577 | 58.11 | 2.48 |
| c21f052c020f | Au98 | anneal | — | 468.555 | 4000000 | −345.98489186 | 98.02 | 19.05 |
| c21f052c020f | Au98 | gmin | — | 713.008 | 4000000 | −343.38619819 | 173.47 | 4.78 |
| f2a628c0157b | Au98 | anneal | — | 556.565 | 4000000 | −345.98489186 | 96.92 | 41.95 |

On `f2a628c0157b`, Au38 is still 12/12 against the stored GMIN 0/12. Fisher exact p = 7.40×10⁻⁷. Best-energy Mann–Whitney U = 0, tie-corrected and continuity-corrected, two-sided p = 1.03×10⁻⁵. The calls moved (median 10494.5 to 12312.5). The wall ratio, stored GMIN over this anneal, is 45.96.

At 75 and 98 every new anneal best is still below every GMIN best, so U is 0. p = 1.42×10⁻⁵ at 75 and 1.03×10⁻⁵ at 98. Eleven Au75 seeds reach −263.27183577; seed 2007 stops at −263.04663180. On `c21f052c020f` the higher energy was seeds 2003 and 2011. All twelve Au98 seeds stop at −345.98489186 on both revisions. Wall ratios are 1.755 and 1.281. Au75's outside time fell to 2.48 µs. Au98's outside time rose to 41.95 µs. Neither size has a Cambridge energy to stop on.

The `c21f052c020f` Au55 one-walk, 5/12, stays in `seeds.csv` under that revision. The k = 2, 4, 8 restarts and the budget-fraction restart were dropped. They described that older search. This develop changes the container, the neighbour shell, and the acceptance temperature, and the Au55 count on the sphere is recorded in `bench/au55-scale`, not here.

The wall-time move is outside the potential. Median inside time at 75 atoms is 231.7 s on `c21f052c020f` and 232.5 s on `f2a628c0157b`; outside falls from 101.2 s to 9.9 s, and the wall falls from 332 s to 243 s. At 98 atoms the inside time is 392.1 s then 387.7 s, and the outside time rises from 76.2 s to 167.8 s, which is the wall rise from 469 s to 557 s.

Cambridge publishes pure Gupta gold at 38 and 55 only. The Gupta pages are Zn, Cd, Pb and Na. Garzón's Au75 binding near 272 eV and Schebarchov's 2018 landscape use other parameter rows (Au38 there is not −130.540). Xue, Xu, Wang and Chen, with this row (A 0.2061, ξ 1.790, p 10.229, q 4.036), tabulate N = 100–200 and not 75 or 98; their N = 100 minimum is −353.126772, −3.531 eV per atom, next to this Au98 value of −3.530 eV per atom. No published minimum with these parameters is lower than −263.27183577 or −345.98489186.

The close-packed and polyicosahedral cuts, quenched on this potential, land at −263.04663180 for Au75 and −343.52066017 for Au98. Seed 2007 stops on that Au75 cut. The other eleven seeds reach −263.27183577, which those cuts do not. The shared value is a lower minimum than the cuts, not the cut itself.

`b81e62a6e8fe` is that same search with the two-shell Mackay cut empty when it cannot hold the cluster, instead of returned and then skipped. Au75 and Au98 at 4_000_000 calls, seeds 2000–2011, hostname bench:

| Revision | System | Hits | Median calls | Inside (s) | Outside (s) | Median best |
| --- | --- | --- | --- | --- | --- | --- |
| b81e62a6e8fe | Au75 | 0/12 | 4000000 | 239.913 | 10.679 | −263.27183577 |
| b81e62a6e8fe | Au98 | 0/12 | 4000000 | 394.499 | 176.434 | −345.98489186 |

There is no Cambridge energy at these sizes, so a hit is not scored. Eleven Au75 seeds are −263.27183577 and seed 2007 is −263.04663180. All twelve Au98 seeds are −345.98489186. Inside µs is 59.98 and 98.62; outside µs is 2.67 and 44.11.

The Mackay offer runs only inside the labelled-core block, which requires `cfg.species`; Lennard-Jones, Morse and Sutton-Chen use `Config::recommended` and leave species unset, so that block never runs for them.

`c21f052c020f1e2d7696fd89ce96e2b9ae8f2024` is `docs: the coarse Au55 taboo merge is doing useful work`. The Au75 outside time falls from 101.2 s on that commit to 9.9 s on `f2a628c0157b` in `8fe94689` (`fix: trace each common-neighbour trail once`), which expands each `(point, used bonds)` state once instead of pushing every continuation of the trail between quenches. That commit is already on this develop, the inside time stays about 232 s, and the Mackay cut does not touch it.
