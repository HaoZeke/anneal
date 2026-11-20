# G1 Component Knockout Ablation

Single-knockout ablation of the `Config::recommended` preset on the nine-system atomic cluster panel. Each knockout arm disables one component group of `recommended` while holding every other field at its `recommended` value.

## Base Revision

- Commit: `61fa67cf380e46876b569da45e7cc091647a775b` (`develop`)
- Tree: `cff796c507a8fa1d0bfb28888c3570b61988055e`

## Knob Mapping

Full notes are in [`knob-map.md`](knob-map.md).

| Arm | Named knob | Real `Config` field | `recommended` value | Knockout value | Source file and line |
| --- | --- | --- | --- | --- | --- |
| `recommended` | Full preset (control) | *(none changed)* | `Config::recommended(n)` | *(control)* | `src/methods/cluster_hopping/config.rs:1786,1827` |
| `a_restart_on_stall` | restart on stall | `restart_on_stall` | `false` | `true` | `src/methods/cluster_hopping/config.rs:1003,2205` |
| `a_restart_on_stall` | climb off | `escape_on_stall` | `true` | `false` | `src/methods/cluster_hopping/config.rs:574,1842` |
| `a_restart_on_stall` | trail off | `trail_on_stall` | `true` | `false` | `src/methods/cluster_hopping/config.rs:596,1841` |
| `a_restart_on_stall` | tabu off | `tabu_on_stall` | `true` | `false` | `src/methods/cluster_hopping/config.rs:986,1832` |
| `b_no_orbit` | `orbit_complete_on_new` off | `orbit_complete_on_new` | `true` | `false` | `src/methods/cluster_hopping/config.rs:672,1837` |
| `c_no_symmetry` | continuous symmetry off | `continuous_symmetry` | `ContinuousSymmetry::Measure { interval: 10 }` | `ContinuousSymmetry::Off` | `src/methods/cluster_hopping/config.rs:738,1864` |
| `c_no_symmetry` | `symmetrise_on_stall` off | `symmetrise_on_stall` | `true` | `false` | `src/methods/cluster_hopping/config.rs:650,1834` |
| `d_no_screens` | return screen off | `return_screen` | `true` | `false` | `src/methods/cluster_hopping/config.rs:1063,1833` |
| `d_no_screens` | Bayes screen off | `bayes_screen` | `true` | `false` | `src/methods/cluster_hopping/config.rs:794,2052` |
| `e_no_surfaces` | no diameter-penalty surface | `surfaces` | `[TwoPhase::relative(0.7, 1.0)]` for `n < 98`; `[]` for `n >= 98` | `[]` | `src/methods/cluster_hopping/config.rs:1406,1849-1853` |
| `f_no_energy_lattice` | `energy_lattice` off | `energy_lattice` | `true` | `false` | `src/methods/cluster_hopping/config.rs:1402,1868` |
| `g_no_priors` | `prior_soap` off | `prior_soap` | `true` | `false` | `src/methods/cluster_hopping/config.rs:1105,2236` |
| `g_no_priors` | `prior_reseed` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:229` |
| `g_no_priors` | `prior_growth` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:230` |
| `g_no_priors` | `prior_core` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:231` |
| `g_no_priors` | `prior_motif` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:233` |
| `h_no_basin_bias` | basin bias off | `bias_height` | `0.25` | `0.0` | `src/methods/cluster_hopping/config.rs:409,2123`; `src/methods/cluster_hopping/preset.rs:9` |

## Systems, Budgets, and References

All budgets follow `PAPER_BUDGETS` (`400000` for 38 atoms, `1000000` for 55 atoms, `4000000` for 75 atoms and above) and the Cambridge Cluster Database / Gupta reference energies from `scripts/gmin_panel/run_panel.py`.

| System | Potential | Atoms | Budget | Reference energy |
| --- | --- | ---: | ---: | ---: |
| `lj38` | Lennard-Jones | 38 | 400000 | -173.928427 |
| `lj55` | Lennard-Jones | 55 | 1000000 | -279.248470 |
| `lj75` | Lennard-Jones | 75 | 4000000 | -397.492331 |
| `lj98` | Lennard-Jones | 98 | 4000000 | -543.665361 |
| `lj150` | Lennard-Jones | 150 | 4000000 | -893.310258 |
| `m55r14` | Morse (`rho=14`) | 55 | 1000000 | -220.646208 |
| `m80r6` | Morse (`rho=6`) | 80 | 4000000 | -378.333471 |
| `m80r14` | Morse (`rho=14`) | 80 | 4000000 | -340.811371 |
| `au55` | Gupta Au | 55 | 1000000 | -191.03085280 |

## Protocol

- **Seeds**: `2000` through `2011` (`n = 12` per `(system, arm)`, `972` runs total).
- **Execution**: One serial session, release build (`examples/g1_knockout.rs` calling `cluster_search::search_from` with `rgpot` kernels installed), pinned to core 0 (`taskset -c 0`), all thread environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `TOKIO_WORKER_THREADS`) set to `1`, all `H2H_*` environment variables unset, no warm-up, no `--quick`.
- **Arm rotation**: Within each `(system, seed)`, the 9 arms run in a cyclically shifted order `shift = (system_index + (seed - 2000) // 2) % 9`, reversed on odd seeds (ABBA pairing). Every arm has mean slot `4.0` within every system and uniform `(arm, slot)` counts across the panel.
- **Hit criterion**: `energy_hits(best, reference, DEFAULT_HIT_TOL)` (`best` finite and `best <= reference + 0.001`). Errors or timeouts stay in `rows.csv` as misses.
- **Statistics**: Computed from raw `rows.csv` only, each knockout arm against `recommended` per system (`72` pairwise tests per statistic family; raw two-sided p-values reported with no multiple-comparison adjustment):
  - Two-sided Fisher's exact test on hits.
  - Kaplan-Meier survival curves of calls-to-hit with misses right-censored at `budget` (`survival.csv`) and two-sided log-rank test.
  - Two-sided tie-corrected Mann-Whitney U test on calls with misses ranked `+inf`.
  - Hyndman-Fan type-7 median calls and median `outside_s / charged` per arm.

## Artifacts

- `knob-map.md`: Detailed arm-to-field mapping and source citations.
- `rows.csv`: Raw per-run records (972 rows).
- `sessions.jsonl`: Session provenance with SHA-256 hashed machine identifier.
- `summary.csv`: Per `(system, arm)` summary statistics and knockout-vs-recommended p-values.
- `survival.csv`: Per `(system, arm)` Kaplan-Meier survival curves.
