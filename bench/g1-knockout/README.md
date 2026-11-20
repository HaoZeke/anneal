# G1 Component Knockout Ablation

Single-knockout ablation of the `Config::recommended` preset on the nine-system atomic cluster panel. Each knockout arm disables one component group of `recommended` while holding every other field at its `recommended` value.

## Base Revision

- Commit: `61fa67cf380e46876b569da45e7cc091647a775b` (`develop`)
- Tree: `cff796c507a8fa1d0bfb28888c3570b61988055e`

## Stock Preset Confirmation (`develop` `61fa67cf`)

Every arm (`recommended` and all eight knockouts) ran on the stock `Config::recommended(n)` preset of `develop` `61fa67cf380e46876b569da45e7cc091647a775b` (the exact defaults a caller receives, with zero `H2H_*` environment overrides). **No arm used `c2de652d` settings.**

- **Structure priors in stock `Config::recommended(n)` at `61fa67cf`**: `prior_soap` is **`true`** (with `soap_mode = "flexible"`), so the SOAP proposal is the only structure prior active in the stock preset. `prior_reseed`, `prior_growth`, `prior_core`, and `prior_motif` are **absent (not on)** at `61fa67cf` — they were removed as inert since `1c1b1238` and appear only in `REMOVED_PRIOR_KEYS` (`src/methods/cluster_hopping/config.rs:228-235`).

### Stock Preset Hashes (`Config::resolved_sha256()`)

| Atoms (systems) | `container` | `surfaces` | Stock preset SHA-256 |
| --- | ---: | --- | --- |
| 38 (`lj38`) | `3.025777866119067` | `[TwoPhase::relative(0.7, 1.0)]` | `32e59764f29aea5583d64df9975c1c1418616d61298f5846c50e5f56666712d6` |
| 55 (`lj55`, `m55r14`, `au55`) | `3.4226572146852523` | `[TwoPhase::relative(0.7, 1.0)]` | `a4b937a5d71aad58ed2ee12b39cf3d37228f97c8afc51a7dc5500d6f6517daed` |
| 75 (`lj75`) | `3.795446993857872` | `[TwoPhase::relative(0.7, 1.0)]` | `cfdcebadd6f3dfbfeacd300098eb644767c7c3080fafe02859446d79075ec660` |
| 80 (`m80r6`, `m80r14`) | `3.877982442057391` | `[TwoPhase::relative(0.7, 1.0)]` | `c07ca603ec1ddd3d73fc94b2457fd0bf096425f6fdd2d4cf729cdf25174841fd` |
| 98 (`lj98`) | `4.149392662852602` | `[]` | `79a5533b2e3c1810d1feb5da317505874bb429815b247af793e9c6652788a962` |
| 150 (`lj150`) | `4.78196356132175` | `[]` | `e9e26f25a7b799624564c90293188a7095e16c43a449d5c8f83c1906ad9160c1` |

### Complete Stock Preset Knobs (`Config::recommended(n)`, all 171 fields)

Serialized fields in `Config::resolved_json()` (`schema = "anneal-cluster-config-v1"`, 139 config fields; `n_points`, `container`, and `surfaces` vary with `n` as tabulated above):

| Field | Stock `recommended` value |
| --- | --- |
| `n_points` | `n` (38, 55, 75, 80, 98, or 150) |
| `length_scale` | `1.0` |
| `energy_scale` | `1.0` |
| `move_library` | `"lean_burst"` |
| `rigid_body_geometry` | `null` |
| `neighbour_cutoff` | `1.6` |
| `symmetrise_cutoff` | `2.5` |
| `temperature` | `0.8` |
| `bias_height` | `0.25` |
| `bias_gamma` | `5.0` |
| `merge_radius` | `0.7` |
| `keying_sigma` | `2.5` |
| `morphology_scale` | `1.0` |
| `theta` | `0.5` |
| `budget_window` | `false` |
| `allocate_moves` | `true` |
| `adaptive_height` | `false` |
| `max_hops` | `null` |
| `replicas` | `1` |
| `swap_period` | `50` |
| `minima_hopping` | `false` |
| `md_escape` | `false` |
| `md_escape_dt` | `0.005` |
| `md_escape_kinetic` | `1.0` |
| `md_escape_minima` | `2` |
| `md_escape_max_steps` | `2000` |
| `md_escape_soften` | `0` |
| `escape_lanczos_steps` | `16` |
| `escape_epsilon` | `0.0001` |
| `escape_amplitude` | `0.25` |
| `escape_overshoot` | `1.5` |
| `escape_max_climb` | `24` |
| `escape_on_stall` | `true` |
| `trail_on_stall` | `true` |
| `escape_stall_patience` | `5000` |
| `escape_stall_factor` | `2.0` |
| `track_funnels` | `false` |
| `funnel_period` | `20000` |
| `symmetrise_on_stall` | `true` |
| `point_symmetrise` | `false` |
| `point_symmetrise_on_new` | `false` |
| `orbit_complete_on_new` | `true` |
| `soap_repel` | `false` |
| `exchange_metropolis` | `false` |
| `shared_deposits` | `8` |
| `shared_visit_policy` | `"tabu"` |
| `jump_on_stall` | `false` |
| `jump_patience` | `5000` |
| `jump_steps` | `10` |
| `jump_step` | `0.38` |
| `symmetrise_core_fraction` | `0.6` |
| `point_symmetrise_every_accept` | `false` |
| `continuous_symmetry` | `{"group": "measure", "interval": 10}` |
| `symmetry_tolerance` | `0.5` |
| `symmetry_merge_radius` | `0.7` |
| `symmetrise_patience` | `2000` |
| `angular_moves` | `false` |
| `angular_target` | `0.5` |
| `keying` | `"distances"` |
| `contextual_moves` | `false` |
| `contextual_floor` | `0.1` |
| `bayes_screen` | `true` |
| `flat_histogram` | `false` |
| `flat_sweep` | `400` |
| `flat_quantile` | `0.5` |
| `statistical_temperature` | `false` |
| `energy_bias` | `false` |
| `depth_reward` | `true` |
| `novel_reward` | `false` |
| `soft_perturb` | `false` |
| `soft_modes` | `6` |
| `soft_steps` | `30` |
| `cov_perturb` | `false` |
| `staged_quench` | `false` |
| `settle_iters` | `20` |
| `group_cutoff` | `3.4` |
| `covalent_cutoff` | `1.3` |
| `species` | `null` |
| `bond_tolerance` | `1.25` |
| `frozen` | `null` |
| `active_region` | `null` |
| `bayes_exploration` | `0.1` |
| `bayes_threshold` | `0.4666666666666667` |
| `bayes_warmup` | `300` |
| `tabu_on_stall` | `true` |
| `tabu_capacity` | `8` |
| `restart_on_stall` | `false` |
| `restart_patience` | `5000` |
| `calibrate_radius` | `false` |
| `calibrate_quantile` | `0.9` |
| `calibrate_warmup` | `200` |
| `bias_by_rung` | `false` |
| `ladder_top` | `4.0` |
| `ladder_mode` | `"Shipped"` |
| `ladder_window` | `10` |
| `ladder_pilot` | `200` |
| `ladder_target_accept` | `0.23` |
| `return_screen` | `true` |
| `soap_class_residual` | `false` |
| `soap_mode` | `"flexible"` |
| `return_polish` | `0` |
| `return_polish_after` | `0` |
| `path_on_stall` | `false` |
| `stall_patience` | `60` |
| `path_images` | `9` |
| `anneal_diversity` | `false` |
| `diversity_floor` | `0.75` |
| `height_revisits` | `4.0` |
| `shape_keyed` | `false` |
| `screen_margin` | `2.0` |
| `calibrated_screen` | `true` |
| `finish_improving` | `true` |
| `screen_steps` | `25` |
| `adaptive_screen` | `false` |
| `record_gradient` | `0.001` |
| `trace_quenched` | `false` |
| `polish_records` | `0` |
| `surrogate_tolerance` | `0.5` |
| `delayed_acceptance` | `false` |
| `construct_width` | `4` |
| `probe_screen` | `false` |
| `quench_warmup` | `4` |
| `quench_confidence` | `2.0` |
| `quench_once` | `false` |
| `incumbent_gap` | `0.0` |
| `relax_steps` | `1000` |
| `two_phase` | `null` |
| `reoccupy_interval` | `5000` |
| `energy_lattice` | `true` |
| `surfaces` | `[{"cutoff": {"Relative": 0.7}, "beta": 1.0, "mu": 0.0, "anisotropic": false}]` for `n < 98`; `[]` for `n >= 98` |
| `container` | `0.9 * n^(1/3)` (see hash table above) |
| `min_separation` | `0.85` |
| `displacement_only` | `false` |
| `superbasin_report` | `false` |
| `superbasin_escape` | `false` |
| `superbasin_period` | `2000` |
| `energy_trace` | `false` |
| `superbasin_quotient` | `false` |
| `superbasin_features` | `false` |

Default-omitted (`#[serde(skip_serializing_if)]`) and non-serialized (`#[serde(skip)]`) fields on `Config` (32 fields):

| Field | Stock `recommended` value | Serialization note |
| --- | --- | --- |
| `basin_stride` | `1` | skipped when 1 |
| `retry_noop_stall_leave` | `false` | skipped when false |
| `align_directed_stall_source` | `false` | skipped when false |
| `orbit_arm` | `false` | skipped when false |
| `orbit_core` | `OrbitCore::Fraction` | skipped when Fraction |
| `surface_hop_on_best` | `false` | skipped when false |
| `arm_selection` | `ArmSelection::Posterior` | skipped when Posterior |
| `species_sigma` | `None` | skipped when None |
| `prior_soap` | `true` | skipped when true; only active structure prior in stock preset |
| `general_collective` | `false` | skipped when false |
| `homotop_descent` | `true` | skipped when true (no-op when species is None) |
| `homotop_ranked` | `true` | skipped when true (no-op when species is None) |
| `homotop_window` | `true` | skipped when true (no-op when species is None) |
| `homotop_reorder` | `false` | skipped when false |
| `precondition_bonds` | `false` | skipped when false |
| `capped_lindh_quench` | `false` | skipped when false |
| `local_step` | `LocalStep::Lbfgs` | skipped when Lbfgs |
| `answer_gradient` | `0.0` | skipped when <= 0.0 |
| `long_ranged` | `false` | skipped when false |
| `plateau_heat` | `false` | skipped when false |
| `target_energy` | `None` | skipped when None |
| `improvement_rewards` | `false` | skipped when false |
| `reoccupy` | `None` | non-serialized (#[serde(skip)]) |
| `pair_range` | `None` | skipped when None |
| `call_floor` | `None` | skipped when None |
| `call_floor_elements` | `None` | skipped when None |
| `hmc` | `None` | non-serialized (#[serde(skip)]) |
| `witness_tripleton_ceiling` | `false` | skipped when false |
| `ring_moves` | `false` | skipped when false |
| `hold_stereocentres` | `false` | skipped when false |
| `measured_energy_scale` | `false` | skipped when false |
| `branch_on_stall` | `false` | skipped when false |

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

## Results

Session `20261120T102829Z` completed all 972 rows (`9` systems × `9` arms × `12` seeds) serially on core `0` with `0` errors and `0` timeouts. Statistics are computed from raw `rows.csv` only (`72` knockout-vs-`recommended` tests per statistic family; raw two-sided p-values reported with no multiple-comparison claims). On `lj98` and `lj150` (`n >= 98`), `e_no_surfaces` is identical to `recommended` and reproduced `24/24` bit-identical outcomes (`hit`, `calls`, `charged`, `best_bits`). The per-run stdout JSON logged total `hops` and `basins` rather than the `outcome.improvements` trace, so `hop_first_best` is not present in the recorded traces and is not included in `rows.csv`.

<!-- g1-knockout table: start -->
| System | Budget | Arm | Hits | Fisher p | Log-rank p | MW p | Median calls | Median outside/charged (us) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 400000 | `recommended` | 12/12 | - | - | - | 6436.5 | 4.001 |
|  |  | `a_restart_on_stall` | 12/12 | 1 | 0.51 | 0.86 | 6436.5 | 4.254 |
|  |  | `b_no_orbit` | 12/12 | 1 | 0.38 | 0.75 | 5989.5 | 4.849 |
|  |  | `c_no_symmetry` | 12/12 | 1 | 0.89 | 1 | 5925 | 3.855 |
|  |  | `d_no_screens` | 12/12 | 1 | 0.64 | 0.84 | 7424 | 4.343 |
|  |  | `e_no_surfaces` | 11/12 | 1 | 3.07e-05 | 3.43e-04 | 93403 | 4.301 |
|  |  | `f_no_energy_lattice` | 12/12 | 1 | 0.56 | 0.77 | 7300 | 3.882 |
|  |  | `g_no_priors` | 12/12 | 1 | 0.78 | 0.89 | 5138 | 4.156 |
|  |  | `h_no_basin_bias` | 12/12 | 1 | 0.80 | 0.95 | 6117.5 | 4.422 |
| LJ55 | 1000000 | `recommended` | 12/12 | - | - | - | 2245 | 7.172 |
|  |  | `a_restart_on_stall` | 12/12 | 1 | 0.83 | 1 | 2245 | 7.166 |
|  |  | `b_no_orbit` | 12/12 | 1 | 0.28 | 0.40 | 4429 | 8.223 |
|  |  | `c_no_symmetry` | 12/12 | 1 | 0.94 | 0.86 | 2479 | 6.047 |
|  |  | `d_no_screens` | 12/12 | 1 | 0.86 | 0.89 | 1966 | 8.845 |
|  |  | `e_no_surfaces` | 12/12 | 1 | 0.43 | 0.31 | 3413.5 | 9.029 |
|  |  | `f_no_energy_lattice` | 12/12 | 1 | 7.10e-04 | 2.01e-03 | 13892 | 7.709 |
|  |  | `g_no_priors` | 12/12 | 1 | 0.77 | 0.54 | 2289.5 | 8.326 |
|  |  | `h_no_basin_bias` | 12/12 | 1 | 0.42 | 0.58 | 2578.5 | 8.423 |
| LJ75 | 4000000 | `recommended` | 12/12 | - | - | - | 45040 | 11.810 |
|  |  | `a_restart_on_stall` | 12/12 | 1 | 0.64 | 0.75 | 34656 | 11.067 |
|  |  | `b_no_orbit` | 12/12 | 1 | 0.99 | 0.93 | 34369 | 10.289 |
|  |  | `c_no_symmetry` | 12/12 | 1 | 0.23 | 0.21 | 15532 | 8.970 |
|  |  | `d_no_screens` | 12/12 | 1 | 0.45 | 0.80 | 44942 | 11.259 |
|  |  | `e_no_surfaces` | 8/12 | 0.093 | 2.31e-07 | 3.52e-05 | 3497741.5 | 10.387 |
|  |  | `f_no_energy_lattice` | 12/12 | 1 | 0.47 | 0.58 | 31214.5 | 10.876 |
|  |  | `g_no_priors` | 12/12 | 1 | 0.31 | 0.24 | 20329.5 | 8.918 |
|  |  | `h_no_basin_bias` | 12/12 | 1 | 0.22 | 0.27 | 24019 | 11.738 |
| LJ98 | 4000000 | `recommended` | 10/12 | - | - | - | 1336507 | 15.457 |
|  |  | `a_restart_on_stall` | 11/12 | 1 | 0.13 | 0.088 | 608864.5 | 16.103 |
|  |  | `b_no_orbit` | 5/12 | 0.089 | 0.080 | 0.22 | 4000000 | 15.845 |
|  |  | `c_no_symmetry` | 11/12 | 1 | 0.69 | 0.71 | 1514397.5 | 14.934 |
|  |  | `d_no_screens` | 12/12 | 0.48 | 0.47 | 0.79 | 1375020 | 14.553 |
|  |  | `e_no_surfaces` | 10/12 | 1 | 1 | 1 | 1336507 | 15.605 |
|  |  | `f_no_energy_lattice` | 12/12 | 0.48 | 0.017 | 0.046 | 654846.5 | 14.237 |
|  |  | `g_no_priors` | 11/12 | 1 | 0.88 | 0.75 | 1367544.5 | 12.899 |
|  |  | `h_no_basin_bias` | 4/12 | 0.036 | 0.025 | 0.068 | 4000000 | 16.683 |
| LJ150 | 4000000 | `recommended` | 9/12 | - | - | - | 656409 | 36.627 |
|  |  | `a_restart_on_stall` | 12/12 | 0.22 | 2.85e-03 | 0.018 | 152388 | 39.504 |
|  |  | `b_no_orbit` | 7/12 | 0.67 | 0.48 | 0.64 | 1381559.5 | 34.483 |
|  |  | `c_no_symmetry` | 8/12 | 1 | 0.64 | 0.70 | 1087149 | 31.212 |
|  |  | `d_no_screens` | 10/12 | 1 | 0.66 | 0.62 | 522142 | 36.464 |
|  |  | `e_no_surfaces` | 9/12 | 1 | 1 | 1 | 656409 | 37.142 |
|  |  | `f_no_energy_lattice` | 12/12 | 0.22 | 0.19 | 0.40 | 385707.5 | 33.673 |
|  |  | `g_no_priors` | 10/12 | 1 | 0.71 | 0.66 | 435932.5 | 27.784 |
|  |  | `h_no_basin_bias` | 10/12 | 1 | 0.27 | 0.17 | 179490 | 38.105 |
| M55 rho=14 | 1000000 | `recommended` | 3/12 | - | - | - | 1000000 | 5.792 |
|  |  | `a_restart_on_stall` | 3/12 | 1 | 0.96 | 0.94 | 1000000 | 5.859 |
|  |  | `b_no_orbit` | 3/12 | 1 | 0.99 | 1 | 1000000 | 5.462 |
|  |  | `c_no_symmetry` | 1/12 | 0.59 | 0.32 | 0.37 | 1000000 | 5.675 |
|  |  | `d_no_screens` | 1/12 | 0.59 | 0.27 | 0.29 | 1000000 | 5.718 |
|  |  | `e_no_surfaces` | 10/12 | 0.012 | 2.36e-03 | 6.28e-03 | 256530.5 | 6.192 |
|  |  | `f_no_energy_lattice` | 0/12 | 0.22 | 0.070 | 0.079 | 1000000 | 5.421 |
|  |  | `g_no_priors` | 1/12 | 0.59 | 0.27 | 0.29 | 1000000 | 5.358 |
|  |  | `h_no_basin_bias` | 0/12 | 0.22 | 0.070 | 0.079 | 1000000 | 5.131 |
| M80 rho=6 | 4000000 | `recommended` | 12/12 | - | - | - | 240209.5 | 11.845 |
|  |  | `a_restart_on_stall` | 12/12 | 1 | 0.25 | 0.64 | 204610.5 | 11.854 |
|  |  | `b_no_orbit` | 12/12 | 1 | 0.37 | 0.84 | 190075 | 10.551 |
|  |  | `c_no_symmetry` | 12/12 | 1 | 0.22 | 0.44 | 217070 | 10.462 |
|  |  | `d_no_screens` | 12/12 | 1 | 0.18 | 0.40 | 214240 | 12.914 |
|  |  | `e_no_surfaces` | 12/12 | 1 | 5.22e-04 | 8.59e-03 | 31131 | 18.664 |
|  |  | `f_no_energy_lattice` | 0/12 | 7.40e-07 | 2.31e-07 | 1.03e-05 | 4000000 | 11.639 |
|  |  | `g_no_priors` | 12/12 | 1 | 0.38 | 0.40 | 258607.5 | 9.240 |
|  |  | `h_no_basin_bias` | 12/12 | 1 | 0.62 | 0.95 | 160744 | 11.320 |
| M80 rho=14 | 4000000 | `recommended` | 8/12 | - | - | - | 3516411 | 12.470 |
|  |  | `a_restart_on_stall` | 2/12 | 0.036 | 0.014 | 0.022 | 4000000 | 10.652 |
|  |  | `b_no_orbit` | 4/12 | 0.22 | 0.14 | 0.18 | 4000000 | 11.020 |
|  |  | `c_no_symmetry` | 5/12 | 0.41 | 0.45 | 0.74 | 4000000 | 12.278 |
|  |  | `d_no_screens` | 3/12 | 0.100 | 0.083 | 0.16 | 4000000 | 11.973 |
|  |  | `e_no_surfaces` | 9/12 | 1 | 0.34 | 0.23 | 2563682.5 | 12.120 |
|  |  | `f_no_energy_lattice` | 0/12 | 1.35e-03 | 5.70e-04 | 1.08e-03 | 4000000 | 11.395 |
|  |  | `g_no_priors` | 5/12 | 0.41 | 0.30 | 0.38 | 4000000 | 11.136 |
|  |  | `h_no_basin_bias` | 2/12 | 0.036 | 0.027 | 0.057 | 4000000 | 11.204 |
| Au55 | 1000000 | `recommended` | 2/12 | - | - | - | 1000000 | 8.109 |
|  |  | `a_restart_on_stall` | 1/12 | 1 | 0.51 | 0.51 | 1000000 | 8.548 |
|  |  | `b_no_orbit` | 4/12 | 0.64 | 0.36 | 0.38 | 1000000 | 7.736 |
|  |  | `c_no_symmetry` | 1/12 | 1 | 0.58 | 0.65 | 1000000 | 8.139 |
|  |  | `d_no_screens` | 0/12 | 0.48 | 0.15 | 0.17 | 1000000 | 5.148 |
|  |  | `e_no_surfaces` | 6/12 | 0.19 | 0.078 | 0.079 | 991482 | 12.049 |
|  |  | `f_no_energy_lattice` | 0/12 | 0.48 | 0.15 | 0.17 | 1000000 | 8.045 |
|  |  | `g_no_priors` | 2/12 | 1 | 0.96 | 0.96 | 1000000 | 6.923 |
|  |  | `h_no_basin_bias` | 2/12 | 1 | 0.96 | 0.96 | 1000000 | 7.987 |
<!-- g1-knockout table: end -->
