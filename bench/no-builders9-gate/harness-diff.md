# blj105n38 driver difference

Same develop library, same potential, seeds 2000-2011,
budget 400000, core 0, every thread variable 1. Hit means a
finite best at or below -177.2090846748 plus 0.001.

## Shared fields

Preset `Config::recommended` for 38 atoms. Potential
`BinaryLennardJones` with 8 atoms of species A and 30 of
species B. Every well depth is 1. sigma_AA is 1, sigma_BB is
1.05, sigma_AB is their mean 1.025. There is no cutoff. The
budget is 400000 and a call is charged inside the relax and
inside the gradient. The search stops at the reference plus
0.001. The start is `StdRng::seed_from_u64(seed)` through
`random_cluster_in_radius`, and the search draws a second
`StdRng::seed_from_u64(seed)`. No budget variable is set.

The release `h2h_blj` binary and a second build of that same
search score 2/12, on the same seeds (2002 and 2008) and the
same best energies. That is the driver this gate used, and it
is the driver in the land4 gate-b harness. The author runner
lists Lennard-Jones and Morse only. It has no blj105n38 entry.
The develop count of 2/12 stated for that series matches this
binary seed by seed.

The driver that scores 8/12 calls `search_from` and passes
per-atom diameters. Its seed 2000 best is -176.6029859958.
The release binary's seed 2000 best is -176.71346390.

## One change at a time

Each row is twelve seeds. `sigma` means diameters are passed,
which also writes neighbour_cutoff and the container. The
labels stay 200 and 201 unless the row says otherwise.

| Change | Hits | Pattern, seeds 2000-2011 |
| --- | ---: | --- |
| release binary, no diameters, run_with_gradient | 2/12 | 001000001000 |
| labels 1 and 2 only | 2/12 | 001000001000 |
| diameters only | 5/12 | 011100001100 |
| labels and diameters | 5/12 | 011100001100 |
| search_from only | 1/12 | 000001000000 |
| search_from and labels 1 and 2 | 1/12 | 000001000000 |
| search_from and diameters | 8/12 | 010111011011 |
| search_from, diameters, labels 1 and 2 | 8/12 | 010111011011 |

Labels do not move a hit. The two rows that score 8/12 have
the same best energy and the same charged count on every seed.
Diameters alone, with the original search, score 5/12.
`search_from` alone scores 1/12. The 8/12 count is the two
changes together: attach diameters, and call `search_from`.

Attaching diameters leaves min_separation at 0.85. It sets
neighbour_cutoff from 1.6 to 1.68 and the container from
3.025777866119067 to 3.17706675942502. The start radius is the
container, so the starting coordinates change. The no-diameter
starts share one sha256. The diameter starts share another,
and that second digest is the start of the 8/12 driver. Both
digests are in `discrepancy-rows.csv`.

## Resolved config

The resolved document has 142 fields. The 8/12 reproduction
that keeps labels 200 and 201 differs from the 2/12 driver in
three of them. Every other field is equal, including
min_separation, the preset flags, and the move library.

| Field | 2/12 driver | diameters on |
| --- | --- | --- |
| `container` | 3.025777866119067 | 3.17706675942502 |
| `neighbour_cutoff` | 1.6 | 1.6800000000000002 |
| `species_sigma` | null | [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05] |

The published 8/12 source also writes species labels 1 and 2
instead of 200 and 201. That is a fourth resolved field. It
does not change the best energy: the 8/12 pattern above is
identical with the labels left at 200 and 201.

Equal fields, value from the 2/12 driver:

- `active_region` = null
- `adaptive_height` = false
- `adaptive_screen` = false
- `allocate_moves` = true
- `angular_moves` = false
- `angular_target` = 0.5
- `anneal_diversity` = false
- `bayes_exploration` = 0.1
- `bayes_screen` = true
- `bayes_threshold` = 0.4666666666666667
- `bayes_warmup` = 300
- `bias_by_rung` = false
- `bias_gamma` = 5.0
- `bias_height` = 0.25
- `bond_tolerance` = 1.25
- `budget_window` = false
- `calibrate_quantile` = 0.9
- `calibrate_radius` = false
- `calibrate_warmup` = 200
- `calibrated_screen` = true
- `construct_width` = 4
- `contextual_floor` = 0.1
- `contextual_moves` = false
- `continuous_symmetry.group` = measure
- `continuous_symmetry.interval` = 10
- `cov_perturb` = false
- `covalent_cutoff` = 1.3
- `delayed_acceptance` = false
- `depth_reward` = true
- `displacement_only` = false
- `diversity_floor` = 0.75
- `energy_bias` = false
- `energy_lattice` = true
- `energy_scale` = 1.0
- `energy_trace` = false
- `escape_amplitude` = 0.25
- `escape_epsilon` = 0.0001
- `escape_lanczos_steps` = 16
- `escape_max_climb` = 24
- `escape_on_stall` = true
- `escape_overshoot` = 1.5
- `escape_stall_factor` = 2.0
- `escape_stall_patience` = 5000
- `exchange_metropolis` = false
- `finish_improving` = true
- `flat_histogram` = false
- `flat_quantile` = 0.5
- `flat_sweep` = 400
- `frozen` = null
- `funnel_period` = 20000
- `group_cutoff` = 3.4
- `height_revisits` = 4.0
- `incumbent_gap` = 0.0
- `jump_on_stall` = false
- `jump_patience` = 5000
- `jump_step` = 0.38
- `jump_steps` = 10
- `keying` = distances
- `keying_sigma` = 2.5
- `ladder_mode` = Shipped
- `ladder_pilot` = 200
- `ladder_target_accept` = 0.23
- `ladder_top` = 4.0
- `ladder_window` = 10
- `length_scale` = 1.0
- `max_hops` = null
- `md_escape` = false
- `md_escape_dt` = 0.005
- `md_escape_kinetic` = 1.0
- `md_escape_max_steps` = 2000
- `md_escape_minima` = 2
- `md_escape_soften` = 0
- `merge_radius` = 0.7
- `min_separation` = 0.85
- `minima_hopping` = false
- `morphology_scale` = 1.0
- `move_library` = lean_burst
- `n_points` = 38
- `novel_reward` = false
- `orbit_complete_on_new` = true
- `path_images` = 9
- `path_on_stall` = false
- `point_symmetrise` = false
- `point_symmetrise_every_accept` = false
- `point_symmetrise_on_new` = false
- `polish_records` = 0
- `probe_screen` = false
- `quench_confidence` = 2.0
- `quench_once` = false
- `quench_warmup` = 4
- `record_gradient` = 0.001
- `relax_steps` = 1000
- `reoccupy_interval` = 5000
- `replicas` = 1
- `restart_on_stall` = false
- `restart_patience` = 5000
- `return_polish` = 0
- `return_polish_after` = 0
- `return_screen` = true
- `rigid_body_geometry` = null
- `schema` = anneal-cluster-config-v1
- `screen_margin` = 2.0
- `screen_steps` = 25
- `settle_iters` = 20
- `shape_keyed` = false
- `shared_deposits` = 8
- `shared_visit_policy` = tabu
- `soap_class_residual` = false
- `soap_mode` = flexible
- `soap_repel` = false
- `soft_modes` = 6
- `soft_perturb` = false
- `soft_steps` = 30
- `staged_quench` = false
- `stall_patience` = 60
- `statistical_temperature` = false
- `superbasin_escape` = false
- `superbasin_features` = false
- `superbasin_period` = 2000
- `superbasin_quotient` = false
- `superbasin_report` = false
- `surfaces` = [{"cutoff":{"Relative":0.7},"beta":1.0,"mu":0.0,"anisotropic":false}]
- `surrogate_tolerance` = 0.5
- `swap_period` = 50
- `symmetrise_core_fraction` = 0.6
- `symmetrise_cutoff` = 2.5
- `symmetrise_on_stall` = true
- `symmetrise_patience` = 2000
- `symmetry_merge_radius` = 0.7
- `symmetry_tolerance` = 0.5
- `tabu_capacity` = 8
- `tabu_on_stall` = true
- `temperature` = 0.8
- `theta` = 0.5
- `trace_quenched` = false
- `track_funnels` = false
- `trail_on_stall` = true
- `two_phase` = null
