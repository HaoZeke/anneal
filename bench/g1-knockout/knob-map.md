# G1 Component Knockout Knob Map

Base revision: `61fa67cf380e46876b569da45e7cc091647a775b` (`develop`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`).

Every knockout arm starts from `Config::recommended(n)` (`src/methods/cluster_hopping/config.rs:1786`, forwarding to `Config::recommended_with_scales(n, 1.0, 1.0)` at `src/methods/cluster_hopping/config.rs:1827`) and mutates only the fields listed for that arm. All other fields remain at their `recommended` values.

## Arm-to-Field Mapping

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
| `g_no_priors` | `prior_reseed` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:229` (`REMOVED_PRIOR_KEYS`) |
| `g_no_priors` | `prior_growth` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:230` (`REMOVED_PRIOR_KEYS`) |
| `g_no_priors` | `prior_core` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:231` (`REMOVED_PRIOR_KEYS`) |
| `g_no_priors` | `prior_motif` off | *(no counterpart at BASE; removed)* | *(absent)* | *(absent)* | `src/methods/cluster_hopping/config.rs:233` (`REMOVED_PRIOR_KEYS`) |
| `h_no_basin_bias` | basin bias off | `bias_height` | `0.25` (`LennardJonesPreset::BIAS_HEIGHT * energy_scale`) | `0.0` | `src/methods/cluster_hopping/config.rs:409,2123`; `src/methods/cluster_hopping/preset.rs:9` |

## Notes on Specific Arms

- **`a_restart_on_stall`**: Stall restarts run at `src/methods/cluster_hopping.rs:5811-5828` only when `restart_on_stall && !escape_on_stall`. Disabling `escape_on_stall`, `trail_on_stall`, and `tabu_on_stall` while enabling `restart_on_stall` replaces the stall-escape bundle with random restart at the default `restart_patience = 5000` (`src/methods/cluster_hopping/config.rs:2206`).
- **`d_no_screens`**: With `return_screen = false` and `bayes_screen = false`, partial-relaxation screening falls back to the base `with_scales` calibrated margin screen (`calibrated_screen = true`, `screen_margin = 2.0`, `src/methods/cluster_hopping.rs:7564-7571`).
- **`e_no_surfaces`**: `Config::recommended(n)` installs `TwoPhase::relative(0.7, 1.0)` only for `n < 98` (`src/methods/cluster_hopping/config.rs:1849-1853`). On `lj38`, `lj55`, `lj75`, `m55r14`, `m80r6`, `m80r14`, and `au55`, clearing `surfaces` removes that diameter-penalty surface. On `lj98` and `lj150`, `recommended` already has `surfaces = []`, so `e_no_surfaces` is configuration-identical to `recommended`; the arm is still run in the balanced schedule and serves as an exact determinism check. Runs execute through `cluster_search::search_from` (`warm_relax`), which honors `cfg.surfaces`.
- **`g_no_priors`**: Four of the five named structure priors (`prior_reseed`, `prior_growth`, `prior_core`, `prior_motif`) were removed as inert prior keys before BASE (see `REMOVED_PRIOR_KEYS` at `src/methods/cluster_hopping/config.rs:228-235`; JSON/TOML/env parsers reject them with `RemovedConfigKeyError`). Only `prior_soap` exists on `Config` at BASE. Setting `prior_soap = false` removes `ClusterMove::Soap` from the proposal library (`src/methods/cluster_hopping/moves.rs:143` and `src/methods/cluster_hopping.rs:2570`), so `soap_mode` is never consulted.
- **`h_no_basin_bias`**: `BasinBias::new` accepts `w0 = 0.0` (`src/bias.rs:1270`), and `bias_gamma` remains at `5.0` (must exceed `1.0`). With `bias_height = 0.0` and `adaptive_height = false`, every deposit adds `0.0` (`src/bias.rs:1558,1582,1610`), so the basin bias potential is identically `0.0` throughout the run.
