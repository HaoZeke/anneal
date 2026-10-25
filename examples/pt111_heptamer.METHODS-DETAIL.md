# Pt(111) heptamer

Driver: `examples/pt111_heptamer.rs`. Registry run `r9-pt111-heptamer` in `paper/seeds.toml`.

## Algorithm

The geometry, the cell, and the frozen mask come from the `.con` file. An atom the file marks fixed stays at those coordinates, and its search gradient is zero. Free atoms are split into layers wherever neighbouring z values differ by more than `LAYER_GAP_A` (1.0 angstrom). The island is the smallest layer. A tie keeps the layer with the highest total z. On the OptBench heptamer that layer is the seven adatoms. The remaining free layers are the upper substrate: they keep a live gradient, so the quench may relax them, and they are not hop seeds. `place_adsorbates` moves only the island. The Morse parameters come from the OptBench problem file. The potential is the scalar `morse/smooth/linear` sum: for each pair `i < j` inside the cutoff, the Morse value minus its value at the cutoff, minus `(r - rc)` times the Morse derivative at the cutoff. A periodic axis subtracts the cell length times the half-even rounding of the reduced separation.

Each seed builds a start with `place_adsorbates` and runs `search_from`. That quench is `WarmLbfgs` (rgmin L-BFGS). One force call is one energy-and-gradient evaluation charged to the ledger. The configuration is `Config::recommended_molecular` or `Config::for_molecular` with `energy_scale` 1.0, `active_region` equal to the island and shell count 0, and `length_scale` twice the largest covalent radius among the island atoms.

After the seeds, each pair of quenched endpoints starts an rgsaddle `MinModeKind::Dimer` session on the island coordinates. The mode is the island displacement between the endpoints. The start is their midpoint. Atoms outside the island stay on the template during the dimer. Translation stops when the maximum force component falls under 1e-3. rgsaddle 0.3.0 has no force-gate field on `MinModeConfig`.

Reference `.con` files under the reference root are evaluated with the same potential. `mid.con` is skipped. A path whose components include `saddle` or `saddles` is a saddle geometry. Every other `.con` is a minimum. The compared coordinates are that file's island layer, selected by the same gap rule. A match is an absolute energy difference of at most 0.05 eV and an RMSD of at most 0.1 angstrom on the island. With the `ira` feature the RMSD is IRA (`CanonicalOrder`, `kmax_factor` 1.8). Without it the RMSD is the best permutation in the slab frame, and the log says `cartesian-permutation`. A saddle counts toward the barrier set when its evaluated energy minus the lowest evaluated reference minimum is below `barrier_below_eV` from the problem file. Saddle coordinates are not stored in this repository. If the reference tree contains none, the saddle match count is zero and false edges are unscored.

## Parameters

| quantity | value | source |
| --- | --- | --- |
| seeds | 4000 through 4047 inclusive (48) | `paper/seeds.toml` run `r9-pt111-heptamer` |
| arms | `recommended`, then `plain` | driver argument |
| budget | 4605 force calls per seed | literal default in `main` when the budget argument is omitted |
| screen steps | 10 | `SCREEN_STEPS`, default in `configure` |
| relax steps | 150 | `RELAX_STEPS`, default in `configure` |
| dimer steps | 20 | `DIMER_STEPS` |
| dimer rotations | 20 | `DIMER_ROTATIONS` |
| dimer pairs | every quenched pair | `DIMER_PAIRS`, default `usize::MAX` |
| dimer force gate | L2 norm, tolerance `1.0e-3` | `MinModeConfig` in `certify_edge` |
| energy window | 0.05 eV | `ENERGY_WINDOW_EV` |
| RMSD window | 0.1 angstrom | `RMSD_WINDOW_A` |
| island layer gap | 1.0 angstrom | `LAYER_GAP_A` |
| Morse D, alpha, r0, cutoff, pbc, barrier | read from the problem file | not literals in the driver |
| stopping rule | the ledger budget, then the dimer step cap or the force tolerance | `Ledger::new`, `MinModeSession::run` |

The OptBench problem `problems/saddle-search/pt-island.toml` supplies `D = 0.7102`, `alpha = 1.6047`, `r0 = 2.897`, `cutoff = 9.5`, `pbc = [true, true, false]`, `barrier_below_eV = 1.5`, `connected_saddles = 27`, `pair_style = "morse/smooth/linear"`. `connected_saddles` is printed as the problem statement. It is not a match count.

Hit counts, distinct minima, matched saddles, false edges, and wall times are TBD-from-run. Smoke output is not a paper number.

## Provenance

Each seed writes `pt111-<arm>-seed<seed>.provenance.json` and a `RunManifest` whose `engine.build_identity` is `anneal-core` plus the package version. The provenance file records the hardware model from `/proc/cpuinfo`, the core count, `RAYON_NUM_THREADS` or the core count, `rustc -vV`, and `git rev-parse HEAD`. Those strings are TBD-from-run.

Pins in `Cargo.toml` at this tree: rgmin 0.2.0 revision `20832fd169932a1cd0d47ac84dc77fa9b435f940`, rgsaddle 0.1.0 revision `84fe96bfc1ed163379c0af8bd664b74bf71407d1`. The driver needs the `rgsaddle` feature.

## Reproduction

```text
pt111_heptamer pos.con problems/saddle-search/pt-island.toml reference_root 4605 48 4000 recommended
pt111_heptamer pos.con problems/saddle-search/pt-island.toml reference_root 4605 48 4000 plain
```

`pos.con` is the OptBench heptamer geometry. `reference_root` is a directory of OptBench `.con` files. Build with `--features rgsaddle`. Add `--features ira` for the IRA RMSD.
