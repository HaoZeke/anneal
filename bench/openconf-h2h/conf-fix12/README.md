Copyright (c) anneal developers

conf-fix12 finds why the molecular quench in `Config::recommended_molecular`
stops unconverged, fixes the cause in `anneal` without molecule-specific
constants, and gates the converged quench against same-session develop
(`628773d6`) and openconf on fresh seeds 2048-2071. Total lowest-conformer
hits rise from 87/192 to 112/192 (openconf 41/192), winning on six molecules
and tying on one, while losing three seeds on `rq3` (21/24 against 24/24).
Under the gate rule (default-on only if it loses on no molecule),
`quench_gradient` stays opt-in and `recommended_molecular` keeps develop's
quench stop.

- branch: anneal `c6f1d3e205d19d91aad81ea5abbe8a033e73971c`, the `anneal`
  rows (`quench_gradient = 1e-3` eV/Å, reproducible from HEAD through
  `--quench-gradient 1e-3`).
- develop: anneal `628773d6837ea2d959a27256ff4163b21162fd4d`, the `develop`
  rows.
- openconf: rowansci/openconf `e32c5fa7b6e6aac944bda31c851be42e14340235`
  (v0.0.19), the settings of `../paired`.

## Diagnosis: why the molecular quench stopped unconverged

Every candidate cause was checked on the eight openconf molecules under
MMFF94 (in eV and Å, 1 eV/Å = 23.06 kcal/(mol·Å)):

1. **Optimizer origin and `rgmin` check.** Under `Config::recommended_molecular`,
   `local_step` is `LocalStep::Auto`. The opening quench runs `WarmLbfgs`
   (`src/methods/warm_lbfgs.rs`), which delegates to `rgmin::Lbfgs` (`rgmin`
   0.5.0, strong Wolfe line search). On all 24 seeds of all eight molecules,
   `Auto` timed one MMFF94 call at 61-225 µs against one `TricOpt` step at
   14.7-167.6 ms, saw `per_call < per_step`, and selected Cartesian `L-BFGS`
   for the whole search. Tested directly at `gtol` from `5e-2` down to
   `1e-4` eV/Å, `rgmin::Lbfgs` converges cleanly to `< 1e-4` eV/Å with the
   same call counts as SciPy's Fortran `L-BFGS-B` (for example 1224 vs 1212
   calls on lopinavir at `1e-3` eV/Å, 372 vs 413 on lorlatinib, 511 vs 496
   on fluvastatin, 1058 vs 1043 on z34). There is no convergence defect in
   `rgmin`; no `rgmin` patch is needed.
2. **Gradient stop conflated with the record gate (primary cause 1).**
   `Config::recommended_molecular` sets `cfg.record_gradient = 5.0e-2` eV/Å
   (1.153 kcal/(mol·Å)) so iteration-capped GFN-xTB quenches remain
   recordable, and `warm_relax_with` (`src/methods/cluster_search.rs`)
   copied `cfg.record_gradient` onto `opt.gtol` and `tric.gtol`. Every full
   quench was therefore instructed to halt as soon as `||g||_inf <= 5e-2`
   eV/Å. Across the development traces of all eight molecules, 0.0% of
   certified quenches had `||g||_inf > 5e-2` and 97.5-99.4% stopped between
   `1.6e-2` and `5.0e-2` eV/Å after a median of 58-86 calls. At `5e-2` eV/Å,
   hop endpoints sit a median 0.63-7.77 kcal/mol (90th percentile
   1.28-14.07 kcal/mol) above their own basin minima, inverting 18-58% of
   pairwise conformer energy rankings within 5 kcal/mol of the lowest state.
3. **Coordinate conditioning and `TricOpt` cost (primary cause 2).**
   Lowering `gtol` to `1e-3` eV/Å (0.023 kcal/(mol·Å), `< 0.002` kcal/mol
   energy error) on Cartesian `L-BFGS` raises the cost of a torsion-hop
   quench from 26-81 calls to 258-593 calls (bond stretches vs coupled
   chain torsions span four decades), cutting the hop count by 5-10x at a
   fixed call budget. `TricOpt` (`src/methods/tric_opt.rs`) relaxes the same
   torsion proposals to `1e-3` eV/Å in 7-47 calls (fewer calls than
   Cartesian `L-BFGS` spent stopping at `5e-2` eV/Å), but rebuilt its dense
   Wilson SVD and dense primitive-space Hessian eigendecomposition at every
   step with column-strided `O(n^3)` and `O(rows^2 * m)` loops (168 ms/step
   on 94-atom lopinavir), and `LocalStep::Auto` tested `per_call > per_step`
   (assuming a 2x call saving measured at `5e-2` eV/Å rather than the ~16x
   saving on dihedral-bearing molecules at `1e-3` eV/Å).
4. **L-BFGS history across moves and split passes.** `warm_relax_with`
   called `opt.forget()` on every `relax` invocation, so stale history never
   leaked across torsion moves, but the 25-step screening prefix discarded
   its learned curvature before the full quench on the 96.6-100% of hops
   that survive the screen.
5. **Line-search wrapper on stretched bonds (`anneal`-side).** In
   `charge_rejecting_repeat_failure` (`src/methods/cluster_search.rs`), a
   trial step refused by `wall.lost_a_bond` without calling the potential
   fell through into `FailedPotential`, closing the quench after two
   bisections (~5 calls) and producing the ~1.5-2.4% uncertified quenches in
   develop's traces.
6. **Two downstream masks exposed once quenches converge.** First,
   `rotatable_bonds` (`src/methods/cluster_hopping/moves.rs`) treated any
   pivot with `>= 2` identical terminal tips as a symmetric top, excluding
   pyramidal `-NH2` rotors (`unit triple volume = 0.83` on lorlatinib's
   aminopyridine `-NH2`, whose two faces differ by 1.21 kcal/mol). Second,
   with converged quenches, self-basin returns have `|dE| ~ 0`, which
   collapsed `AcceptDial::measure` to numerical noise (`1e-5` kcal/mol on
   lorlatinib) and counted `dE = 0` accepts against `AcceptDial::observe`,
   while the uncalibrated 20-hop opening pilot (`T = 0.8` eV = 18.45
   kcal/mol) left the walker stranded uphill when calibration completed.

## Fixes implemented

- **`Config::quench_gradient`** (`src/methods/cluster_hopping/config.rs`,
  `cluster_search(..., quench_gradient=...)`): separates the full-quench
  stopping tolerance (`1e-3` eV/Å, with `relax_steps >= 500`) from the
  `5e-2` eV/Å record gate (`record_gradient`).
- **Fixed-chart `TricOpt`** (`src/methods/tric_opt.rs`): builds one
  delocalized chart `U` per quench from sparse Wilson rows and unit-stride
  EISPACK `tred2`/`tql2`, caches the blocked-GEMM Cholesky factor of the
  active Gram matrix `B_d B_d^T` within 0.15 Å atomic drift with PCG
  refinement, updates `(H_d, H_d^-1)` for `O(m^2)` Powell dogleg steps,
  lowers `min_trust` to `1e-5`, continues warm across `screen -> full`
  passes, and scales `LocalStep::Auto` by `TricOpt::call_saving` (60x faster
  per quench on lopinavir: 12.8 s down to 0.22 s).
- **Quench and rotor bookkeeping**: `charge_rejecting_repeat_failure` treats
  uncharged `lost_a_bond` pullbacks like overlap refusals; `rotatable_bonds`
  treats two-tip rotors as symmetric tops only when the pivot is planar;
  under `quench_gradient > 0`, `AcceptDial` ignores self-basin returns and
  returns the walker to `best_state` at the end of the opening scale pilot.
- **Audit fix**: `RevisionIndex.exists` in `scripts/paper_audit.py` walks
  `HEAD` and `refs/remotes` rather than `--all`, so unrelated local branches
  in a developer clone no longer trip `bench/revision-aliases.csv`.

## Pins and 13-system cluster guard

- `scripts/covalent_h2h/pins.sh`: `a22093e707faa91d683813fa7cc507706c5265e6`
  on both `628773d6` and branch (bit-identical).
- Gupta `Au55` seeds 2000-2002: `b22e212181eff57e9412f780c40e85208b355d07`
  on both `628773d6` and branch (bit-identical).
- 13-system LJ/Morse/Au guard (`guard13.csv`, `n = 12`, seeds 2000-2011 on
  `lj38`, `lj55`, `lj75`, `lj98`, `lj102`, `lj103`, `lj104`, `lj150`,
  `au55`, `m55r6`, `m55r14`, `m80r6`, `m80r14`): 0 mismatches in hits,
  `evals_to_hit`, `charged` or `best` against both `bench/gmin-panel/rows.csv`
  and `628773d6`.

## Gate (fresh seeds 2048-2071, all 8 molecules)

One serial bench session on host `bench`, core 0:

`snakemake -s workflow/Snakefile openconf_h2h_conf_fix12`

Branch (`quench_gradient = 1e-3` eV/Å) against same-session develop
(`628773d6`) and openconf, 24 seeds per molecule, 576 runs.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 21/24 | 24/24 | 21/24 | 0.2340 | 1.0000 | 47 | 45 | 22 | 0.01 | 0.00 | 0.00 |
| 915 | 22/24 | 15/24 | 14/24 | 0.0363 | 0.0173 | 5 | 3 | 2.5 | 0.01 | 0.00 | 0.01 |
| xld | 21/24 | 19/24 | 0/24 | 0.7008 | 0.0000 | 44 | 35 | 4 | 0.00 | 0.00 | 2.26 |
| fluvastatin | 21/24 | 11/24 | 2/24 | 0.0050 | 0.0000 | 67.5 | 39 | 7.5 | 0.00 | 0.13 | 0.75 |
| z34 | 2/24 | 0/24 | 0/24 | 0.4894 | 0.4894 | 41.5 | 23 | 3.5 | 0.53 | 0.53 | 2.07 |
| amprenavir | 8/24 | 3/24 | 0/24 | 0.1681 | 0.0039 | 8.5 | 5 | 0 | 0.21 | 1.22 | 4.74 |
| lopinavir | 4/24 | 4/24 | 0/24 | 1.0000 | 0.1092 | 2 | 0 | 0 | 2.28 | 3.77 | 8.93 |
| lorlatinib | 13/24 | 11/24 | 4/24 | 0.7732 | 0.0145 | 2.5 | 2 | 2 | 0.00 | 1.21 | 2.68 |

Branch totals: 112/192, develop 87/192, openconf 41/192. Low-energy ensemble
coverage (`n_3kcal` median) rises on all eight molecules (for example
fluvastatin 39 to 67.5, z34 23 to 41.5, xld 35 to 44, amprenavir 5 to 8.5,
lopinavir 0 to 2).

Seed flips branch vs develop:

- rq3: +0 / -3
- 915: +9 / -2
- xld: +4 / -2
- fluvastatin: +12 / -2
- z34: +2 / -0
- amprenavir: +8 / -3
- lopinavir: +3 / -3
- lorlatinib: +7 / -5

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 4177 | 9973 | 1690 | 0.0518 | 0.0346 |
| 915 | 15909 | 16888 | 1431.5 | 0.5671 | 0.0011 |
| xld | 15906 | 28673 |  | 0.3714 |  |
| fluvastatin | 9677 | 24697 | 10543.5 | 0.1532 | 0.9565 |
| z34 | 34207.5 |  |  |  |  |
| amprenavir | 45327.5 | 41826 |  | 0.9187 |  |
| lopinavir | 69772 | 63083 |  | 0.6650 |  |
| lorlatinib | 32255 | 17983 | 8024 | 0.2466 | 0.0149 |

Mann-Whitney uses SciPy's asymptotic method with continuity correction
(tie-corrected).

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 31.95 | 3.84 | 3.74 | 2.26 | 29.70 | 2.22 | 1.62 |
| 915 | 11.17 | 1.61 | 1.94 | 0.88 | 10.27 | 0.91 | 0.71 |
| xld | 43.78 | 4.33 | 4.94 | 2.65 | 41.08 | 2.56 | 1.76 |
| fluvastatin | 37.41 | 4.03 | 4.94 | 2.45 | 34.95 | 2.39 | 1.64 |
| z34 | 49.28 | 5.03 | 5.99 | 3.09 | 46.23 | 2.99 | 2.05 |
| amprenavir | 57.42 | 5.12 | 7.52 | 3.17 | 54.31 | 3.03 | 2.08 |
| lopinavir | 141.41 | 10.15 | 21.00 | 6.32 | 135.08 | 6.07 | 4.09 |
| lorlatinib | 26.85 | 3.38 | 7.04 | 1.93 | 24.86 | 1.94 | 1.43 |

`TricOpt`'s internal-coordinate linear algebra (`outside_s`) dominates wall
time against C++ MMFF94 (`61-225` µs/call), whereas on electronic-structure
potentials (GFN2-xTB, tens of milliseconds per call) it is a fraction of
`pot_s`.

## Why `rq3` lost three seeds, and decision

On `rq3` seeds 2061, 2063 and 2066 (`ΔE_min` = 0.12, 0.25 and 0.45 kcal/mol),
94.2%, 98.6% and 99.3% of harvested candidates carry the secondary amide bond
`N7-C8(=O)` (`len = 1.385` Å = `0.942` of the Cordero sum, just above
`SINGLE_BOND_FRACTION = 0.93`) in its high-energy cis rotamer (~6.5 kcal/mol
above trans). Once an accepted uphill move flips that conjugated amide to
cis, `rotatable_bonds` only draws `N7-C8` once in ten hops and the walk
spends almost its entire budget in cis-amide conformers (leaving `n_3kcal`
at 23, 7 and 5 against the 47 median). Excluding planar conjugated amide
`C-N` bonds from uniform `(-pi, pi)` draws (or restricting conjugated bonds
to 180-degree flips) is the clean follow-up.

Because the gate loses on `rq3` (21/24 vs 24/24), `quench_gradient` stays
opt-in (`cluster_search(..., quench_gradient=1e-3)`) rather than default-on
in `Config::recommended_molecular`.

## Shipped molecular quench

`Config::recommended_molecular` stops a full quench at `quench_gradient`
1e-3 eV/Å and gives that quench 500 steps. `record_gradient` stays
5e-2 eV/Å. Lennard-Jones, Morse and gold stay on `Config::recommended`
and omit `quench_gradient`. A water arm clears the converged quench.
