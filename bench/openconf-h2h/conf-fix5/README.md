Copyright (c) anneal developers

The three-arm paired run of `../conf-fix` with the plateau reheat, on the
same eight molecules, seeds 2000-2023 and MMFF94 through RDKit 2025.09.6
at a dielectric of one. One serial session under `sudo unshare --uts`,
hostname bench, every job on core 0, `snakemake -s workflow/Snakefile
openconf_h2h_conf_fix5`.

- branch: anneal ef26ee0f6b15aa5cf8bce20f8a03628040ca8466 (053f4cb7 in
  this series, the same tree), the `anneal` rows: the plateau reheat on
  in `recommended_molecular`, over the stereocentre hold and the measured
  energy scale.
- develop: anneal 93c14173e433ad7deacc97992d3bc6a2a5fae395, the `develop`
  rows: the stereocentre hold and the measured energy scale.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

Budgets, arm order, hits, ΔE_min and the tests are those of `../conf-fix`:
a hit is a reported conformer that polishes to the lowest valid conformer
of the session's three arms, and a conformer whose stereocentres or double
bonds differ from the SMILES is invalid and dropped.

The reheat stays off in `recommended_molecular`: lorlatinib's trap
narrows but does not go away, and on dev seeds it costs hits elsewhere.

## Lorlatinib's trap

With the measured energy scale lorlatinib's chain runs at a few tenths of
a kcal/mol, the size of its ring puckers, and in two runs of `../conf-fix4`
(seeds 2004 and 2021) it never leaves the macrocycle it starts in, 11.7 to
11.9 kcal/mol above the lowest conformer. Wrapping the energy on dev seeds
3000-3059 shows what sets those runs apart: their running minimum last
drops in the first 15% of the budget, while runs that reach the lowest
conformer keep improving to the end. Their acceptance, 0.35 to 0.48, lies
inside the other runs' 0.29 to 0.77, so the acceptance dial cannot see
them. Runs that do reach the lowest conformer often sit as long between
incumbents early on, so no plateau length read from the walk alone
separates the two.

With `plateau_reheat` the temperature returns to the declared scale once,
in the second half of the budget, the walk has gone as many hops without
a new incumbent as it took to reach the one it holds, never fewer than
the 20 measuring hops, and to the measured scale at the next new
incumbent. The deposit height stays measured.

Dev seeds 3000-3059 of the eight molecules, `recommended=True`, a hit
taken as a polished energy within 0.03 kcal/mol of the lowest conformer
of `../paired` (no RMSD), 480 runs per arm:

| arm | hits | lorlatinib hits | lorlatinib runs over 5 kcal/mol up |
|---|---|---|---|
| measured scale, no reheat (develop) | 222 | 34 | 6 |
| reheat until the next incumbent, from the first plateau | 174 | 30 | 1 |
| 20-hop reheats, deposit height reheated too | 194 | 33 | 5 |
| 20-hop reheats, temperature only | 201 | 24 | 7 |
| no reheat, upper quartile of the opening changes | 156 | 29 | 5 |
| no reheat, 90th percentile of the opening changes | 132 | 34 | 1 |
| reheat after 30% of the budget | 181 | 26 | 5 |
| reheat after half the budget (this branch) | 197 | 30 | 3 |

A reheat that holds from the first plateau frees lorlatinib and loses 48
hits, where walks still improving cold are made to walk hot.
Deposits laid at the declared scale outlive a reheat and bury the
conformers near them. A larger measured scale without a reheat costs more
still. Waiting for half the budget keeps most of the measured walk and
halves the trapped runs, at 25 hits.

## 24 seeds

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 20/24 | 1.0000 | 0.1092 | 45.5 | 45 | 22.5 | 0.00 | 0.00 | 0.00 |
| 915 | 12/24 | 15/24 | 10/24 | 0.5612 | 0.7725 | 4 | 4 | 2 | 0.01 | 0.00 | 0.01 |
| xld | 16/24 | 16/24 | 2/24 | 1.0000 | 0.0001 | 28 | 33 | 4 | 0.00 | 0.00 | 2.74 |
| fluvastatin | 11/24 | 10/24 | 8/24 | 1.0000 | 0.5556 | 37 | 39.5 | 7 | 0.05 | 0.05 | 0.48 |
| z34 | 1/24 | 0/24 | 0/24 | 1.0000 | 1.0000 | 24 | 24 | 1.5 | 0.53 | 0.53 | 2.38 |
| amprenavir | 1/24 | 3/24 | 0/24 | 0.6085 | 1.0000 | 2.5 | 3 | 0 | 2.23 | 2.08 | 4.19 |
| lopinavir | 2/24 | 2/24 | 0/24 | 1.0000 | 0.4894 | 0 | 0 | 0 | 6.69 | 5.89 | 9.29 |
| lorlatinib | 13/24 | 12/24 | 5/24 | 1.0000 | 0.0355 | 3 | 3 | 2 | 0.00 | 0.60 | 2.68 |

The branch hits 80 of 192 runs, develop 82 and openconf 45; no molecule
moves significantly against develop. 107 of the 192 anneal runs differ
from develop's; the other 85 match it row for row. Seed
2021 of lorlatinib leaves the macrocycle and seed 2004 does not: one run
11.9 kcal/mol up against develop's two, and a median ΔE_min of 0.00
against 0.60. Neither arm has an invalid candidate.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 12613 | 12613 | 4165 | 1.0000 | 0.0027 |
| 915 | 15479 | 18883 | 4339 | 0.5252 | 0.0062 |
| xld | 36271.5 | 36271.5 | 10363 | 0.9398 | 0.0523 |
| fluvastatin | 17079 | 13292 | 3523.5 | 0.6718 | 0.0328 |
| z34 | 59061 |  |  |  |  |
| amprenavir | 57119 | 56050 |  | 0.6374 |  |
| lopinavir | 48892.5 | 48892.5 |  | 1.0000 |  |
| lorlatinib | 56016 | 55483.5 | 7971 | 0.7853 | 0.0009 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.85 | 3.86 | 3.83 | 2.25 | 1.60 | 2.26 | 1.61 |
| 915 | 1.61 | 1.62 | 1.96 | 0.92 | 0.69 | 0.92 | 0.69 |
| xld | 4.19 | 4.19 | 4.96 | 2.51 | 1.68 | 2.50 | 1.68 |
| fluvastatin | 3.90 | 3.92 | 5.00 | 2.34 | 1.56 | 2.36 | 1.56 |
| z34 | 5.06 | 5.01 | 5.98 | 3.04 | 1.99 | 3.01 | 2.00 |
| amprenavir | 5.17 | 5.12 | 7.66 | 3.12 | 2.04 | 3.11 | 2.02 |
| lopinavir | 9.89 | 9.87 | 21.24 | 6.09 | 3.81 | 6.07 | 3.79 |
| lorlatinib | 3.47 | 3.48 | 7.23 | 2.01 | 1.45 | 2.02 | 1.45 |

## Pins

On 93c14173 and on this series' head, with the reheat off:
`scripts/covalent_h2h/pins.sh` prints sha1
`a22093e707faa91d683813fa7cc507706c5265e6` on both, and `h2h_metal run
au55` on seeds 2000-2002, clock columns removed, prints sha1
`b22e212181eff57e9412f780c40e85208b355d07` on both. With ChemFit 4.0.0 and
array_api_strict, `pytest pytest/test_chemfit*.py` gives 1005 passed and 4
skipped, and `examples/chemfit_anneal.py` prints best loss
2.347665788572983.
