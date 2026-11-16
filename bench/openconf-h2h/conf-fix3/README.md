Copyright (c) anneal developers

The three-arm paired run of `../conf-fix` again, on the same eight
molecules, seeds 2000-2011 and MMFF94 through RDKit 2025.09.6 at a
dielectric of one, and xld alone on seeds 2000-2023 (`xld24/`). One serial
session under `sudo unshare --uts`, hostname bench, every job on core 0,
`snakemake -s workflow/Snakefile openconf_h2h_conf_fix3`.

- branch: anneal cf660e192df3d8e220e2e15a8089c2fa1f30d3d3, the `anneal`
  rows: the quench holds stereocentres, on in `recommended_molecular`.
- develop: anneal 7d8fa6ea6ec819a197b5390d87abab28423fd967, the `develop`
  rows. It carries the conf-fix ring crankshafts.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

Budgets, arm order, hits, ΔE_min and the tests are those of `../conf-fix`:
a hit is a reported conformer that polishes to the lowest valid conformer
of the session's three arms, and a conformer whose stereocentres or double
bonds differ from the SMILES is invalid and dropped.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 12/12 | 12/12 | 10/12 | 1.0000 | 0.4783 | 41 | 41 | 22 | 0.00 | 0.00 | 0.00 |
| 915 | 2/12 | 2/12 | 6/12 | 1.0000 | 0.1930 | 1.5 | 1.5 | 2.5 | 0.87 | 0.87 | 0.00 |
| xld | 8/12 | 8/12 | 1/12 | 1.0000 | 0.0094 | 18.5 | 18.5 | 2.5 | 0.00 | 0.00 | 2.70 |
| fluvastatin | 3/12 | 3/12 | 6/12 | 1.0000 | 0.4003 | 25.5 | 25.5 | 7 | 0.40 | 0.40 | 0.39 |
| z34 | 1/12 | 1/12 | 0/12 | 1.0000 | 1.0000 | 15.5 | 15.5 | 2 | 0.67 | 0.67 | 2.09 |
| amprenavir | 1/12 | 1/12 | 0/12 | 1.0000 | 1.0000 | 0 | 0 | 0 | 3.41 | 3.41 | 3.76 |
| lopinavir | 2/12 | 2/12 | 0/12 | 1.0000 | 0.4783 | 0 | 0 | 0 | 4.89 | 4.89 | 6.13 |
| lorlatinib | 8/12 | 8/12 | 3/12 | 1.0000 | 0.0995 | 3 | 3 | 2 | 0.00 | 0.00 | 2.68 |

The branch and develop each hit 37 of 96 runs, openconf 26. 95 of the 96
branch runs are develop's run call for call. The one that is not is
lorlatinib seed 2000, where a crankshaft leaves the CH(CH3)-O carbon
nearly planar and the quench used to descend through it into the mirror
well: develop reports 591 candidates there, 98 of them inverted and
dropped, and the branch 567 with none inverted, at the same ΔE_min (1.21
kcal/mol). No run of the branch has an invalid candidate.

xld at 24 seeds: branch 16/24, develop 16/24 (Fisher p 1.0000), openconf
2/24 (p 0.0001 against the branch); the branch and develop runs are the
same.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 8266 | 8266 | 1779 | 1.0000 | 0.0698 |
| 915 | 27454.5 | 27454.5 | 4339 | 1.0000 | 0.0668 |
| xld | 35688.5 | 35688.5 | 13369 | 1.0000 | 0.5613 |
| fluvastatin | 23857 | 23857 | 5616 | 1.0000 | 0.3662 |
| z34 | 14961 | 14961 |  |  |  |
| amprenavir | 29466 | 29466 |  |  |  |
| lopinavir | 73630 | 73630 |  | 1.0000 |  |
| lorlatinib | 39210 | 39210 | 7971 | 1.0000 | 0.1258 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.87 | 3.86 | 3.81 | 2.25 | 1.62 | 2.25 | 1.62 |
| 915 | 1.59 | 1.59 | 1.96 | 0.91 | 0.68 | 0.91 | 0.68 |
| xld | 4.17 | 4.14 | 4.86 | 2.50 | 1.67 | 2.46 | 1.68 |
| fluvastatin | 4.12 | 3.91 | 4.89 | 2.47 | 1.58 | 2.35 | 1.56 |
| z34 | 5.07 | 5.04 | 6.00 | 3.04 | 2.03 | 3.03 | 2.00 |
| amprenavir | 5.10 | 5.07 | 7.62 | 3.09 | 2.01 | 3.08 | 1.99 |
| lopinavir | 9.77 | 9.65 | 21.09 | 6.03 | 3.74 | 5.97 | 3.68 |
| lorlatinib | 3.55 | 3.56 | 7.25 | 2.04 | 1.49 | 2.04 | 1.49 |

## Pins

In the same session, on 7d8fa6ea and on cf660e19:
`scripts/covalent_h2h/pins.sh` prints sha1
`a22093e707faa91d683813fa7cc507706c5265e6` on both, and `h2h_metal run
au55` on seeds 2000-2002, clock columns removed, prints sha1
`b22e212181eff57e9412f780c40e85208b355d07` on both. With ChemFit 4.0.0 and
array_api_strict, `pytest pytest/test_chemfit*.py` gives 1005 passed and 4
skipped, and `examples/chemfit_anneal.py` prints best loss
2.347665788572983.
