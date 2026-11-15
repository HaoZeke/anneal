Copyright (c) anneal developers

The three-arm paired run of `../conf-fix` again, on the same eight
molecules, seeds 2000-2011 and MMFF94 through RDKit 2025.09.6 at a
dielectric of one, and the six molecules whose hits moved again on seeds
2000-2023 (`n24/`). One serial session under `sudo unshare --uts`,
hostname bench, every job on core 0, `snakemake -s workflow/Snakefile
openconf_h2h_conf_fix4` and then `openconf_h2h_conf_fix4_n24`.

- branch: anneal 2a5b45534da2e9c4a927fa9a15f80ef364f93b08 (`n24/`:
  26b70c01, which adds only the rule), the `anneal` rows: the stereocentre
  hold and the measured energy scale, both on in `recommended_molecular`.
- develop: anneal 7d8fa6ea6ec819a197b5390d87abab28423fd967, the `develop`
  rows.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

Budgets, arm order, hits, ΔE_min and the tests are those of `../conf-fix`:
a hit is a reported conformer that polishes to the lowest valid conformer
of the session's three arms, and a conformer whose stereocentres or double
bonds differ from the SMILES is invalid and dropped.

## Why fluvastatin was missed

The molecular preset's temperature and deposit height are multiples of
`energy_scale`, which the Python binding declares as one. In electronvolts
that is a Metropolis temperature of 18 kcal/mol and deposits of 5.8
kcal/mol, and the adaptive height starts its estimate at 115 kcal/mol, over
conformers a kcal/mol or two apart. Develop's fluvastatin chain accepts 96
to 98% of its hops: it is a random walk over conformers. Its near misses
are one coupled turn of the two torsions flanking the trans C=C from the
lowest conformer; either turn alone quenches only 3 to 5 kcal/mol up, but
a chain that accepts everything leaves a basin as readily as it reaches
one. With the measured energy scale the median quenched energy change of
the first 20 hops takes the place of `energy_scale` in the temperature and
the deposit height, and the chain accepts about half its hops.

## 12 seeds

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 12/12 | 12/12 | 10/12 | 1.0000 | 0.4783 | 45.5 | 41 | 22 | 0.00 | 0.00 | 0.00 |
| 915 | 8/12 | 2/12 | 6/12 | 0.0361 | 0.6802 | 3.5 | 1.5 | 2.5 | 0.00 | 0.87 | 0.00 |
| xld | 10/12 | 5/12 | 1/12 | 0.0894 | 0.0006 | 29.5 | 18.5 | 2.5 | 0.00 | 0.00 | 2.70 |
| fluvastatin | 6/12 | 3/12 | 6/12 | 0.4003 | 1.0000 | 37 | 25.5 | 7 | 0.12 | 0.40 | 0.39 |
| z34 | 0/12 | 1/12 | 0/12 | 1.0000 | 1.0000 | 28 | 15.5 | 2 | 0.43 | 0.67 | 2.09 |
| amprenavir | 1/12 | 1/12 | 0/12 | 1.0000 | 1.0000 | 2.5 | 0 | 0 | 2.23 | 3.41 | 3.76 |
| lopinavir | 1/12 | 0/12 | 0/12 | 1.0000 | 1.0000 | 0 | 0 | 0 | 7.13 | 8.03 | 9.28 |
| lorlatinib | 6/12 | 8/12 | 3/12 | 0.6802 | 0.4003 | 2.5 | 3 | 2 | 0.60 | 0.00 | 2.68 |

The branch hits 44 of 96 runs, develop 32 and openconf 26. Every anneal
run of the branch differs from develop's. The branch has no invalid
candidate; develop drops 98 inverted ones on lorlatinib seed 2000.

xld has two conformers at its lowest energy more than the RMSD threshold
apart, and which one the session takes as the reference decides which
runs hit. By energy the branch reaches xld's lowest in 12 of 12 runs and
develop in 9 of 12; the same runs score 8 and 8 hits against the
reference `n24/` takes.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 4546 | 8266 | 1779 | 0.5440 | 0.1062 |
| 915 | 9852 | 27454.5 | 4339 | 0.4000 | 0.0200 |
| xld | 34388.5 | 23840 | 21918 | 0.4620 | 0.7273 |
| fluvastatin | 16039 | 23857 | 5616 | 0.7954 | 0.2403 |
| z34 |  | 14961 |  |  |  |
| amprenavir | 57119 | 29466 |  | 1.0000 |  |
| lopinavir | 33079 |  |  |  |  |
| lorlatinib | 63752 | 39210 | 7971 | 0.7546 | 0.0476 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.86 | 3.87 | 3.80 | 2.25 | 1.61 | 2.26 | 1.60 |
| 915 | 1.61 | 1.58 | 1.96 | 0.92 | 0.69 | 0.91 | 0.67 |
| xld | 4.15 | 4.25 | 4.91 | 2.47 | 1.68 | 2.54 | 1.68 |
| fluvastatin | 3.95 | 3.93 | 4.90 | 2.37 | 1.58 | 2.36 | 1.57 |
| z34 | 4.99 | 4.98 | 6.02 | 3.02 | 1.98 | 3.00 | 1.98 |
| amprenavir | 5.07 | 5.11 | 7.70 | 3.07 | 2.00 | 3.11 | 2.00 |
| lopinavir | 9.77 | 9.72 | 21.11 | 6.01 | 3.76 | 6.01 | 3.71 |
| lorlatinib | 3.52 | 3.53 | 7.20 | 2.03 | 1.48 | 2.06 | 1.47 |

## 24 seeds

915, xld, fluvastatin, z34, lopinavir and lorlatinib, whose hits moved
at 12 seeds. The runs on seeds 2000-2011 are those of the 12-seed run.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 915 | 15/24 | 5/24 | 10/24 | 0.0077 | 0.2476 | 4 | 1.5 | 2 | 0.00 | 0.87 | 0.01 |
| xld | 16/24 | 16/24 | 2/24 | 1.0000 | 0.0001 | 33 | 16 | 4 | 0.00 | 0.00 | 2.74 |
| fluvastatin | 10/24 | 6/24 | 8/24 | 0.3587 | 0.7661 | 39.5 | 25.5 | 7 | 0.05 | 0.22 | 0.48 |
| z34 | 0/24 | 1/24 | 0/24 | 1.0000 | 1.0000 | 26.5 | 16 | 1.5 | 0.43 | 0.58 | 2.28 |
| lopinavir | 2/24 | 0/24 | 0/24 | 0.4894 | 0.4894 | 0 | 0 | 0 | 5.89 | 7.87 | 9.29 |
| lorlatinib | 12/24 | 13/24 | 5/24 | 1.0000 | 0.0687 | 3 | 3 | 2 | 0.60 | 0.00 | 2.68 |

915 gains 10 hits (Fisher p 0.0077) and fluvastatin 4, now ahead of
openconf's 8. xld is 16 against 16; by energy the branch reaches its
lowest in 24 of 24 runs and develop in 19 of 24. Lorlatinib is 12 against
13. Its hops are a few tenths of a kcal/mol apart, so its measured scale
is small and the chain cold, and in two runs (seeds 2004 and 2021) it
stays in a macrocycle 11.7 to 11.9 kcal/mol up that develop's hot chain
leaves; its median ΔE_min is 0.60 against 0.00.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| 915 | 18883 | 19397 | 4339 | 0.8001 | 0.0017 |
| xld | 36271.5 | 33966 | 10363 | 0.3558 | 0.0523 |
| fluvastatin | 13292 | 43611 | 3523.5 | 0.1035 | 0.0545 |
| z34 |  | 14961 |  |  |  |
| lopinavir | 48892.5 |  |  |  |  |
| lorlatinib | 55483.5 | 32657 | 7971 | 0.2648 | 0.0013 |

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| 915 | 1.61 | 1.61 | 1.97 | 0.92 | 0.69 | 0.93 | 0.68 |
| xld | 4.18 | 4.17 | 4.92 | 2.54 | 1.68 | 2.51 | 1.67 |
| fluvastatin | 3.92 | 3.99 | 4.88 | 2.34 | 1.57 | 2.41 | 1.56 |
| z34 | 4.97 | 4.99 | 5.99 | 2.98 | 1.98 | 3.01 | 1.98 |
| lopinavir | 9.87 | 9.82 | 21.20 | 6.07 | 3.80 | 6.07 | 3.75 |
| lorlatinib | 3.47 | 3.46 | 7.23 | 2.01 | 1.46 | 2.02 | 1.46 |

## Pins

In the same session, on 7d8fa6ea and on 2a5b4553:
`scripts/covalent_h2h/pins.sh` prints sha1
`a22093e707faa91d683813fa7cc507706c5265e6` on both, and `h2h_metal run
au55` on seeds 2000-2002, clock columns removed, prints sha1
`b22e212181eff57e9412f780c40e85208b355d07` on both. With ChemFit 4.0.0 and
array_api_strict, `pytest pytest/test_chemfit*.py` gives 1005 passed and 4
skipped, and `examples/chemfit_anneal.py` prints best loss
2.347665788572983.
