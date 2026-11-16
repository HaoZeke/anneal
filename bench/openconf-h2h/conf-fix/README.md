Copyright (c) anneal developers

The paired run of `../paired` with a third arm, on the same eight
molecules, seeds 2000-2011 and MMFF94 through RDKit 2025.09.6 at a
dielectric of one. One serial session under `sudo unshare --uts`, hostname
bench, every job on core 0, `snakemake -s workflow/Snakefile
openconf_h2h_conf_fix`.

- branch: anneal b09285bb1160222e689a81ac7c014efdaf9ed46e, the `anneal`
  rows: the declared-start bond tolerance fix and ring crankshafts, on in
  `recommended_molecular`.
- develop: anneal 92bc5805b476b529c90ab3b2c1db4b48c97ab93a, the `develop`
  rows.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

Both anneal arms are `cluster_search` on the default preset, built with
rustc 1.88.0, and get the energy plus gradient calls openconf spent on the
seed, halved per point. Their order within a pair is drawn from the
schedule seed. A hit is a reported conformer that polishes to the lowest
conformer of the union of all three arms (heavy-atom RMSD under 0.25 Å,
energy within 0.5 kcal/mol); a conformer whose stereocentres or double
bonds differ from the SMILES is invalid and dropped. ΔE_min is the median
of each run's lowest valid polished energy above that minimum, in
kcal/mol. `stats.csv` tests the branch against openconf, `stats-develop.csv`
against develop: two-sided Fisher exact on hits, Mann-Whitney U on calls to
hit (normal approximation with the tie and continuity corrections), no
correction for eight tests.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 12/12 | 12/12 | 10/12 | 1.0000 | 0.4783 | 41 | 41 | 22 | 0.00 | 0.00 | 0.00 |
| 915 | 2/12 | 2/12 | 6/12 | 1.0000 | 0.1930 | 1.5 | 0 | 2.5 | 0.87 | 6.65 | 0.00 |
| xld | 5/12 | 8/12 | 1/12 | 0.4136 | 0.1550 | 18.5 | 15 | 2.5 | 0.00 | 0.00 | 2.70 |
| fluvastatin | 3/12 | 3/12 | 6/12 | 1.0000 | 0.4003 | 25.5 | 25.5 | 7 | 0.40 | 0.40 | 0.39 |
| z34 | 0/12 | 1/12 | 0/12 | 1.0000 | 1.0000 | 15 | 18 | 2 | 0.76 | 0.54 | 2.19 |
| amprenavir | 1/12 | 0/12 | 0/12 | 1.0000 | 1.0000 | 0 | 0 | 0 | 3.41 | 3.70 | 3.76 |
| lopinavir | 2/12 | 0/12 | 0/12 | 0.4783 | 0.4783 | 0 | 0 | 0 | 4.89 | 5.01 | 6.13 |
| lorlatinib | 8/12 | 0/12 | 3/12 | 0.0013 | 0.0995 | 3 | 0 | 2 | 0.00 | 10.63 | 2.68 |

The branch hits 33 of 96 runs, develop 26, openconf 26. rq3 and
fluvastatin have no ring a crankshaft turns, and their branch and develop
rows are the same runs. On 915 the branch leaves the start's twist-boat
(ΔE_min 6.65 to 0.87 kcal/mol) but hits no more often. No molecule is
significantly worse on the branch than on develop; xld goes from 8 to 5
hits (p 0.41) and z34 from 1 to 0.

Develop raises "no finite start after 8 attempts" on lorlatinib seeds 2007
and 2009 and is scored on what it harvested; the branch finishes every run.
98 of the branch's 7724 lorlatinib candidates, all from seed 2000, polish
to the inverted CH(CH3)-O centre and are dropped as invalid; every other
run of every arm has none.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 8266 | 8266 | 1779 | 1.0000 | 0.0698 |
| 915 | 27454.5 | 9617 | 4339 | 0.2453 | 0.0668 |
| xld | 23840 | 14640 | 21918 | 0.7144 | 0.5582 |
| fluvastatin | 23857 | 23857 | 5616 | 1.0000 | 0.3662 |
| z34 | | 10765 | | | |
| amprenavir | 29466 | | | | |
| lopinavir | 73630 | | | | |
| lorlatinib | 39210 | | 7971 | | 0.1258 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.85 | 3.83 | 3.82 | 2.24 | 1.61 | 2.25 | 1.58 |
| 915 | 1.58 | 1.58 | 1.95 | 0.90 | 0.68 | 0.91 | 0.66 |
| xld | 4.12 | 4.08 | 4.87 | 2.46 | 1.66 | 2.46 | 1.61 |
| fluvastatin | 3.92 | 3.92 | 4.89 | 2.36 | 1.56 | 2.38 | 1.54 |
| z34 | 5.10 | 4.98 | 5.98 | 3.06 | 2.04 | 3.00 | 1.98 |
| amprenavir | 5.06 | 5.03 | 7.58 | 3.10 | 1.98 | 3.08 | 1.95 |
| lopinavir | 9.74 | 9.65 | 20.90 | 6.03 | 3.71 | 6.00 | 3.65 |
| lorlatinib | 3.47 | 3.72 | 7.19 | 2.01 | 1.46 | 1.98 | 1.74 |

## Pins

In the same session, on 92bc5805 and on b09285bb:
`scripts/covalent_h2h/pins.sh` (LJ13, LJ38, LJ55 and LJ75 whole-budget
trace hashes, Morse, the Gupta metals and TIP4P) prints sha1
`97677e2820a32d62d21c5b45e3a62138c69ec3ce` on both, and `h2h_metal run
au55` on seeds 2000-2002, clock columns removed, prints sha1
`e16d6ec8a81d50471e0d9f73a8f501a9755aa0b6` on both. With ChemFit 4.0.0 and
array_api_strict, `pytest pytest/test_chemfit*.py` gives 1005 passed and 4
skipped, and `examples/chemfit_anneal.py` prints best loss
2.347665788572983.
