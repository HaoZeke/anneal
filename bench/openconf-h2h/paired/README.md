Copyright (c) anneal developers

anneal against openconf on the eight molecules of `../molecules.csv`,
seeds 2000-2011, MMFF94 through RDKit 2025.09.6 at a dielectric of one for
both codes. One serial session under `sudo unshare --uts`, hostname bench,
every job on core 0, `snakemake -s workflow/Snakefile openconf_h2h_n12`.

openconf is rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
(tag v0.0.19, the same files as the 0.0.19 wheel), preset spectroscopic,
macrocycle for lorlatinib, one thread, every MMFF call pinned to MMFF94.
anneal is `cluster_search` on its default (recommended) preset with the
rigid groups between rotatable bonds, built from
d178bd157ad5e35aec56dfe227b4bd3badeffe30 with rustc 1.88.0; the library
is develop 5d44d218366f5c75a3ebe3bdef786e5010a26f45.

A call is one outermost RDKit force-field energy or gradient call, counted
by `scripts/openconf_h2h/rdff_count.c`; ETKDG embedding is not counted.
openconf runs first on each seed and anneal may then make as many calls.
anneal spent no more calls than openconf on any seed.

A hit is a reported conformer that, polished on MMFF94, is the lowest
conformer of the union of both arms (heavy-atom RMSD under 0.25 Å, energy
within 0.5 kcal/mol). `n_3kcal` counts distinct conformers within
3 kcal/mol of that minimum. Calls to hit are the calls spent when a run's
trace first held a minimum that polishes to it; medians are over hitting
runs. `seeds.csv` holds one row per run, `summary.csv` the medians and
`stats.csv` the two-sided Fisher exact test on hits and Mann-Whitney U
test on calls to hit (normal approximation with the tie and continuity
corrections). No correction for eight tests is applied.

| molecule | hits anneal | hits openconf | Fisher p | n_3kcal anneal | n_3kcal openconf | calls to hit anneal | calls to hit openconf | Mann-Whitney p |
|---|---|---|---|---|---|---|---|---|
| rq3 | 12/12 | 10/12 | 0.4783 | 41 | 22 | 8266 | 1779 | 0.0698 |
| 915 | 2/12 | 6/12 | 0.1930 | 0 | 2.5 | 9617 | 4339 | 0.6171 |
| xld | 8/12 | 1/12 | 0.0094 | 15 | 2.5 | 14640 | 21918 | 0.8465 |
| fluvastatin | 3/12 | 6/12 | 0.4003 | 25.5 | 7 | 23857 | 5616 | 0.3662 |
| z34 | 1/12 | 0/12 | 1.0000 | 18 | 2 | 10765 | | |
| amprenavir | 0/12 | 1/12 | 1.0000 | 0 | 0 | | 18311 | |
| lopinavir | 1/12 | 0/12 | 1.0000 | 1 | 0 | 57687 | | |
| lorlatinib | 0/12 | 3/12 | 0.2174 | 0 | 2 | | 7971 | |

Both codes hit 27 of 96 runs.

On lorlatinib seeds 2007 and 2009 anneal raised "no finite start after 8
attempts" after three calls from a finite ETKDG start; both are scored as
misses. Its median lowest energy on lorlatinib is 10.63 kcal/mol above the
reference, openconf's 2.68.

Median wall time per run, and anneal's median time inside and outside the
potential:

| molecule | wall anneal (s) | wall openconf (s) | anneal inside (s) | anneal outside (s) |
|---|---|---|---|---|
| rq3 | 3.83 | 3.81 | 2.24 | 1.58 |
| 915 | 1.57 | 1.97 | 0.91 | 0.66 |
| xld | 4.08 | 4.90 | 2.47 | 1.62 |
| fluvastatin | 3.91 | 4.94 | 2.38 | 1.53 |
| z34 | 4.98 | 6.03 | 2.99 | 1.98 |
| amprenavir | 5.10 | 7.69 | 3.12 | 1.98 |
| lopinavir | 9.66 | 21.03 | 5.99 | 3.67 |
| lorlatinib | 3.78 | 7.28 | 2.01 | 1.77 |

Over all runs of a molecule anneal spends 53-62 % of its wall time inside
the potential, openconf 25-52 %.
