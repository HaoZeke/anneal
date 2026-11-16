Copyright (c) anneal developers

Coupled torsion draws: a general search move for the openconf misses whose
lowest conformer differs from the chain's in several torsions that only
pay when they turn together (`../conf-fix6`, on `record/conf-fix6`). The
move knows nothing of the molecule, of hydrogen bonds or of the potential;
it reads only which rotatable bonds a chain's accepted hops turned.

- branch: anneal 9392b44efc995dff961f8db1e01de9432c5f236d, the `anneal`
  and `coupled` rows: `torsion_coupling`, on in `recommended_molecular`.
- develop: anneal fb178f4023d2da4e5fc9060985a5c8cf5c58bb37 (0.12.0), the
  `develop` rows.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

## The move

A plain torsion draw turns one to three rotatable bonds through uniform
angles. A minimum several coupled torsions away is out of its reach, and a
turn of any one of those torsions alone quenches uphill and is refused.

With `torsion_coupling` each chain keeps, for the last 32 accepted hops of
the torsion arm, the rotatable bonds whose dihedral moved by 30° or more
between the two minima. Once one of those hops turned two bonds or more,
a quarter of the bond draws pick one such hop and turn every bond it
turned, each through a fresh uniform angle, with the clash repair and the
stereo check of the plain draw. The ring crankshaft is decided first and
per redraw, as in the plain draw, so ring puckers keep their share. A
chain with no coupled hop on file draws the plain move, random stream
included. Rejected hops, hops that quench back unchanged and hops of other
arms record nothing.

The history is indexed by bond, not by position in the bond list: a
conjugated bond near the single-bond cut is rotatable in one minimum and
not the next (915's bond 9-11), and keying by position cleared the history
on nearly every hop.

## Development seeds

Seeds 3000-3119, not the evaluation seeds 2000-2023, each run scored
against the global reference of `../conf-fix6` by its
`dev_diagnose.py analyse`. A hit is a valid polished minimum within
0.1 kcal/mol of the reference. `dev-rows.csv` has one row per (molecule,
arm, seed), in the columns of `../conf-fix6/dev-rows.csv`. The dev runs
ran several at a time, so their times are not compared; the gate's are.

The variants tried on seeds 3000-3059, hits over the eight molecules
(480 runs). Every variant except the shipped one was a temporary build
that is not part of the branch:

| variant | z34 | amprenavir | lopinavir | 915 | fluvastatin | xld | lorlatinib | rq3 | total |
|---|---|---|---|---|---|---|---|---|---|
| develop | 3 | 5 | 0 | 45 | 24 | 51 | 34 | 60 | 222 |
| replay the accepted angles, share adapted | | | | | | | | | 201 |
| fresh uniform angles, share adapted | | | | | | | | | 209 |
| replay with jitter | | | | | | | | | 214 |
| crankshaft first, once per proposal | 6 | 8 | 2 | 42 | 27 | 54 | 23 | 60 | 222 |
| crankshaft first, per redraw, share adapted | 5 | 6 | 3 | 41 | 27 | 48 | 34 | 60 | 224 |
| as above, sets of three bonds or more | | | | | | | | | 222 |
| as above, success counted only for a new minimum | | | | | | | | | 215 |
| as above, adapted share halved | 3 | 1 | 4 | 43 | 27 | 54 | 34 | 60 | 226 |
| fixed quarter share, sign drawn | 0 | 10 | 2 | 45 | 30 | 52 | 34 | 60 | 233 |
| fixed quarter share (shipped) | 4 | 9 | 1 | 47 | 26 | 52 | 34 | 60 | 233 |

Replaying an accepted change sends the chain back to the minima it came
from. A coupled draw that replaces the crankshaft costs 915 its ring
puckers (45 to 28 hits), and a crankshaft decided once per proposal rather
than per redraw costs lorlatinib (34 to 23). Shares adapted from how often
each kind of draw moved the chain, the acceptance-rate rule, settled near
one half and lost hits on 915 and xld; a fixed quarter did better on both
halves of the seeds.

The shipped move on all 120 dev seeds:

| molecule | hits develop | hits coupled | Fisher p |
|---|---|---|---|
| z34 | 6/120 | 7/120 | 1.0000 |
| amprenavir | 8/120 | 9/120 | 1.0000 |
| lopinavir | 1/120 | 2/120 | 1.0000 |
| 915 | 81/120 | 88/120 | 0.3962 |
| fluvastatin | 46/120 | 54/120 | 0.3594 |
| xld | 93/120 | 108/120 | 0.0136 |
| lorlatinib | 56/120 | 56/120 | 1.0000 |
| rq3 | 120/120 | 119/120 | 1.0000 |

443 hits against 411 over 960 runs (Fisher p 0.15). Lorlatinib's runs are
the same as develop's: its torsion hops never turn two bonds at once, so
it never has a coupled set and draws the plain move throughout. The best
energy a lopinavir run reaches falls, median 6.49 against 7.25 kcal/mol
above the reference on seeds 3000-3059.

## Gate

The three-arm paired run of `../conf-fix4`, all eight molecules on seeds
2000-2023, MMFF94 through RDKit 2025.09.6 at a dielectric of one, in one
serial session under `sudo unshare --uts`, hostname bench, every job on
core 0: `snakemake -s workflow/Snakefile openconf_h2h_conf_fix7`. Budgets,
arm order, hits, ΔE_min and the tests are those of `../conf-fix`. The
develop rows repeat `../conf-fix4/n24`'s branch rows hit for hit, as they
should: the same code at the same seeds.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf | Mann-Whitney p ΔE_min vs develop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 20/24 | 1.0000 | 0.1092 | 45.5 | 45 | 22.5 | 0.00 | 0.00 | 0.00 | 1.0000 |
| 915 | 18/24 | 15/24 | 10/24 | 0.5343 | 0.0392 | 4 | 4 | 2 | 0.00 | 0.00 | 0.01 | 0.2900 |
| xld | 21/24 | 16/24 | 2/24 | 0.1681 | 0.0000 | 38 | 33 | 4 | 0.00 | 0.00 | 2.74 | 0.1616 |
| fluvastatin | 14/24 | 10/24 | 8/24 | 0.3868 | 0.1468 | 40.5 | 39.5 | 7 | 0.03 | 0.05 | 0.48 | 0.1585 |
| z34 | 2/24 | 0/24 | 0/24 | 0.4894 | 0.4894 | 22.5 | 24 | 1.5 | 0.53 | 0.53 | 2.38 | 0.9570 |
| amprenavir | 2/24 | 3/24 | 0/24 | 1.0000 | 0.4894 | 4 | 3 | 0 | 1.52 | 2.08 | 4.19 | 0.3473 |
| lopinavir | 0/24 | 2/24 | 0/24 | 0.4894 | 1.0000 | 0 | 0 | 0 | 4.57 | 5.89 | 9.29 | 0.2882 |
| lorlatinib | 12/24 | 12/24 | 5/24 | 1.0000 | 0.0687 | 3 | 3 | 2 | 0.60 | 0.60 | 2.68 | 1.0000 |

The branch hits 93 of 192 runs, develop 82 and openconf 45 (Fisher p 0.31
vs develop over the eight). z34 gains its first two hits; no molecule
moves significantly against develop. Lopinavir's two develop hits are lost,
but its median ΔE_min falls from 5.89 to 4.57 kcal/mol, and amprenavir's
from 2.08 to 1.52. No arm has an invalid candidate or an error.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 7526 | 12613 | 4165 | 0.2440 | 0.0610 |
| 915 | 23489.5 | 18883 | 4339 | 0.8142 | 0.0012 |
| xld | 19029 | 36271.5 | 10363 | 0.0554 | 0.5692 |
| fluvastatin | 21820 | 13292 | 3523.5 | 0.5387 | 0.0421 |
| z34 | 18271.5 | | | | |
| amprenavir | 27980 | 56050 | | 0.2000 | |
| lopinavir | | 48892.5 | | | |
| lorlatinib | 55483.5 | 55483.5 | 7971 | 1.0000 | 0.0013 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.95 | 3.91 | 3.82 | 2.26 | 1.69 | 2.26 | 1.65 |
| 915 | 1.67 | 1.65 | 1.96 | 0.93 | 0.73 | 0.92 | 0.73 |
| xld | 4.29 | 4.27 | 4.94 | 2.50 | 1.79 | 2.51 | 1.76 |
| fluvastatin | 3.95 | 4.02 | 4.91 | 2.34 | 1.63 | 2.36 | 1.65 |
| z34 | 5.17 | 5.07 | 6.00 | 3.04 | 2.12 | 3.00 | 2.07 |
| amprenavir | 5.30 | 5.26 | 7.75 | 3.13 | 2.18 | 3.10 | 2.16 |
| lopinavir | 10.27 | 10.60 | 21.36 | 6.10 | 4.15 | 6.24 | 4.23 |
| lorlatinib | 3.57 | 3.53 | 7.25 | 2.01 | 1.54 | 2.02 | 1.51 |

The move costs at most 0.05 s per run outside the potential; inside, the
call counts match and so does the time.

LJ, Morse and reduced-unit pins (`scripts/covalent_h2h/pins.sh`) and the
Gupta Au55 rows are bit-identical between develop and the branch: a run
without torsion groups never builds the history.

## Decision

The gate passes: z34 gains (2 against 0) and the eight molecules together
gain 11 hits (93 against 82), with no significant loss on any molecule; the
dev seeds agree (443 against 411). `torsion_coupling` stays on in
`recommended_molecular`. The gain on the three weak molecules is small:
their misses are mostly runs that never visit the lowest conformer, and a
chain only learns couplings it has already crossed.
