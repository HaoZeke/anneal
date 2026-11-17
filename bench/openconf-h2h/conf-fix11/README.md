Copyright (c) anneal developers

conf-fix11 asks why develop's walk misses the lowest conformer of lopinavir
and lorlatinib, fixes the cause with a search move that applies to every
molecule unchanged, and gates it against same-session develop and openconf.
It is an opt-in record: the gate loses on three molecules.

"wdbase minimum" in the task is read as each molecule's lowest-conformer
reference energy, `e_ref_kcal` of `seeds.csv` and the frame energy of
`references.xyz` (`wdbase` in this repository names only the H2O16 CREST
reference). A development hit is a reported conformer within 0.1 kcal/mol of
it after the harness's polish.

- branch: anneal a72482d6 plus the harness commits up to aab8db16, the
  `anneal` rows: `torsion_walk` on, the rest of `recommended_molecular`.
- develop: anneal 0ae5bfe0efb874a718f8e9632d18ff5fbc29f782, the `develop`
  rows.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`.

## Diagnosis (development seeds 3000-3119, develop, `scripts/openconf_h2h/walk_trace.py`)

120 develop searches of each molecule with the hop trace on
(`diagnosis.csv`, arm `develop`). A hit is 7 of 120 runs on lopinavir and 47
of 120 on lorlatinib.

| | lopinavir | lorlatinib |
|---|---|---|
| quenched-hop acceptance, run / first third / last third | 0.52 / 0.65 / 0.45 | 0.53 / 0.60 / 0.51 |
| hops that change no torsion by more than 45 degrees or ring arc by 30 (share, acceptance) | 0.35, 0.78 | 0.91, 0.58 |
| hops that change one / two / three or more (acceptance) | 0.43 / 0.33 / 0.28 | 0.13 for one, none more |
| hops that change a ring arc (share, acceptance) | 0.04, 0.49 | 0.09, 0.13 |
| quenches whose largest gradient exceeds 1e-3 after 1000 steps | 0.98 | 0.99 |
| polish below the best harvested minimum, misses / hits (kcal/mol) | 0.77 / 1.76 | 0.29 / 0.62 |
| hops that quench to the walker's own energy (within 0.02 kcal/mol) | 0.011 | 0.129 |
| torsions the best minimum of a miss has wrong (mean) | 6.2 | 0 |
| ring arcs it has wrong (mean) | 0.33 | 0.60 |
| gap of the misses, quartiles (kcal/mol) | 4.3 / 6.1 / 8.2 | 1.2 / 2.7 / 2.9 |
| misses within 3 kcal/mol of the reference | 18 of 113 | 58 of 73 |

- Acceptance does not collapse. It sits near one half in both, and falls
  from 0.65 to 0.45 on lopinavir and from 0.60 to 0.51 on lorlatinib from
  the first to the last third of the run, as a walk that has found low
  conformers should.
- Torsions are not coupled in the way a pair table would find. Among the
  torsions a lopinavir miss has wrong, 0.51 of the pairs sit within two bonds
  of each other, against 0.41 of all pairs. Lorlatinib has no rotatable bond
  the reference differs in; its misses have a ring arc wrong.
- Lorlatinib's cost is the macrocycle. Nine hops in ten change no torsion
  or ring arc past the thresholds, and are accepted 0.58 of the time as
  no-ops; the tenth changes a ring arc and is accepted once in eight. The
  lowest conformers differ in the macrocycle, and the walk gets there
  rarely: 0.60 wrong arcs in a miss against none in a hit.
- Lopinavir's cost is the size of the space. A third of the hops turn
  nothing that matters, and acceptance falls as more torsions change (0.78,
  0.43, 0.33, 0.28), so the walk spends its calls on moves that either do
  nothing or are refused. The misses end a median 6 kcal/mol above the
  reference with six torsions wrong.
- The quench is not converged (98% of full quenches stop above a gradient
  of 1e-3 at 1000 steps) and the harness's polish lowers the best minimum by
  0.8 kcal/mol in a lopinavir miss and by 1.8 in a hit. A hit is partly the
  polish; it does not explain the misses, whose gaps are far above it.

Raising or lowering the walk's temperature is not the cause. Scaling the
temperature by 0.4, 1 and 2.5 gave 0, 2 and 1 hits in 40 lopinavir
seeds, and by 1, 4 and 16 gave 15, 19 and 16 in 40 lorlatinib seeds
(`variants.csv`, arms `temperature-x...`; a temporary hook, not committed).

## Mechanism: `torsion_walk`

Config `torsion_walk` (default off, `torsion_walk=True` in Python) lets the
walk learn which torsions are worth turning from its own accepted moves.

- Per torsion, a decayed count of its proposals, the force calls they cost,
  and the accepted hops that left it more than a third of a turn from the
  walker's (a dihedral measured about the torsion's own axis).
- The first torsion of a hop is drawn from its uniform share scaled by
  the square root of its yield per call over the walk's. A torsion's yield
  starts from its class's (rotatable bond or ring arc), so a torsion the walk
  has not tried takes its class's record. A fifth of the draws stay uniform.
- A torsion is named by the atoms at the ends of its axis and its class,
  not by an index: the ring arcs a geometry offers change with the
  geometry.
- Memory is 1000 hops, long enough for a ring arc, which yields roughly once
  in two hundred and fifty hops on lorlatinib, to be seen.
- Until a hop yields the draw and the random stream are the uniform one's.
  Only a lone molecule's torsion hop uses it.

No molecule-specific rule or constant enters: the memory, the exponent and
the uniform share are fixed once and apply to every molecule.

## Development seeds (3000-3059 for six molecules, 3000-3119 for lopinavir and lorlatinib)

`diagnosis.csv`, arms `develop` and `torsion_walk`. Hits are within 0.1
kcal/mol of the reference.

| molecule | seeds | hits develop | hits walk | Fisher p | gap develop mean / median | gap walk mean / median | Mann-Whitney p (gap) | accept develop | accept walk |
|---|---|---|---|---|---|---|---|---|---|
| rq3 | 3000-3059 | 60/60 | 60/60 | 1.0000 | 0.00 / 0.00 | 0.00 / 0.00 | 0.3018 | 0.538 | 0.533 |
| 915 | 3000-3059 | 45/60 | 43/60 | 0.8367 | 0.48 / 0.00 | 0.52 / 0.00 | 0.7601 | 0.540 | 0.536 |
| xld | 3000-3059 | 45/60 | 51/60 | 0.2536 | 0.23 / 0.00 | 0.23 / 0.00 | 0.3228 | 0.597 | 0.521 |
| fluvastatin | 3000-3059 | 26/60 | 27/60 | 1.0000 | 0.20 / 0.22 | 0.21 / 0.21 | 0.5967 | 0.539 | 0.539 |
| z34 | 3000-3059 | 2/60 | 0/60 | 0.4958 | 1.36 / 1.06 | 1.25 / 0.92 | 0.2126 | 0.438 | 0.369 |
| amprenavir | 3000-3059 | 4/60 | 7/60 | 0.5289 | 1.80 / 1.15 | 1.83 / 1.26 | 0.9748 | 0.548 | 0.552 |
| lopinavir | 3000-3119 | 7/120 | 8/120 | 1.0000 | 5.67 / 5.84 | 5.67 / 5.82 | 0.9444 | 0.524 | 0.523 |
| lorlatinib | 3000-3119 | 47/120 | 66/120 | 0.0197 | 2.17 / 1.21 | 1.72 / 0.00 | 0.0344 | 0.534 | 0.544 |

Hits 236 for develop and 262 for the walk. The development gain is
lorlatinib's (+19 hits, p 0.02); lopinavir moves one hit.

Variants tried on lopinavir and lorlatinib at 120 seeds (`variants.csv`, one
arm name per variant; only the memory-1000 variant is committed):

| variant | lopinavir hits | lorlatinib hits |
|---|---|---|
| develop | 7 | 47 |
| coupled partner draw from a table of torsions turned together | 4 | 47 |
| yield per call, torsions named by index, memory 200 | 4 | 55 |
| yield per call, torsions named by axis, memory 200 | 6 | 53 |
| as above, memory 1000 (this branch) | 8 | 66 |
| memory without decay | 5 | 64 |
| memory 1000, exponent 1.0 | 3 | 61 |
| memory 1000, uniform share 0.1 | 5 | 57 |
| memory 1000, exponent 1.0, uniform share 0.1 | 3 | 53 |

The coupled draw, the first idea (a partner torsion from the walk's accepted
covariance), did nothing: the misses are not about which torsions turn
together. Naming torsions by index also lost most of the learning, because
the list of ring arcs changes with the geometry; the table reset whenever it
did. Lopinavir's counts are single digits and cannot rank the later
variants; they were fixed on lorlatinib's, and the exponent and uniform share
were not tuned further.

## Gate (fresh seeds 2048-2071, all 8 molecules)

One serial bench session on host `bench`, core 0:

`snakemake -s workflow/Snakefile openconf_h2h_conf_fix11`

Branch against develop and openconf, 24 seeds per molecule, 576 runs.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 21/24 | 1.0000 | 0.2340 | 46 | 45 | 22 | 0.00 | 0.00 | 0.00 |
| 915 | 19/24 | 15/24 | 14/24 | 0.3412 | 0.2124 | 4 | 3 | 2.5 | 0.00 | 0.00 | 0.01 |
| xld | 22/24 | 21/24 | 1/24 | 1.0000 | 0.0000 | 42 | 35 | 4 | 0.00 | 0.00 | 2.26 |
| fluvastatin | 9/24 | 11/24 | 2/24 | 0.7702 | 0.0363 | 40 | 39 | 7.5 | 0.05 | 0.13 | 0.75 |
| z34 | 1/24 | 2/24 | 0/24 | 1.0000 | 1.0000 | 26.5 | 23 | 3.5 | 0.53 | 0.53 | 2.07 |
| amprenavir | 3/24 | 3/24 | 0/24 | 1.0000 | 0.2340 | 6 | 5 | 0 | 0.15 | 1.22 | 4.74 |
| lopinavir | 2/24 | 4/24 | 0/24 | 0.6662 | 0.4894 | 0 | 0 | 0 | 5.03 | 3.77 | 8.93 |
| lorlatinib | 12/24 | 11/24 | 4/24 | 1.0000 | 0.0305 | 2 | 2 | 2 | 0.00 | 1.21 | 2.68 |

Branch totals: 92/192, develop 91/192, openconf 42/192.

Seed flips branch vs develop:

- 915: +4 / -0
- xld: +3 / -2
- fluvastatin: +4 / -6
- z34: +0 / -1
- amprenavir: +3 / -3
- lopinavir: +2 / -4
- lorlatinib: +3 / -2
- rq3: no change

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 8702 | 8549 | 3156 | 0.7649 | 0.0057 |
| 915 | 10343 | 16888 | 1431.5 | 0.0925 | 0.0048 |
| xld | 22730 | 21806 | 1393 | 0.9709 | 0.1134 |
| fluvastatin | 27831 | 24697 | 10543.5 | 0.8792 | 0.4094 |
| z34 | 59240 | 23265 |  | 0.5403 |  |
| amprenavir | 39851 | 41826 |  | 1.0000 |  |
| lopinavir | 56673.5 | 63083 |  | 0.8170 |  |
| lorlatinib | 31194.5 | 17983 | 8024 | 0.6664 | 0.1631 |

Mann-Whitney uses SciPy's asymptotic method with continuity correction
(tie-corrected).

Median wall time and inside/outside the potential stay within a few percent
of develop's:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.89 | 3.84 | 3.76 | 2.22 | 1.67 | 2.22 | 1.62 |
| 915 | 1.64 | 1.63 | 1.96 | 0.91 | 0.73 | 0.91 | 0.72 |
| xld | 4.36 | 4.31 | 4.97 | 2.57 | 1.80 | 2.55 | 1.77 |
| fluvastatin | 4.10 | 4.06 | 4.95 | 2.40 | 1.70 | 2.40 | 1.66 |
| z34 | 5.17 | 5.04 | 6.01 | 3.04 | 2.13 | 2.98 | 2.06 |
| amprenavir | 5.12 | 5.07 | 7.58 | 3.01 | 2.11 | 2.99 | 2.08 |
| lopinavir | 10.22 | 10.18 | 20.96 | 6.06 | 4.18 | 6.07 | 4.10 |
| lorlatinib | 3.44 | 3.41 | 7.04 | 1.96 | 1.48 | 1.95 | 1.45 |

One lorlatinib candidate in each anneal arm is invalid (`invalid_total` of
`summary.csv`); the rows drop it.

## Decision

The gate loses on three molecules: fluvastatin (9 against 11), lopinavir (2
against 4) and z34 (1 against 2). The rule was default-on only if it loses
on none, so `torsion_walk` stays off in `recommended_molecular`
and is an opt-in record. The gate's gain (915 +4, lorlatinib +1) is
smaller than the development gain on lorlatinib (+19), and the lorlatinib
result inside the gate is not significant (p 1.0). On lopinavir, the molecule
that started the task, the walk does nothing measurable: its misses are
six torsions wrong in a conformer space the yield table does not narrow. A move that crosses a ring or a macrocycle together,
or a change of the cost per hop, is what remains.

Files: `seeds.csv`, `summary.csv`, `stats.csv` and `stats-develop.csv` are
the gate; `diagnosis.csv` and `variants.csv` are the development-seed runs;
`references.xyz` holds the reference energies the diagnosis scores against.
