Copyright (c) anneal developers

Why develop's chains on amprenavir, lopinavir and 915 stay above the
lowest conformer, a stall return built on that reading, and the paired run
that decides whether it goes into the molecular preset. It does not: the
switch stays opt-in. Same eight molecules, MMFF94 through RDKit 2025.09.6
at a dielectric of one, openconf settings of `../paired`.

## Where the chains miss

Hop traces of develop at 027c9c715564a065d02ae566d845c58e7e6da2a5 on
development seeds 3000-3119 at the median budgets of `../conf-fix4`
(`traces-dev.csv`, `scripts/openconf_h2h/elite_trace.py`). A row's
`energy_hit` is set when a valid polished minimum the run harvested lies
within 0.1 kcal/mol of the reference (`references.xyz`, the session
references of the paired run below); the rest of the row describes the
lowest valid polished minimum and the trace.

| molecule | arm | energy hits | lowest gap (kcal/mol) | hops near own best | height above own best (kcal/mol) | height above reference (kcal/mol) | share at last new best |
|---|---|---|---|---|---|---|---|
| 915 | develop | 81/120 | 0.01 | 0.25 | 3.53 | 6.4 | 0.55 |
| 915 | branch | 91/120 | 0.00 | 0.40 | 1.79 | 4.3 | 0.69 |
| amprenavir | develop | 8/120 | 1.97 | 0.13 | 4.61 | 9.7 | 0.53 |
| amprenavir | branch | 18/120 | 1.26 | 0.28 | 2.65 | 7.7 | 0.82 |
| lopinavir | develop | 1/120 | 6.84 | 0.15 | 5.29 | 17.8 | 0.63 |
| lopinavir | branch | 7/120 | 5.86 | 0.28 | 2.83 | 13.8 | 0.87 |

Medians over runs. "Near own best" is the share of hops within 1 kcal/mol
of the run's best so far, the heights are medians over hops, and the last
column is the share of the budget spent when the best last fell by more
than 0.05 kcal/mol. Of the misses:

| molecule | arm | misses | wrong torsions, median | ring wrong | one torsion reaches | all wrong torsions reach | greedy reaches | greedy finds no step | lowest eight hold every torsion |
|---|---|---|---|---|---|---|---|---|---|
| 915 | develop | 39 | 1 | 35 | 3 | 4 | 3 | 25 | 38 |
| 915 | branch | 29 | 1 | 23 | 6 | 6 | 6 | 21 | 25 |
| amprenavir | develop | 112 | 4 | 17 | 7 | 52 | 12 | 63 | 43 |
| amprenavir | branch | 102 | 4 | 14 | 3 | 36 | 3 | 65 | 14 |
| lopinavir | develop | 119 | 7 | 44 | 4 | 55 | 17 | 42 | 14 |
| lopinavir | branch | 113 | 7 | 45 | 4 | 57 | 22 | 44 | 2 |

A torsion is wrong more than 45 degrees from the reference after mapping
atoms over graph automorphisms, a ring when one of its dihedrals is more
than 30 degrees off. "Reaches" means setting those torsions to the
reference and polishing lands within 0.1 kcal/mol of it; greedy repair
sets one torsion at a time, the one that lowers the polished energy most.
The last column counts misses where every reference torsion stands in at
least one of the eight lowest distinct accepted minima.

- 915 misses on ring pucker. 35 of 39 misses have a ring wrong, and in 32
  the only wrong torsion, if any, is torsion 0, the aryl-S bond, whose
  ring flip is close enough to degenerate that 23 of the 81 hits carry it
  too. The missed basins sit a median 1.46 kcal/mol up.
- Amprenavir and lopinavir miss on many rotors at once, not on rings: a
  median 4 and 7 torsions wrong, a ring in 17 and 44 of the misses. One
  torsion repairs 7 of 112 and 4 of 119; setting all of them 52 and 55. In
  more than half of amprenavir's misses greedy repair has no downhill
  step: the best is a local minimum of single-torsion moves.
- The budget goes to the plateau above the best, not to the best's
  neighbourhood. Develop's walker stands within 1 kcal/mol of its own best
  on 13 to 25% of hops and a median 3.5 to 5.3 kcal/mol above it, and still
  finds a new best past half its budget. The molecular runs make 250 to 850
  hops, so `escape_stall_patience` (5000) and every stall response keyed
  to it never fire.

## The return

`Config::elite_return` (`src/methods/stall_archive.rs`,
`DescriptorArchive`). A chain keeps eight seats of distinct minima keyed
by the bias collective variable at the bias merge radius: the start, every
accepted quenched hop, and its best at each stall. After
`elite_patience` hops (40) without a new best it moves to the lowest seat
farther than the merge radius from where it stands and not among its last
four destinations, spends one gradient call there and continues with its
bias. The choice reads no random number, so a chain walks develop's hops
until its first return. The branch rows above are this return at
7fd1ebc87706fb62d299a94754da6d6d61d0a9f7: the walker spends 1.6 to 2.2
times the share of hops near its best at about half the height above it,
keeps improving later, and hits 91, 18 and 7 times against 81, 8 and 1.
Its eight lowest minima are less varied (14 and 2 misses hold every
torsion against 43 and 14), so a crossover of them would have even less to
work with.

Development seeds, all eight molecules, lowest-conformer hits of 120 runs
per molecule (`dev-rows.csv`; develop at 027c9c71, the return at
1a2d81be961afcd24fd23bc6a82e097bae6526c9 through `cluster_search` keywords):

| arm | z34 | amprenavir | lopinavir | 915 | fluvastatin | xld | lorlatinib | rq3 | total |
|---|---|---|---|---|---|---|---|---|---|
| develop | 6 | 8 | 1 | 81 | 46 | 93 | 56 | 120 | 411 |
| elite20 | 11 | 12 | 2 | 81 | 41 | 91 | 53 | 120 | 411 |
| elite40 | 15 | 18 | 7 | 91 | 51 | 101 | 61 | 120 | 464 |
| elite60 | 14 | 7 | 6 | 81 | 45 | 99 | 55 | 120 | 427 |
| elite80 | 8 | 8 | 5 | 88 | 48 | 96 | 56 | 120 | 429 |

Seeds each patience wins and loses against develop:

| arm | z34 | amprenavir | lopinavir | 915 | fluvastatin | xld | lorlatinib | rq3 |
|---|---|---|---|---|---|---|---|---|
| elite20 | +9/-4 | +11/-7 | +2/-1 | +22/-22 | +22/-27 | +21/-23 | +28/-31 | +0/-0 |
| elite40 | +13/-4 | +17/-7 | +7/-1 | +22/-12 | +27/-22 | +19/-11 | +31/-26 | +0/-0 |
| elite60 | +10/-2 | +6/-7 | +5/-0 | +14/-14 | +18/-19 | +19/-13 | +22/-23 | +0/-0 |
| elite80 | +4/-2 | +7/-7 | +4/-0 | +16/-9 | +23/-21 | +21/-18 | +16/-16 | +0/-0 |

Patience 40 was the default before any of these ran and is the best of
the four. The churn is large: a return redraws the rest of the chain, so
a run that develop hits late is as likely to be lost as a miss is to be
won. Split into five blocks of 24 seeds, patience 40 loses on no molecule
in one block, and patience 20, 60 and 80 in zero, one and one. The rows
come from a development harness that makes the same `cluster_search`
call at the same budgets as `elite_trace.py run` and scores hits by
`energy_hit`; on 915, amprenavir and lopinavir the hits of `traces-dev.csv`
match its develop and elite40 rows seed for seed.

## Paired run

One serial session under `sudo unshare --uts`, hostname bench, every job
on core 0, seeds 2000-2023, `snakemake -s workflow/Snakefile
openconf_h2h_conf_fix9`. Budgets, arm order, hits, ΔE_min and the tests are
those of `../conf-fix`: a hit is a reported conformer that polishes to the
lowest valid conformer of the session's three arms, and a conformer whose
stereocentres or double bonds differ from the SMILES is invalid and
dropped.

- branch: anneal 7fd1ebc87706fb62d299a94754da6d6d61d0a9f7, the `anneal`
  rows, with `elite_return` on in `recommended_molecular`. The preset has
  it off again after this record; the rule's `--elite-return` turns it on
  for the anneal arm.
- develop: anneal 027c9c715564a065d02ae566d845c58e7e6da2a5, the `develop`
  rows.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19).

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 20/24 | 1.0000 | 0.1092 | 45 | 45 | 22.5 | 0.00 | 0.00 | 0.00 |
| 915 | 17/24 | 15/24 | 10/24 | 0.7601 | 0.0798 | 4 | 4 | 2 | 0.00 | 0.00 | 0.01 |
| xld | 21/24 | 22/24 | 4/24 | 1.0000 | 0.0000 | 30 | 33 | 4 | 0.00 | 0.00 | 2.74 |
| fluvastatin | 6/24 | 10/24 | 8/24 | 0.3587 | 0.7516 | 36.5 | 39.5 | 7 | 0.22 | 0.05 | 0.48 |
| z34 | 1/24 | 0/24 | 0/24 | 1.0000 | 1.0000 | 19.5 | 24 | 1.5 | 0.53 | 0.53 | 2.38 |
| amprenavir | 4/24 | 3/24 | 0/24 | 1.0000 | 0.1092 | 4.5 | 3 | 0 | 1.70 | 2.08 | 4.19 |
| lopinavir | 3/24 | 2/24 | 0/24 | 1.0000 | 0.2340 | 1 | 0 | 0 | 2.45 | 5.89 | 9.29 |
| lorlatinib | 11/24 | 12/24 | 5/24 | 1.0000 | 0.1246 | 3 | 3 | 2 | 1.21 | 0.60 | 2.68 |

The branch hits 87 of 192 runs, develop 88 and openconf 47. It gains on
915, z34, amprenavir and lopinavir and loses fluvastatin 6 against 10, xld
21 against 22 and lorlatinib 11 against 12; no Fisher p against develop
is below 0.35. Lopinavir's median ΔE_min falls from 5.89 to 2.45 kcal/mol.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 12341 | 12613 | 4165 | 0.8125 | 0.0029 |
| 915 | 11641 | 18883 | 4339 | 0.1307 | 0.0053 |
| xld | 19970 | 32168.5 | 12388.5 | 0.1587 | 0.1483 |
| fluvastatin | 7667 | 13292 | 3523.5 | 0.2763 | 0.4014 |
| z34 | 27404 |  |  |  |  |
| amprenavir | 46157.5 | 56050 |  | 0.5959 |  |
| lopinavir | 70378 | 48892.5 |  | 0.1489 |  |
| lorlatinib | 50436 | 55483.5 | 7971 | 0.6665 | 0.0065 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.86 | 3.85 | 3.78 | 2.23 | 1.64 | 2.23 | 1.62 |
| 915 | 1.62 | 1.62 | 1.93 | 0.91 | 0.71 | 0.91 | 0.71 |
| xld | 4.30 | 4.23 | 4.88 | 2.51 | 1.77 | 2.49 | 1.75 |
| fluvastatin | 3.93 | 3.89 | 4.80 | 2.30 | 1.63 | 2.30 | 1.60 |
| z34 | 4.98 | 5.01 | 5.93 | 2.98 | 2.04 | 2.98 | 2.04 |
| amprenavir | 5.15 | 5.15 | 7.56 | 3.04 | 2.12 | 3.05 | 2.10 |
| lopinavir | 10.12 | 10.09 | 21.07 | 6.03 | 4.08 | 6.00 | 4.08 |
| lorlatinib | 3.51 | 3.47 | 7.12 | 2.01 | 1.49 | 1.98 | 1.49 |

Wall time and the time inside and outside the potential stay within 2% of
develop's.

## Where the paired runs part

`traces-gate.csv` repeats every pair of both anneal arms at its budget,
`snakemake -s workflow/Snakefile openconf_h2h_conf_fix9_traces`. Each
trace spends the calls of its paired run to the call, so it is the same
trajectory; its `energy_hit` is not the paired `hit`, which matches
structures among the conformers the run reports.

| molecule | first return, median share of budget | returns per run | seeds won | seeds lost (develop hit at share) |
|---|---|---|---|---|
| rq3 | 0.08 | 17 |  |  |
| 915 | 0.20 | 5 | 2003, 2010, 2012, 2014, 2017, 2018, 2020 | 2000 (0.75), 2013 (0.62), 2021 (0.38), 2022 (0.74), 2023 (0.70) |
| xld | 0.10 | 10 | 2007 | 2002 (0.36), 2011 (0.51) |
| fluvastatin | 0.13 | 10.5 |  | 2001 (0.55), 2004 (0.27), 2008 (0.39), 2023 (0.25) |
| z34 | 0.12 | 12 | 2015 |  |
| amprenavir | 0.19 | 6.5 | 2000, 2001, 2003, 2015 | 2002 (0.69), 2013 (0.75), 2019 (0.90) |
| lopinavir | 0.15 | 7.5 | 2001, 2010, 2019 | 2000 (0.33), 2021 (0.71) |
| lorlatinib | 0.15 | 9 | 2000, 2004, 2008, 2011, 2015 | 2005 (0.82), 2006 (0.68), 2007 (0.85), 2013 (0.57), 2016 (0.64), 2023 (0.86) |

Every branch run returns, the first time after a median 8 to 20% of its
budget, so every seed develop hits later than that is open to loss.
Fluvastatin's four lost seeds are runs develop hits after a quarter to a
half of its budget, and on fluvastatin the branch wins none back.
Lorlatinib's walker already stands within 1 kcal/mol of its own best on
96% of develop's hops (0.89 on the branch), a cold chain with no plateau
to leave, and the return only reshuffles which seeds hit.

## Decision

The return loses on fluvastatin, xld and lorlatinib, so it is not in
`recommended_molecular`. It stays a switch, `elite_return=True` and
`elite_patience` in `cluster_search`, that leaves a run bit-identical to
develop until a chain has gone `elite_patience` hops without a new best.
