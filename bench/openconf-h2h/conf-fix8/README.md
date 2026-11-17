Copyright (c) anneal developers

Why coupled torsion draws lose lowest conformers develop finds, and a
patience before they start. A negative record: the 24-seed gate still
loses on three molecules, so `torsion_coupling` stays out of
`recommended_molecular`.

- branch: anneal 85e649390af80c00deabf955b0895008e6af7835, the `anneal`
  rows: coupled torsion draws (conf-fix7) on in `recommended_molecular`,
  after `torsion_patience` = 100 hops without a new best.
- develop: anneal 98dcfb3106f191baf7381fecd5624426831094fe, the `develop`
  rows,   built and run in the same session. Its 192 rows repeat those of
  conf-fix7's develop fb178f40 hit for hit and call for call.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235
  (v0.0.19), the settings of `../paired`; its 192 rows repeat conf-fix7's.

## Why conf-fix7's move lost lopinavir and amprenavir

conf-fix7 gate (seeds 2000-2023, develop fb178f40) lost lopinavir's two
develop hits (seeds 2000 and 2021) and three of amprenavir's (2002, 2013,
2019), while winning amprenavir 2015 and 2023. `trace-patience0.csv`
reruns both arms of every gate pair at its gate budget with the hop
trace on (`scripts/openconf_h2h/coupling_trace.py --patience 0`, the
conf-fix7 move; its rows match the conf-fix7 runs). Per molecule, over
the 24 seeds:

| molecule | runs with a coupled draw | median first coupled hop | median hop the chains part | coupled draws accepted | plain draws accepted, coupled arm | plain draws accepted, develop | median energy above reference, develop (kcal/mol) | coupled (kcal/mol) |
|---|---|---|---|---|---|---|---|---|
| z34 | 24/24 | 5.5 | 3 | 35% of 3860 | 39% | 38% | 7.2 | 6.4 |
| amprenavir | 24/24 | 6 | 3 | 47% of 2411 | 54% | 52% | 10.4 | 9.5 |
| lopinavir | 24/24 | 5 | 2.5 | 51% of 2777 | 57% | 56% | 20.3 | 20.8 |
| 915 | 24/24 | 6 | 4 | 50% of 1971 | 54% | 57% | 7.8 | 6.1 |
| fluvastatin | 24/24 | 5.5 | 3 | 46% of 3241 | 54% | 54% | 3.9 | 3.9 |
| xld | 24/24 | 7.5 | 4 | 49% of 3299 | 51% | 51% | 4.7 | 4.4 |
| lorlatinib | 0/24 | | | none | 52% | 52% | 2.7 | 2.7 |
| rq3 | 24/24 | 5.5 | 3 | 49% of 4748 | 53% | 54% | 2.3 | 2.4 |

The energy columns are the median, over seeds, of the unpolished energy
the chain stands at across its run. Coupled draws are accepted about as
often as plain ones, the plain draws of the coupled arm as often as
develop's, and the chains stand as high above the reference either way;
lopinavir's sit 20 kcal/mol up in both arms. Nothing in the coupled draws
pushes the chain uphill. What changes is the trajectory: the quench after
a plain one-to-three bond turn often moves further bonds by 30 degrees,
so every chain but lorlatinib's has a coupled change on file between
its second and sixteenth hop, the third or fourth in the median. From
then on each torsion draw spends one more
random number on the coupled share, and the chain leaves develop's path
at once. The sets turned average 2.4 to 2.9 bonds.

The lost seeds are develop runs that happened to fall into the funnel of
the lowest conformer late in the run (share of develop's calls spent at
its hit), against coupled runs that parted from them in the first few
hops:

| seed | develop hit at | chains part at hop | coupled draws accepted | best develop (kcal/mol) | best coupled (kcal/mol) |
|---|---|---|---|---|---|
| lopinavir 2000 | 33% | 3 of 423 | 88/133 | 0.95 | 13.13 |
| lopinavir 2021 | 71% | 3 of 447 | 38/98 | 1.35 | 7.99 |
| amprenavir 2002 | 69% | 3 of 461 | 53/130 | 0.58 | 5.81 |
| amprenavir 2013 | 75% | 3 of 360 | 47/82 | 2.22 | 4.92 |
| amprenavir 2019 | 90% | 7 of 354 | 35/90 | 1.78 | 3.93 |

Best energies are unpolished, above the reference; a hit is scored on the
polished minima. Develop's chain reaches 0.6 to 2.2 kcal/mol above the
reference; the coupled chain from the same seed, accepting about as
often and standing about as high on the median (lopinavir 2021: 16.1
develop, 15.4 coupled; amprenavir 2013: 11.7 and 9.3), never comes
closer than 3.9 to 13.1. Across the 24 seeds the outcome flips both ways (amprenavir +2/-3, lopinavir 0/-2,
915 +7/-4, xld +7/-2, fluvastatin +6/-2): on these molecules a hit is a
rare descent, and a move that changes the random stream from hop three
redraws every seed's luck. On the development seeds (3000-3119) the
same move has lopinavir 2 and amprenavir 9 hits against develop's 1
and 8; it does not find them less often, it finds them on other seeds.

## The move, and its share

conf-fix7's move is cherry-picked unchanged (326e390e, 01a377e8). A
chain records the rotatable bonds each accepted hop changed after the
quench; a torsion draw of a chain with a set of two or more bonds on file
turns one such set to fresh uniform angles with probability one quarter,
and otherwise draws the plain move. Its 24-seed gate hit 93 of 192 runs
against develop fb178f40's 82, losing amprenavir 2002, 2013 and 2019 and
lopinavir 2000 and 2021. On the first half of the development seeds
(seeds 3000-3059) a share adapted from how often
each kind of draw moved the chain settled near one half and hit 201 to
226 runs out of 480 across the variants tried, against 233 for the fixed
quarter, losing most on 915 and xld; the quarter stays.

## Development variants

Seeds 3000-3119 at the molecules' median budgets, scored by the
conf-fix6 `dev_diagnose.py analyse` hit rule (a valid polished minimum
within 0.1 kcal/mol of the conf-fix6 reference). Each row's code is
conf-fix7's move rebased on 98dcfb31 (01a377e8) with temporary knobs that
are not part of the branch; the patience-100 build was checked against
the committed code, which repeats its rows. `dev-rows.csv` has one row
per (molecule, arm, seed); develop and patience 0 are the conf-fix7 rows
(develop fb178f40, branch 9392b44e). The dev runs ran several at a time
outside the bench namespace, so the rows carry no host and their times
are not compared; the gate's are. Hits out of 120, with the seeds won
and lost against develop:

| arm | z34 | amprenavir | lopinavir | 915 | fluvastatin | xld | lorlatinib | rq3 | total | 24-seed blocks losing nowhere | P(lose nowhere at 24) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| develop | 6 | 8 | 1 | 81 | 46 | 93 | 56 | 120 | 411 | | |
| patience 0 (conf-fix7) | 7 (+5/-4) | 9 (+7/-6) | 2 (+2/-1) | 88 (+24/-17) | 54 (+37/-29) | 108 (+23/-8) | 56 | 119 (+0/-1) | 443 | 2/5 | 0.17 |
| motif: set to the angles of the recorded minimum | 5 (+4/-5) | 11 (+11/-8) | 2 (+2/-1) | | | | | | | 1/5 | 0.38 |
| incumbent: set to the angles of the best minimum | 6 (+6/-6) | 10 (+9/-7) | 2 (+2/-1) | | | | | | | 3/5 | 0.39 |
| downhill: record only hops that lowered the energy | 5 (+5/-6) | 15 (+15/-8) | 3 (+3/-1) | 82 (+22/-21) | 54 (+34/-26) | 104 (+24/-13) | 56 | 120 | 439 | 1/5 | 0.16 |
| patience 100 (shipped) | 5 (+2/-3) | 12 (+7/-3) | 2 (+1/-0) | 87 (+13/-7) | 56 (+28/-18) | 102 (+18/-9) | 56 | 120 | 440 | 2/5 | 0.30 |
| patience 200 | 8 (+3/-1) | 8 (+1/-1) | 1 (+0/-0) | 83 (+2/-0) | 45 (+8/-9) | 95 (+8/-6) | 56 | 120 | 416 | 2/5 | 0.31 |
| patience 100, downhill | 8 (+5/-3) | 13 (+7/-2) | 3 (+2/-0) | 81 (+8/-8) | 49 (+22/-19) | 98 (+18/-13) | 56 | 120 | 428 | 1/5 | 0.21 |

The motif and incumbent rows ran the three weak molecules only; their
last two columns cover those three. A block is 24 consecutive seeds, a
stand-in for the gate. P(lose nowhere at 24) treats each molecule's wins
and losses per 24 seeds as Poisson at the rates of the 120 seeds and
multiplies the chance that wins reach losses over the molecules.

Setting the coupled bonds to angles the chain had already found, instead
of fresh uniform ones, changes nothing measurable. Recording only
downhill hops lifts amprenavir but gives back 915. A patience leaves the
chain on develop's trajectory, random stream included, until the first
100 hops without a new best, and cuts the seeds lost on every molecule;
100 hops keeps the gains on 915, fluvastatin and xld, 200 loses most of
them because few chains stall that long. No variant reaches better than
about one chance in three of losing nowhere at 24 seeds: the gains come
from changed trajectories, and a changed trajectory redraws develop's
rare late hits.

## Gate

The three-arm paired run of `../conf-fix4`, all eight molecules on seeds
2000-2023, MMFF94 through RDKit 2025.09.6 at a dielectric of one, in one
serial session under `sudo unshare --uts`, hostname bench, every job on
core 0: `snakemake -s workflow/Snakefile openconf_h2h_conf_fix8`. Budgets,
arm order, hits, ΔE_min and the tests are those of `../conf-fix`.
`seeds.csv` carries each row's `code_revision` and `host`.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf | Mann-Whitney p ΔE_min vs develop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 20/24 | 1.0000 | 0.1092 | 46 | 45 | 22.5 | 0.00 | 0.00 | 0.00 | 1.0000 |
| 915 | 13/24 | 15/24 | 10/24 | 0.7702 | 0.5639 | 4 | 4 | 2 | 0.00 | 0.00 | 0.01 | 0.6851 |
| xld | 19/24 | 16/24 | 2/24 | 0.5171 | 0.0000 | 38 | 33 | 4 | 0.00 | 0.00 | 2.74 | 0.1617 |
| fluvastatin | 13/24 | 10/24 | 8/24 | 0.5639 | 0.2443 | 40.5 | 39.5 | 7 | 0.03 | 0.05 | 0.48 | 0.5124 |
| z34 | 1/24 | 0/24 | 0/24 | 1.0000 | 1.0000 | 24.5 | 26.5 | 1.5 | 0.43 | 0.43 | 2.28 | 0.3008 |
| amprenavir | 0/24 | 3/24 | 0/24 | 0.2340 | 1.0000 | 4 | 3 | 0 | 1.17 | 2.08 | 4.19 | 0.2970 |
| lopinavir | 1/24 | 2/24 | 0/24 | 1.0000 | 1.0000 | 0 | 0 | 0 | 6.15 | 5.89 | 9.29 | 0.8689 |
| lorlatinib | 12/24 | 12/24 | 5/24 | 1.0000 | 0.0687 | 3 | 3 | 2 | 0.60 | 0.60 | 2.68 | 1.0000 |

The branch hits 83 of 192 runs, develop 82 and openconf 45 (Fisher p 1.0
vs develop over the eight). xld (19 against 16), fluvastatin (13 against
10) and z34 (1 against 0) gain; 915 (13 against 15), amprenavir (0
against 3) and lopinavir (1 against 2) lose. No molecule moves
significantly. No arm has an invalid candidate or an error.

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 12613 | 12613 | 4165 | 0.9178 | 0.0025 |
| 915 | 17388 | 18883 | 4339 | 0.6447 | 0.0039 |
| xld | 32324 | 36271.5 | 10363 | 0.7404 | 0.0381 |
| fluvastatin | 21529 | 13292 | 3523.5 | 0.4560 | 0.0126 |
| z34 | 52892 | | | | |
| amprenavir | | 56050 | | | |
| lopinavir | 33079 | 48892.5 | | 1.0000 | |
| lorlatinib | 55483.5 | 55483.5 | 7971 | 1.0000 | 0.0013 |

`trace-patience100.csv` is the trace of the branch at the gate budgets.
The chains now part at hop 164 to 216 (median over seeds, per molecule),
13% to 97% of the way in, and the seeds that flip are develop hits made after that:

| seed | develop hit at | chains part at hop | coupled draws accepted | best develop (kcal/mol) | best branch (kcal/mol) | branch hit |
|---|---|---|---|---|---|---|
| lopinavir 2000 | 33% | 374 of 423 | 0/0 | 0.95 | 0.80 | yes |
| lopinavir 2021 | 71% | 167 of 447 | 3/10 | 1.35 | 9.54 | no |
| amprenavir 2002 | 69% | 308 of 461 | 3/16 | 0.58 | 0.79 | no |
| amprenavir 2013 | 75% | 116 of 360 | 19/54 | 2.22 | 8.03 | no |
| amprenavir 2019 | 90% | 228 of 354 | 18/33 | 1.78 | 1.78 | no |
| 915 2000 | 75% | 177 of 254 | 1/18 | 0.43 | 0.43 | no |
| 915 2008 | 87% | 199 of 332 | 12/35 | 0.35 | 0.35 | no |
| 915 2022 | 74% | 200 of 367 | 21/39 | 0.18 | 1.85 | no |

The patience keeps lopinavir 2000, whose hit precedes the first stall,
and loses the hits that come later. Amprenavir's three develop hits all
come in the last third of the budget. In amprenavir 2019 and 915
2000/2008 the branch's lowest unpolished energy equals develop's and it
still misses: the hit is scored on the polished minima, and the two
chains polished different sets.

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.95 | 3.92 | 3.86 | 2.27 | 1.68 | 2.26 | 1.66 |
| 915 | 1.68 | 1.65 | 2.02 | 0.94 | 0.74 | 0.93 | 0.73 |
| xld | 4.44 | 4.49 | 5.04 | 2.61 | 1.79 | 2.64 | 1.85 |
| fluvastatin | 4.01 | 4.10 | 4.97 | 2.35 | 1.67 | 2.40 | 1.69 |
| z34 | 5.10 | 5.05 | 6.06 | 3.00 | 2.09 | 3.00 | 2.06 |
| amprenavir | 5.28 | 5.35 | 7.83 | 3.11 | 2.17 | 3.18 | 2.18 |
| lopinavir | 10.35 | 10.26 | 21.43 | 6.13 | 4.21 | 6.11 | 4.16 |
| lorlatinib | 3.64 | 3.60 | 7.32 | 2.08 | 1.55 | 2.06 | 1.54 |

The patience costs nothing measurable: outside the potential the arms
differ by at most 0.06 s per run, inside by the call counts, which match.

LJ, Morse and reduced-unit pins (`scripts/covalent_h2h/pins.sh`) and the
Gupta Au55 rows are bit-identical between develop 98dcfb31 and the
branch: a run without torsion groups never builds the history.

## Decision

The gate fails: the branch loses on 915, amprenavir and lopinavir. It
keeps gains on xld, fluvastatin and z34. `torsion_coupling` stays off in
`recommended_molecular`; the move and its `torsion_patience` stay
opt-in, with the patience at 100. Losing nowhere at 24 seeds against a
same-seed develop is out of reach of a move that changes trajectories
unless it raises every weak molecule's rate well above develop's: on
these molecules develop's hits are rare late descents, and a changed
trajectory redraws them.
