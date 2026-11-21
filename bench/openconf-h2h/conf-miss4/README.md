Copyright (c) anneal developers

conf-miss4 is an 8-molecule, 24-seed paired panel (seeds 2048-2071) of
this series against its own base, develop
`61fa67cf380e46876b569da45e7cc091647a775b`, and against openconf, plus a
guard and pins record for the same two anneal revisions. Every number
below is read from `rows.csv`, `panel.tsv`, `paired.tsv`,
`local_step.csv`, `guard.csv`, `rescore.tsv` or `pins.txt`.

The series changes no search behaviour (see "What the series changes"),
so the question this panel asks of it is whether anything moved. Nothing
did in the outcomes; the only difference measured is a small one in
time.

## Revisions and host

- this series: anneal `259befae8221961d2249aa27d19ca426064af57b`, tree
  `b1369a7edccdb56002f9667acff4f9c122083bb0`, the commit whose tree the
  wheel was built from; the `anneal` rows cite it. The next commit,
  `5c40c711f6f49ce0950d6025349adcde7f17cdce`, tree
  `89078345063859912c378bb5344b959b365331c6`, adds only
  `scripts/openconf_h2h/paired_stats.py` and `pytest/test_paired_stats.py`,
  neither of which is in the wheel; the guard examples were built from
  its tree and the harness ran from it, so the `openconf` rows and the
  guard rows cite it. Both commits are ancestors of this record on this
  branch. The commits after them change the scorer, the statistics and
  the documentation, not the code that ran. `provenance.tsv` lists the
  three commits the rows cite with their trees; the audit checks each
  tree against the repository.
- develop: `61fa67cf380e46876b569da45e7cc091647a775b`, built into its
  own wheel, the `develop` rows. The series is built on this revision.
- openconf: rowansci/openconf `e32c5fa7b6e6aac944bda31c851be42e14340235`
  (v0.0.19), the settings of `../paired`. Its `code_revision` is the
  harness revision, which is this series.
- host label `cloud-vm`, an 8-core virtual machine. The label is the
  `BENCH_HOST` the harness was given; `scripts/openconf_h2h/bench_host.py`
  writes only that label, `bench` or `unlabelled`, never a node name.
- The panel ran serially on one pinned core. For each molecule and seed
  the openconf arm ran first; anneal and develop then ran one after the
  other, each given the budget `(energy_calls + grad_calls) // 2` of
  that openconf run. The other cores of the host were busy with
  per-commit gates, example builds and the guard, so wall times here
  are not quiet-host times; hits, calls and energies do not depend on
  them. The `budget` column of `rows.csv` is that budget, and it equals
  `(energy_calls + grad_calls) // 2` of the seed's openconf row.
- `rows.csv` was scored by `scripts/openconf_h2h/paired_score.py` after
  every candidate was polished; `hit` is its strict test (heavy-atom
  RMSD below 0.25 A and energy within 0.5 kcal/mol of the reference)
  and `hit_mirror` also accepts the reflected candidate, but only when
  the reflected geometry keeps the reference stereo key: for a molecule
  with no stereocentre or a meso one, not for a chiral molecule, whose
  reflection is the other enantiomer.
- `local_step.csv` is read from each anneal and develop `job.json`: the
  resolved local step and, for this series, the `ledger` that
  `cluster_search` returns, whose keys are defined in its docstring.
  Develop has no ledger, so its ledger columns are empty.

## What the series changes

It adds a test that pins which double bonds the torsion move leaves
alone, a `ledger` entry in the Python search result, a mirror-aware hit
score that respects stereo, the host label above and a paired-statistics
script. The `ledger` is assembled in `src/python.rs` after the search has
ended, from counters the search already keeps; nothing in the search
loop changes, which the panel and the guard both confirm below. No tuned
constant is introduced.

## Panel

Hits (strict) over the 192 runs of each arm: this series 91, develop
91, openconf 42. With `hit_mirror`: 94, 94 and 44.

The mirror rule was corrected after the first record of this panel: a
mirror match counts only when the reflected geometry keeps the reference
stereo key. `rescore.tsv` compares the earlier scoring of these rows
(the `rows.csv` of commit `2235db00`) with the one in this directory,
per molecule and arm, and `rescore-cells.tsv` lists every differing
cell of `hit`, `hit_mirror`, `calls_to_hit`, `delta_min_kcal`,
`candidates` and `invalid`. It has no row: no hit or mirror hit of any
of the 576 runs moved. The 8 runs that count only as a mirror hit are
xld (3 anneal, 3 develop) and rq3 (2 openconf). Neither molecule has a
stereocentre, so reflection keeps its key and the rule lets them stand.
The earlier rule counted no run of this panel that the corrected rule
rejects. The corrected rule is tested on ethanol-1-d, chiral by its
deuterium alone (`pytest/test_h2h_mirror.py`).

Per-seed outcome (hit, calls to the hit, energy above the reference,
calls, candidates, invalid) is identical between this series and
develop on all 192 seeds. Every exact two-sided McNemar test of
`paired.tsv` therefore has no discordant seed and p = 1.0000. The hits
per molecule, this series and develop alike, are 915 15/24, amprenavir
3/24, fluvastatin 11/24, lopinavir 4/24, lorlatinib 11/24, rq3 24/24,
xld 21/24 and z34 2/24.

`invalid` counts candidates that fail the stereo check on polishing. It
is 1 for each of this series and develop (lorlatinib) and 0 for
openconf.

`local_step`: all 384 anneal and develop runs resolved to `lbfgs`. On
this base `Auto` picks the local step by timing one call of each, and
on this host it picked `lbfgs` every time; a host with a more
expensive potential may pick `tric`, and these rows say nothing about
that case.

## Time outside the potential

`paired.tsv` compares, per seed, the time outside the potential per
call, `outside_s / calls`, of this series with develop. The medians in
microseconds per call, this series then develop, the median of the
per-seed ratio, and the two-sided Wilcoxon signed-rank p. The p is
exact; pairs with equal values are dropped first, and none of the 192
pairs is equal or ties another in absolute difference, so scipy's
automatic choice gives the same p here:

| molecule    | series | develop | ratio  | p      |
|-------------|--------|---------|--------|--------|
| 915         | 15.20  | 15.00   | 1.0091 | 0.0179 |
| amprenavir  | 29.54  | 29.32   | 1.0021 | 0.3596 |
| fluvastatin | 21.05  | 21.03   | 1.0058 | 0.1074 |
| lopinavir   | 47.21  | 47.54   | 1.0027 | 0.5646 |
| lorlatinib  | 20.74  | 21.12   | 1.0051 | 0.9664 |
| rq3         | 19.25  | 18.82   | 1.0226 | 0.0229 |
| xld         | 22.80  | 22.92   | 0.9961 | 0.3748 |
| z34         | 26.17  | 26.08   | 1.0235 | 0.1600 |

The ratio is above 1 on seven molecules, by 0.2 to 2.4 per cent, and
the p values are not corrected for eight tests; two are below 0.05 and
none is below 0.05 / 8. Both arms run the same search code, so a ratio
here is not a cost of this series. The host was shared, and anneal ran
before develop on each seed; neither effect was separated from the
other, so the table does not say why the ratio is above 1 on seven
molecules.

## Guard

`guard.csv` holds the 156 jobs of the 13-system guard (`lj38`, `lj55`,
`lj75`, `lj98`, `lj102`, `lj103`, `lj104`, `lj150`, `au55`, `m55r6`,
`m55r14`, `m80r6`, `m80r14`; seeds 2000-2011) for develop and for this
series, 312 rows. Comparing the two row by row, `hit`, `evals_to_hit`,
`charged` and `best` have 0 mismatches. 104 of the 156 jobs hit on each
side.

`pins.txt` is the 91-line output of `scripts/covalent_h2h/pins.sh` for
this series. The output for develop is byte-identical; sha1
`83d07464f1493b3846bc92b3d1f191331dacd586`.
