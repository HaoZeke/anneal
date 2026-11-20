Copyright (c) anneal developers

conf-miss4 is an 8-molecule, 24-seed paired panel (seeds 2048-2071) of
this series against its own base, develop
`61fa67cf380e46876b569da45e7cc091647a775b`, and against openconf, plus a
guard and pins record for the same two anneal revisions. Every number
below is read from `rows.csv`, `panel.tsv`, `paired.tsv`,
`local_step.csv`, `guard.csv` or `pins.txt`.

The series changes no search behaviour (see "What the series changes"),
so the question this panel asks of it is whether anything moved. Nothing
did in the outcomes; the only difference measured is a small one in
time.

## Revisions and host

- this series: anneal `373c7817b29bcfeb001afba62e772a9f57e7ee17`, the
  revision the wheel was built from. The next commit of the series,
  `02e6d90aa492d06cabac5a52891ec479102f67dc`, adds only
  `scripts/openconf_h2h/paired_stats.py` and `pytest/test_paired_stats.py`,
  neither of which is in the wheel; the guard examples were built from
  `02e6d90a`. The records commit adds only this directory.
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
  and `hit_mirror` also accepts the reflected candidate.
- `local_step.csv` is read from each anneal and develop `job.json`: the
  resolved local step and, for this series, the budget ledger. Develop
  has no ledger, so its ledger columns are empty.

## What the series changes

It adds a test that pins which double bonds the torsion move leaves
alone, a `ledger` entry in the Python search result, a mirror-aware hit
score, the host label above and a paired-statistics script. The search
code is not touched, which the panel and the guard both confirm below.
No tuned constant is introduced.

## Panel

Hits (strict) over the 192 runs of each arm: this series 91, develop
91, openconf 42. With `hit_mirror`: 94, 94 and 44.

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
per-seed ratio, and the two-sided Wilcoxon signed-rank p:

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
none is below 0.05 / 8. The ledger adds counters to the search, and the
host was shared, so this is consistent with a cost of about a per cent
and does not establish one.

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
