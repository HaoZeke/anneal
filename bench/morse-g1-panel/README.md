# Same-machine G1 Morse panel

This directory records anneal and GMIN on the G1 Morse cells. Both arms
run in one serial session on core 0. Seeds are 2000 through 2011. The
first arm alternates by seed, producing ABBA repeatedly.

Anneal is the release build of
`61fa67cf380e46876b569da45e7cc091647a775b`. GMIN is revision
`9cbec1a446352bc9ef064bd2b1a5767194655919` from the archive with
SHA-256
`ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776`.
The archive digest is checked before extraction.

References are the Cambridge Cluster Database Morse values in
`examples/fixtures/morse_ccd.csv`. A hit is a finite best energy no
greater than the reference plus 0.001. A miss records its full budget.

| Systems | rho | Budget per seed |
| --- | --- | ---: |
| M30 | 6, 14 | 400000 |
| M38 | 6, 14 | 400000 |
| M55 | 6, 14 | 1000000 |
| M70 | 3, 6, 10, 14 | 1000000 |
| M80 | 3, 10 | 4000000 |

`datafile-rule.md` defines the single GMIN input rule. `rows.csv` is
the complete raw record. `summary.csv` contains the per-cell tests and
type-7 medians. `anneal-check.csv` compares deterministic calls and
energy bits with the G1 anneal rows. `g1-stored-provenance.csv` audits
every stored Morse row presented under a GMIN label in G1.
`gmin-comparison.csv` compares only stored rows with `code=gmin` and a
nonempty recorded GMIN revision.

The G1 `summary.csv` field named `gmin_hits` is not an input to this
analysis. The G1 `panel-rows.csv` Morse records whose cell names begin
with `gmin-` contain anneal revisions and no code field; they are not
treated as GMIN observations. Rows without a recorded GMIN revision
are also excluded from the stored-GMIN comparison.

## Results

Calls and wall time are type-7 medians. The Mann-Whitney test ranks a
miss at positive infinity.

| Cell | Hits A/B | Fisher p | Median calls A/B | Median wall A/B (s) | MWU p, calls |
| --- | --- | ---: | ---: | ---: | ---: |
| M30, rho=6 | 12/11 | 1 | 5184.5/66113.5 | 0.048/0.525 | 0.000308 |
| M30, rho=14 | 8/1 | 0.00942 | 195880/400000 | 1.348/3.107 | 0.00718 |
| M38, rho=6 | 12/10 | 0.478 | 49438.5/157694.5 | 0.544/1.804 | 0.0120 |
| M38, rho=14 | 4/1 | 0.317 | 400000/400000 | 4.107/4.515 | 0.113 |
| M55, rho=6 | 12/12 | 1 | 22596/37047.5 | 0.515/0.853 | 0.931 |
| M55, rho=14 | 9/0 | 0.000337 | 451561.5/1000000 | 9.730/20.981 | 0.000383 |
| M70, rho=3 | 11/10 | 1 | 374226/295339 | 11.501/8.705 | 0.470 |
| M70, rho=6 | 11/12 | 1 | 65359/235267 | 1.936/7.397 | 0.0690 |
| M70, rho=10 | 6/0 | 0.0137 | 995598/1000000 | 29.735/29.850 | 0.00705 |
| M70, rho=14 | 10/0 | 0.0000673 | 368941/1000000 | 10.521/28.763 | 0.000125 |
| M80, rho=3 | 12/12 | 1 | 916054/80970 | 34.629/3.004 | 0.000592 |
| M80, rho=10 | 6/1 | 0.0686 | 3539041.5/4000000 | 134.516/153.860 | 0.0340 |

The deterministic check found exact call and energy-bit equality for
all 120 anneal rows present in G1. G1 has no corresponding M55 rows in
`rows.csv`, so the 24 M55 checks are reported as not present rather
than as mismatches.

## Stored-row audit

The audit found 60 `rows.csv` records with `code=gmin` and a recorded
GMIN revision. They cover M30 at rho 6 and 14, M38 at rho 6 and 14,
and M80 at rho 3. Another 60 rows have a GMIN code but no revision or
host and are excluded. All 168 Morse rows in `panel-rows.csv` whose
labels begin with `gmin-` carry anneal revisions and no code field;
they are classified as anneal rows under a GMIN label.

| Cell | Fresh/stored GMIN hits | Fresh/stored median calls | Fresh/stored median wall (s) |
| --- | ---: | ---: | ---: |
| M30, rho=6 | 11/11 | 66113.5/66113.5 | 0.525/0.418 |
| M30, rho=14 | 1/1 | 400000/400000 | 3.107/2.488 |
| M38, rho=6 | 10/11 | 157694.5/146865.5 | 1.804/1.332 |
| M38, rho=14 | 1/0 | 400000/400000 | 4.515/3.591 |
| M80, rho=3 | 12/12 | 80970/114617.5 | 3.004/4.047 |

The stored eligible rows use revision
`aac1548a8aec13b53ad12a8d6904625081685d2e`, the example Morse
keywords, and no explicit radius. The fresh rows use the pinned
revision above and write the native Morse default radius explicitly.
The stored and fresh machine identifiers differ. These are input and
machine differences, so the table is descriptive rather than a paired
same-machine inference.
