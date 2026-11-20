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
