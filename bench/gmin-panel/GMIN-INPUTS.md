# Lennard-Jones GMIN inputs

This note compares the GMIN arm behind `rows-lj-per-size.csv` with the
panel recorded on `record/g1-develop-stock` at tree `b3ce4162`
(`bench/g1-develop-stock/panel-rows.csv` and `summary.csv`). The
comparison is taken from those files.

## Where the earlier hit counts come from

`panel-rows.csv` stores one measured arm. Its `gmin-lj98` and
`gmin-lj150` cells are the stock recommended runs reused from
`bench/g0-lj-stock/rows.csv`. On calls and energies those cells match
the anneal arm of `rows-lj-per-size.csv` on every seed.

`summary.csv` lists `gmin_hits` 10 for LJ98 and 5 for LJ150.
`bench/g1-develop-stock/summarize.py` fills that column from
`bench/g0-lj-stock/manifest.csv` field `paper_hit`. The manifest's
`paper_source` is `bench/gmin-panel/rows.csv`. The `paper_hit` and
`paper_calls` values match the anneal rows of that file
(`code` `anneal`, `code_revision`
`c2de652dca0bc75e2b0cb4afffac91fd656404e4`). The GMIN rows of the same
file (`code` `gmin`, `code_revision`
`aac1548a8aec13b53ad12a8d6904625081685d2e`) are 0/12 on LJ98 and 9/12
on LJ150. The GMIN arm of `rows-lj-per-size.csv` is 0/12 on LJ98 and
9/12 on LJ150.

`bench/g1-develop-stock/run_panel.py` launches the anneal example,
`taskset -c 0 <binary> run <system> <seed>`. It does not launch GMIN.
The GMIN command and data file below are the ones in
`scripts/gmin_panel/run_panel.py` on that tree, which is the harness
that wrote `bench/gmin-panel/rows.csv`.

## RADIUS

`examples/h2h_gmin_lj.data` on that tree has no `RADIUS` line. Its
comment says the same file is used at every size. `gmin_data` returns
that text plus `RANSEED <seed>`.

This panel writes `bench/gmin-panel/lj-data/<system>.data` and appends
the same `RANSEED` line. Each file copies the keyword lines of
`examples/h2h_gmin_lj.data` and adds one `RADIUS` line, from
`lj-datafile-rule.md`: the unset-radius expression in `source/io1.f90`
at GMIN revision `9cbec1a446352bc9ef064bd2b1a5767194655919`, including
the `CASE DEFAULT` factor `2**(1/6)`.

| Size | RADIUS |
| --- | --- |
| LJ38 | 4.3305271907 |
| LJ55 | 4.6040875226 |
| LJ75 | 4.8610434366 |
| LJ98 | 5.1050105138 |
| LJ102 | 5.1434055170 |
| LJ103 | 5.1528468913 |
| LJ104 | 5.1622273524 |
| LJ150 | 5.5410279220 |

An omitted `RADIUS` leaves the container to the binary that reads the
file. This note does not re-read `io1.f90` at `aac1548a`.

## Other keywords

The keyword lines other than `RADIUS` are the same text, in the same
order, in `examples/h2h_gmin_lj.data` and in every per-size file:

```
SLOPPYCONV 5.0D-3
TIGHTCONV 1.0D-3
EDIFF 0.003
MAXIT 600 500
SAVE 2
CENTRE
SORT
TEMPERATURE 1.0
STEPS 90000000 1.0
STEP 0.40 0.51
NEWRESTART 40
AVOID 4.0 20 F
```

Both harnesses append `RANSEED <seed>` in the run directory. Both write
the start with the anneal example's `coords` subcommand for that system
and seed.

## Budget accounting

Both harnesses set `H2H_BUDGET` from `PAPER_BUDGETS`: 400,000 at LJ38,
1,000,000 at LJ55, and 4,000,000 at LJ75, LJ98, LJ102, LJ103, LJ104 and
LJ150. Both arms of a size use that budget. `STEPS` is
`90000000 1.0` in both data files, above every one of those budgets.
The evaluator cap is `bench/lj104-lattice/v2/h2h-gmin.patch`, and that
patch is the same text on the two trees. It reads `H2H_BUDGET` and
refuses a potential call once `NPCALL` exceeds the cap. A miss is
stored with `evals_to_hit` and `charged` equal to the budget.

## Command line

On both trees `run_gmin` uses the same command and the same environment
keys:

```
taskset -c 0 <bindir>/GMIN
```

with `H2H_BUDGET`, `H2H_SYSTEM`, `H2H_SEED`, `H2H_TARGET` (the reference
to ten decimal places), `H2H_TOL` and `H2H_ROW`, and with the working
directory holding `data` and `coords`. The difference in the launch is
the `data` file, described above.

## Binary

The GMIN rows in `bench/gmin-panel/rows.csv` record code revision
`aac1548a8aec13b53ad12a8d6904625081685d2e`. `gmin-source.txt` records
tarball sha256
`a972c29b081a74235286bf3e8755c8811da2f7e2a4faab81ffb41570d6d2fe05` and
binary sha256
`4665badc28f4d765117771bea47cf1ebcb93c5e7a7456679f17b53aaff15fe69`.

`rows-lj-per-size.csv` records GMIN revision
`9cbec1a446352bc9ef064bd2b1a5767194655919`, tarball sha256
`ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776` and
binary sha256
`7e3dad3eebadad1948ac388c29d9fb915ab8a09dd177d2d670a17d4475cb18ad`.

`scripts/gmin_panel/build_gmin.sh` on tree `b3ce4162` pins that same
`9cbec1a4` tarball. The rows `summary.csv` counts under `gmin_hits`
were not produced by a build of that pin.
