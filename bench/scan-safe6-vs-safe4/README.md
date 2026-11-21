# scan-safe6 r6 against develop

This revision was written before any build or run of the r6
comparison.
The cell list, the budgets, the references, the seeds, the round
order and the bars below are fixed. Nothing in them is chosen from
the outcomes.

## Arms

- A is `develop` at
  `fac18c6ee91a0b0d266e231d05af3875ba09b376`, tree
  `facb1879318bd87905129cbf14ea2841001b1da8`.
- A2 is a second release build of that same commit, in its own target
  directory. It is the A/A placement arm.
- B is the supplied 14-commit r6 stack on A. Its canonical tip is
  `de19575d6d7c07a5c4b85bc321871fae5eb798f3`, and its tree must be
  `ed7566889ff824533b855c0304d2a142ed27e334`. The supplied mailbox
  SHA-256 is
  `d14c1a620dd77c52fad5640edfce92d8f37c896c25e20b044327a86401248c74`.

The stack explicitly records changed search trajectories, charged
counts, energies, hit masks, and symmetry behaviour. Bit identity is
therefore reported but does not gate this comparison.

## Cells

Budgets and references are the ones already used by the nm3 scan-safe
panel driver. Lennard-Jones values are the paper budgets in
`examples/h2h_gmin_proof.rs`. Morse values are the Cambridge table in
`examples/h2h_morse.rs`. Gold at 55 is the Cambridge Gupta entry in
`examples/h2h_metal.rs`, with the same basin stride of 8 that driver
uses when no environment override is set. A hit is a finite best
energy at or below the reference plus `1e-3`.

| cell | atoms | budget | reference |
| --- | ---: | ---: | ---: |
| lj38 | 38 | 400000 | -173.928427 |
| lj55 | 55 | 1000000 | -279.248470 |
| lj75 | 75 | 4000000 | -397.492331 |
| lj98 | 98 | 4000000 | -543.665361 |
| lj102 | 102 | 4000000 | -569.363652 |
| lj103 | 103 | 4000000 | -575.766131 |
| lj104 | 104 | 4000000 | -582.086642 |
| lj150 | 150 | 4000000 | -893.310258 |
| au55 | 55 | 1000000 | -191.03085280 |
| m80r6 | 80 | 4000000 | -378.333471 |
| m80r14 | 80 | 4000000 | -340.811371 |

## Seeds and rounds

Six seeds, `2000` through `2005`.

Eight rounds, in this order, each round covering every cell and every
seed once:

1. A
2. B
3. B
4. A
5. A
6. A2
7. A2
8. A

That is ABBA, followed by the A/A placement band A-A2-A2-A.
Forty-eight planned runs per cell, 528 planned runs in total.

## How a run is launched

Release builds. The measured process is `taskset -c 0`. Every
environment variable whose name ends in `_NUM_THREADS` is `1`,
including `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`RAYON_NUM_THREADS` and the rest of that family. One machine. The
machine id stored on a row is the SHA-256 of the local machine id,
not the id itself. No quick flag. No budget, reference, or search
setting is changed after this file. `H2H_*` and `ANNEAL_*` variables
are removed from the measured environment so they cannot retune a run.

The orchestrator is pinned to the other cores so its work does not
share core 0 with the measured process.

## Rows, journal, exits

Each row stores the best energy (the value that decides the hit), the
final energy, and the process exit code, plus calls to the hit,
charged calls, the trajectory digest, wall time, time inside the
potential, time outside it, the nearest-index call count, and the
three memory figures below.

`calls` is the charged count at the first improving record that hits,
or the budget when the run misses. `charged` is the ledger spend at
the end. Time outside the potential per call is `outside_s / charged`.

The trajectory digest is the SHA-256 of a little-endian encoding:
charged, calls to the hit, final-energy bits, best-energy bits, then
each accepted transition in order (hop, action, from-energy bits,
to-energy bits, adopted flag, destination coordinate bits), then the
final coordinates.

`journal.log` and `rows.csv` are append-only. Every run gets a start
line before the
process is spawned and an end line after it is reaped. A later start
of the orchestrator appends a restart line and does not repeat a run
that already has a start line. Rows are never rewritten. Earlier
safe4 rows are retained under `superseded-safe4/` and are not input
to this analysis.

A non-zero exit or unreadable measurement may be replaced, but the
superseded row remains. At most two replacement runs are allowed per
cell across the entire schedule. A third need for replacement makes
the cell inconclusive. A start line with no row is not launched again
and also makes the cell inconclusive.

## Bars

Statistics reuse the scan-safe panel definitions. Medians are
Hyndman-Fan type 7. Fisher's exact test is two-sided. Mann-Whitney U
is two-sided, with the normal approximation, the tie correction and
the continuity correction. The bootstrap redraws each arm with
replacement 10000 times and reports the type-7 0.025 and 0.975
quantiles of the median ratio. The generator for cell number `k`
(from 0 in the table above) is seeded with `2000 + k`.

A seed contributes one hit and one calls-to-hit value per arm. Those
are the common value of that arm's replicates. If the replicates of
one arm disagree on the hit or on calls-to-hit, the hits bar is
inconclusive for the cell.

1. Hits. B loses no hits against A on any cell: the bar fails when
   B's hit count is lower than A's. The two-sided Fisher exact p-value
   on the hit table is reported with that comparison. The same bar
   also fails when the Mann-Whitney test on calls-to-hit, with a miss
   entered as positive infinity, has p below 0.05 and B's calls are
   the worse sample (its U is above the null mean).
2. Time outside the potential per call. The point estimate is the
   median of B divided by the median of A. The bar fails on a cell
   when that ratio is above 1.10 and the lower end of its 95%
   bootstrap interval lies above the upper end of the A2/A interval.
3. Bit identity of B against A, per cell, in the final energy, in
   `calls`, and in the trajectory digest. Every B replicate of a seed
   is compared with every A replicate of that seed on all three.
   This is report-only because the r6 stack claims behaviour changes.
4. Safe path exercise. Every B run must report a nonzero count of
   calls from the r6 timed binary's always-on safe6 nearest-image
   counter. A zero or missing count fails this bar. A and A2 report
   zero because the counter does not exist in A.

Memory is recorded three ways. After exec, the benchmark process
polls its own status from before the measured search until after it
finishes and emits its peak `RssAnon`. This self-reported value is the
only memory gate. It is analysed as a median ratio with the same
bootstrap comparison as the time bar and with no 1.10 allowance. The
bar fails when the lower end of the B/A interval lies above the upper
end of the A2/A interval. `ru_maxrss` and `VmHWM` are reported and
never gate.

The B preparation step requires exactly one native safe6 counter
entry point before its release build. The B driver enables the
matching compile-time feature and reads the counter before and after
the timed search. Neither A binary is instrumented.

## Analysis

`analyze.py` reads `rows.csv` and `journal.log` and writes
`summary.csv`, `REPORT.md`, and `RESTARTS.md`. It never writes this
file. This registration and the analysis implementation are frozen
in the harness commit before any r6 data is collected. Each bar
has its own verdict. A bar fails if any cell fails it, is
inconclusive if any cell is inconclusive and none fail, and passes
otherwise. The overall verdict gates on hits, outside time, peak
`RssAnon`, and safe path exercise. Bit identity is report-only.

## Log archive

Only selected text logs are archived. Before the archive is made,
machine-local directory prefixes, inherited-environment notices, and
environment values longer than 16 characters are scrubbed from a
staging copy. This includes the registered home, store, software,
temporary, and checkout prefixes, plus machine host names beginning
with `ip-`. A second scan must find none of them. Environment dumps
are not collected. The archive fixes owner and group to numeric zero
and uses numeric ownership. All staged text and CSV output is LF-only;
CSV writers use an explicit newline terminator.
