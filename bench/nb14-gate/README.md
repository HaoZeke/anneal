# no-builders14 same-runner gate

This file declares the gate before any build or measured run.

## Revisions and arms

NB14 is the 33-patch series applied to develop
`61fa67cf380e46876b569da45e7cc091647a775b`. Its tree is
`5d21b6f92713b82b421fb44f2b5edd932619819b`.

| Arm | Source | Configuration |
| --- | --- | --- |
| A | develop `61fa67cf` | defaults |
| A2 | an independent release rebuild of develop `61fa67cf` | defaults |
| B | NB14 | defaults |
| G | NB14 | `H2H_COMPLETION=fitted` |

Every other variable beginning with `H2H_` or `ANNEAL_` is removed.
The quoted G setting is exactly `H2H_COMPLETION=fitted`.

## Panel

The 24 non-water paper cells use their existing driver references and
`PAPER_BUDGETS`.

- LJ: lj38, lj55, lj75, lj98.
- Binary LJ: blj105n38, blj115n38, blj130n38, blj105n55,
  blj115n55, blj130n55, blj130n98.
- Morse: m55r6, m55r14, m80r6, m80r14.
- Sutton-Chen: sc126-38, sc108-38, sc96-38, sc126-55,
  sc108-55, sc96-55, sc126-80, sc108-80, sc96-80.

Each driver must report the declared budget and reference before the
first measured row. A disagreement makes that cell inconclusive.

## Protocol

All four arms are separate release builds. A and A2 are separate builds
of the same source. The runs use one machine, one process at a time,
`taskset -c 0`, and every recognized thread-count variable set to 1.
There is no quick mode, warm-up, retuning, or wall-time limit.

Seeds are 2000 through 2023. For each cell and seed, the measured order
is A, B, G, G, B, A. For the first six seeds only, that block is followed
by A, A2, A2, A. Every invocation produces a distinct append-only row.
No failed or missing invocation is rerun, replaced, or omitted.

An append-only journal records every invocation start and end. A restart
of the orchestrator also appends a journal entry. Partial rows are copied
to the handoff directory after every four completed cells.

Every row records the best energy, final energy, charged calls,
calls-to-hit, outside-potential seconds, total wall seconds, exit code,
arm tree, seed, and a SHA-256 machine identifier. It also records peak
RssAnon, ru_maxrss, and VmHWM. Raw process output is kept outside Git.

Hits are derived again from the best energy. A hit is a finite best
energy at or below that cell's reference plus 0.001. A miss has
calls-to-hit equal to positive infinity for analysis.

## Declared bars

Each bar is evaluated separately for B against A and G against A.

1. Hit bar: the candidate has at least as many derived hits as A in
   every cell. Two-sided Fisher exact p-values and two-sided
   Mann-Whitney U p-values on calls-to-hit are reported. A single lower
   candidate hit count fails the cell regardless of significance.
2. Outside-time bar: divide each row's outside-potential seconds by its
   charged calls, then take the type-7 median. The candidate-to-A median
   ratio must be at most 1.10. A paired seed bootstrap with 10,000
   resamples gives a 95% percentile interval. The first-six-seed A2/A
   ratio and interval form the reported same-source runner band. The
   candidate interval must not exceed the larger of 1.10 and the upper
   edge of that band.

Peak RssAnon is the memory gate. For each candidate, its type-7 median
peak RssAnon must not exceed the larger of 1.10 times A and the upper
edge of the A2/A same-source band. ru_maxrss and VmHWM are reported and
never gate.

A non-zero exit, malformed output, or absent planned row makes that cell
inconclusive for both bars. It is never converted to a miss or replaced.
The overall gate passes only when both candidate arms pass both bars in
all 24 cells and no cell is inconclusive.

The analysis writes `summary.csv` and `REPORT.md`, with one verdict for
each arm, cell, and bar.
