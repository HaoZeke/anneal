# basin-bound memory gate

Three builds, one session, one core. This file states the rule before
any timed row is recorded. The change under test is a performance and
memory-layout change, so search trajectories must not move.

- Arm A is tree `15032ef401fb681e230d27c7b315d89f7ac53e4e`.
- Arm A2 is a separate release rebuild of the A tree in a separate
  target directory. A2 against A is the A/A band.
- Arm B is the four-change tree
  `f2786b12dc42dab75994a0380e10bd3ff4ba0dc7`. No other tree may
  stand in for B.

Every arm is a locked release build with `rgpot-ex`, using the same
compiler, flags and benchmark-only measurement overlay. Each arm has
its own detached worktree and target directory. The overlay changes
measurement only: it samples memory in the benchmark process and
counts entries to the fixed-ring bound. Its source hash and every
binary hash are journalled.

## Harness

`run_gate.py` extends the earlier three-arm gate. It records elapsed,
user and system time and `ru_maxrss` from the child rusage. The
driver's own clocks supply wall, potential and outside time.

The gate memory reading is the maximum RssAnon observed inside the
benchmark process. The process reads its own status after every basin
store and at search end. A sampler thread also reads it every 10 ms.
The driver stops and joins that sampler after capturing `wall_s`, so
sampler shutdown does not enter wall or outside time. The parent
reads child status only after confirming execution of the benchmark
binary, polling RssAnon and VmHWM every 5 ms. Both parent readings are
cross-checks only. Missing parent RssAnon or either VmHWM reading
never makes a row inconclusive. `ru_maxrss` is also diagnostic. Every
row whose internal and external RssAnon peaks differ by more than
2 MiB is reported.

Each row carries status and exit code. A non-zero exit, timeout,
unparsable output or pinning fault is inconclusive. Rows are
append-only. A superseded row is marked as such and remains present.
The campaign refuses to start on an unrecognised non-empty row file.
Each run log is written as `.tmp`, synced, renamed, and read back
before its sha256 is accepted.

## Protocol

Seeds 2000 through 2023, twenty-four seeds. Each seed has two ABBA
blocks: `(A, B, B, A)` measures B against A, then `(A, A2, A2, A)`
measures the rebuild band. For seed offset k, the eight labelled
observations are rotated left by k modulo eight. Rotation, slot and
observation label are recorded in every row. Each label occupies each
slot three times. The eight observations of a seed run together, then
the next seed. Cells run in table order. No quick mode, warm-up or
other timed work shares the session.

The harness and every driver are pinned to core 0 with `taskset -c
0`. Every `*_NUM_THREADS` variable is `1` and the child environment is
otherwise limited to the harness allow-list.

Budgets are the paper list: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and
above. TIP4P uses 2e6 for every size from 8 through 21 and stops at
the reference plus 0.001.

| Cell | Driver | Budget | Reference |
| --- | --- | ---: | ---: |
| lj38 | h2h_gmin_proof | 400000 | -173.928427 |
| lj75 | h2h_gmin_proof | 4000000 | -397.492331 |
| lj98 | h2h_gmin_proof | 4000000 | -543.665361 |
| lj102 | h2h_gmin_proof | 4000000 | -569.363652 |
| lj104 | h2h_gmin_proof | 4000000 | -582.086642 |
| blj105n38 | h2h_blj | 400000 | -177.2090846748 |
| blj105n55 | h2h_blj | 1000000 | -283.1545313810 |
| blj115n38 | h2h_blj | 400000 | -187.3676867453 |
| blj115n55 | h2h_blj | 1000000 | -289.2931853058 |
| m55r6 | h2h_morse | 1000000 | -250.286609 |
| m55r14 | h2h_morse | 1000000 | -220.646208 |
| m80r6 | h2h_morse | 4000000 | -378.333471 |
| m80r14 | h2h_morse | 4000000 | -340.811371 |
| sc126-38 | h2h_sc | 400000 | -35419.9804 |
| sc126-55 | h2h_sc | 1000000 | -52884.6806 |
| tip4p8 .. tip4p21 | h2h_seed | 2000000 | data/tip4p_ccd.csv |

The budget probe checks every non-water driver's budget and reference
on every arm before the preflight.

## Rule

A hit is re-derived from `best` against the cell reference at
tolerance 0.001 (`best` finite and at or below `reference + 0.001`),
never read from the `hit` column. A miss, a non-finite energy or an
inconclusive row is a miss.

A cell passes (`PASS`) only when all five clauses hold. It fails
(`FAIL`) if any clause fails. A cell with any inconclusive row among
its 192 observations is `INCONCLUSIVE` and cannot pass. Clauses are
still computed from seeds whose eight observations are conclusive.

(a) Trajectory identity. On every seed all seven other observations
equal the first A row in `evals`, `charged`, `calls_to_hit` and
bit-identical `best_bits`. Any difference fails the cell.

(b) No hit loss. Neither B observation may miss where A hits, and
neither B hit count may be below A's.

(c) Time outside energy calls per call, `outside_s / max(1,
charged)`. The upper edge of the B/A bootstrap 95% confidence
interval of the ratio of type-7 medians (10,000 paired seed
resamples, fixed seed `20261121`) is at most `max(1.10, U)`, where
`U` is the upper edge of the A2/A interval. Every arm contributes the
midpoint of its two observations in its ABBA block, so each interval
has 24 paired units.

(d) Peak RssAnon. Every arm contributes the maximum of its two
observations in its ABBA block. The B/A point ratio of type-7 medians
must be at most `max(1.05, U)`, where `U` is the upper edge of the
A2/A bootstrap 95% interval. Both intervals use the fixed seed and
10,000 paired-seed draws. The B/A interval is reported but its point
ratio, not its upper edge, is the preregistered gate.

(e) The fixed-ring path counter is zero in every A and A2 row and
nonzero when summed over a cell's B rows. B increments it when the
bounded lookup path is entered, before scalar-key filtering. A fast
0-to-3-hop hit may legitimately have a zero row counter, so the
cell sum proves execution without rejecting that trajectory.

Every row's trajectory and outcome must therefore equal A, no hit may
be lost, outside time must clear its preregistered band, memory must
clear its preregistered band, and the new path must execute. The gate
passes only if every cell passes.

Rows are appended as each arm finishes. `summary.csv`, `hosts.csv` and
the results section below are generated after the last row by
`run_gate.py summarize`.

For blj105n38, the report lists each arm and seed in the approximately
28 MiB and 32 MiB modes and gives a mode-count table. It also lists
all internal/external peak disagreements over 2 MiB. In particular,
seed 2002 is retained even if it repeats the earlier 1.5 versus
22.9 MiB diagnostic; it is not relabelled or discarded.

## Results

This section is generated from `rows.csv` by `run_gate.py summarize`.

29 cells, twenty-four seeds, eight observations, 473 rows, 27 inconclusive. Gate verdict: **INCONCLUSIVE**. 2 cells pass, 0 fail and 27 are inconclusive.

- Clause (a), trajectory identity, fails on: none.
- Clause (b), hit loss, fails on: none.
- Clause (c), outside time per call, fails on: none.
- Clause (d), peak RssAnon, fails on: none.
- Clause (e), fixed-ring execution, fails on: none.

`Traj` counts B observations and A2 observations that match A.
Hits are A, A2, B1 and B2, re-derived from `best`. Outside is the median time outside
energy calls per charged call. Its B/A ratio carries the bootstrap
95% interval, and the limit is the clause (c) bound. RssAnon carries
the paired-seed bootstrap interval. VmHWM and `ru_maxrss` never gate.

| Cell | Identity B, A2 | Hits A, A2, B1, B2 | Outside B/A [95% CI] | Outside A2/A [95% CI] | Outside limit | RssAnon B/A [95% CI] | RssAnon A2/A [95% CI] | RSS limit | Bound A, A2, B sum | Verdict |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| lj38 | 48, 48 | 23, 23, 23, 23 | 0.95681 [0.955351, 0.964258] | 0.996125 [0.993897, 1.00124] | 1.1 | 1.0067 [0.998737, 1.01166] | 0.998885 [0.998686, 1.00126] | 1.05 | 0, 0, 74248 | PASS |
| lj75 | 48, 48 | 7, 7, 7, 7 | 0.934832 [0.923506, 0.94898] | 1.00101 [0.988502, 1.01186] | 1.1 | 1.00249 [1.00109, 1.00326] | 1 [0.999069, 1.00074] | 1.05 | 0, 0, 1420120 | PASS |
| lj98 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| lj102 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| lj104 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| blj105n38 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| blj105n55 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| blj115n38 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| blj115n55 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| m55r6 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| m55r14 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| m80r6 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| m80r14 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| sc126-38 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| sc126-55 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p8 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p9 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p10 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p11 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p12 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p13 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p14 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p15 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p16 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p17 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p18 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p19 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p20 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |
| tip4p21 | nan, nan | nan, nan, nan, nan | nan [nan, nan] | nan [nan, nan] | 1.1 | nan [nan, nan] | nan [nan, nan] | 1.05 | 0, 0, 0 | INCONCLUSIVE |

### Trajectory differences

None in complete cells. Incomplete cells were not compared.

### Inconclusive rows

- lj98: incomplete panel (89 of 192 rows)
- lj102: incomplete panel (0 of 192 rows)
- lj104: incomplete panel (0 of 192 rows)
- blj105n38: incomplete panel (0 of 192 rows)
- blj105n55: incomplete panel (0 of 192 rows)
- blj115n38: incomplete panel (0 of 192 rows)
- blj115n55: incomplete panel (0 of 192 rows)
- m55r6: incomplete panel (0 of 192 rows)
- m55r14: incomplete panel (0 of 192 rows)
- m80r6: incomplete panel (0 of 192 rows)
- m80r14: incomplete panel (0 of 192 rows)
- sc126-38: incomplete panel (0 of 192 rows)
- sc126-55: incomplete panel (0 of 192 rows)
- tip4p8: incomplete panel (0 of 192 rows)
- tip4p9: incomplete panel (0 of 192 rows)
- tip4p10: incomplete panel (0 of 192 rows)
- tip4p11: incomplete panel (0 of 192 rows)
- tip4p12: incomplete panel (0 of 192 rows)
- tip4p13: incomplete panel (0 of 192 rows)
- tip4p14: incomplete panel (0 of 192 rows)
- tip4p15: incomplete panel (0 of 192 rows)
- tip4p16: incomplete panel (0 of 192 rows)
- tip4p17: incomplete panel (0 of 192 rows)
- tip4p18: incomplete panel (0 of 192 rows)
- tip4p19: incomplete panel (0 of 192 rows)
- tip4p20: incomplete panel (0 of 192 rows)
- tip4p21: incomplete panel (0 of 192 rows)

### Internal and external peak disagreements

None over 2 MiB.

### blj105n38 RssAnon modes

| Seed | A for B/A | B | A for A2/A | A2 |
| ---: | --- | --- | --- | --- |
| 2000 | not run | not run | not run | not run |
| 2001 | not run | not run | not run | not run |
| 2002 | not run | not run | not run | not run |
| 2003 | not run | not run | not run | not run |
| 2004 | not run | not run | not run | not run |
| 2005 | not run | not run | not run | not run |
| 2006 | not run | not run | not run | not run |
| 2007 | not run | not run | not run | not run |
| 2008 | not run | not run | not run | not run |
| 2009 | not run | not run | not run | not run |
| 2010 | not run | not run | not run | not run |
| 2011 | not run | not run | not run | not run |
| 2012 | not run | not run | not run | not run |
| 2013 | not run | not run | not run | not run |
| 2014 | not run | not run | not run | not run |
| 2015 | not run | not run | not run | not run |
| 2016 | not run | not run | not run | not run |
| 2017 | not run | not run | not run | not run |
| 2018 | not run | not run | not run | not run |
| 2019 | not run | not run | not run | not run |
| 2020 | not run | not run | not run | not run |
| 2021 | not run | not run | not run | not run |
| 2022 | not run | not run | not run | not run |
| 2023 | not run | not run | not run | not run |

| Arm | ~28 MiB | ~32 MiB | Other |
| --- | ---: | ---: | ---: |
| A for B/A | 0 | 0 | 0 |
| B | 0 | 0 | 0 |
| A for A2/A | 0 | 0 | 0 |
| A2 | 0 | 0 | 0 |
