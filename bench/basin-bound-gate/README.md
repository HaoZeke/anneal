# basin-bound memory gate

Three builds, one session, one core. This file states the rule before
any timed row is recorded. The change under test is a performance and
memory-layout change, so search trajectories must not move.

- Arm A is tree `15032ef401fb681e230d27c7b315d89f7ac53e4e`.
- Arm A2 is a separate release rebuild of the A tree in a separate
  target directory. A2 against A is the A/A band.
- Arm B is the four-change tip whose tree starts `f2786b12`. No other
  tip may stand in for B.

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
store and at search end. A sampler thread also reads it every 5 ms.
The parent polls RssAnon every 5 ms only after confirming that the
child has executed the benchmark binary. That external peak is a
cross-check and never gates. VmHWM and `ru_maxrss` are diagnostics.
Every row whose internal and external peaks differ by more than
2 MiB is reported.

Each row carries status and exit code. A non-zero exit, timeout,
unparsable output or pinning fault is inconclusive. Rows are
append-only. A superseded row is marked as such and remains present.
The campaign refuses to start on an unrecognised non-empty row file.
Each run log is written as `.tmp`, synced, renamed, and read back
before its sha256 is accepted.

## Protocol

Seeds 2000 through 2023, twenty-four seeds. Each seed has the four
labelled observations `(A, B1, B2, A2)`. For seed offset k, that ABBA
sequence is rotated left by k modulo four. The rotation, slot and
observation label are recorded in every row. Each labelled
observation occupies each slot six times. The four observations of a
seed run together, then the next seed. Cells run in table order. No
quick mode, warm-up or other timed work shares the session.

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
its 96 observations is `INCONCLUSIVE` and cannot pass. Clauses are
still computed from seeds whose four observations are conclusive.

(a) Trajectory identity. On every seed both B rows and the A2 row
equal A in `evals`, `charged`, `calls_to_hit` and bit-identical
`best_bits`. Any difference fails the cell.

(b) No hit loss. Neither B observation may miss where A hits, and
neither B hit count may be below A's.

(c) Time outside energy calls per call, `outside_s / max(1,
charged)`. The upper edge of the B/A bootstrap 95% confidence
interval of the ratio of type-7 medians (10,000 paired seed
resamples, fixed seed `20261121`) is at most `max(1.10, U)`, where
`U` is the upper edge of the A2/A interval. A seed resample keeps its
two B observations together, so they are not pseudoreplicates.

(d) Peak RssAnon. The B/A ratio of type-7 medians is at most
`max(1.05, U)`, where `U` is the upper edge of the A2/A bootstrap 95%
interval. Both B/A and A2/A use the fixed seed and 10,000
seed-cluster draws. The point ratio and interval are reported.

(e) The fixed-ring bound execution counter is nonzero in every B row
and zero in every A and A2 row.

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

Not yet run.
