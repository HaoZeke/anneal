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

29 cells, twenty-four seeds, eight observations, 5568 rows, 0 inconclusive. Gate verdict: **PASS**. 29 cells pass, 0 fail and 0 are inconclusive.

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
| lj98 | 48, 48 | 16, 16, 16, 16 | 0.93787 [0.935145, 0.947883] | 1.00191 [0.99767, 1.00693] | 1.1 | 1.00356 [0.999842, 1.00645] | 1.00021 [0.999118, 1.00127] | 1.05 | 0, 0, 990870 | PASS |
| lj102 | 48, 48 | 11, 11, 11, 11 | 0.940321 [0.928975, 0.947913] | 0.999359 [0.996333, 1.00978] | 1.1 | 1.00203 [1.00029, 1.00448] | 0.998959 [0.997815, 1.00178] | 1.05 | 0, 0, 1320320 | PASS |
| lj104 | 48, 48 | 11, 11, 11, 11 | 0.935139 [0.932869, 0.946007] | 0.999572 [0.992892, 1.00303] | 1.1 | 1.00198 [1.00065, 1.00351] | 1.0009 [0.997456, 1.00173] | 1.05 | 0, 0, 1243522 | PASS |
| blj105n38 | 48, 48 | 3, 3, 3, 3 | 1.00434 [1.00171, 1.01433] | 0.999408 [0.99517, 1.01393] | 1.1 | 0.938705 [0.883335, 1.11496] | 1 [0.999934, 1.00013] | 1.05 | 0, 0, 60512 | PASS |
| blj105n55 | 48, 48 | 24, 24, 24, 24 | 1.00589 [0.983701, 1.03891] | 0.997229 [0.990087, 1.00448] | 1.1 | 1.00023 [0.98181, 1.13123] | 1 [0.999871, 1.00008] | 1.05 | 0, 0, 38848 | PASS |
| blj115n38 | 48, 48 | 23, 23, 23, 23 | 1.01753 [1.01007, 1.03468] | 0.999177 [0.997822, 1.02729] | 1.1 | 1.00768 [1.00018, 1.01638] | 0.999918 [0.999832, 1] | 1.05 | 0, 0, 22600 | PASS |
| blj115n55 | 48, 48 | 0, 0, 0, 0 | 1.00088 [0.985578, 1.01217] | 0.998582 [0.991157, 1.00213] | 1.1 | 0.998846 [0.911094, 1.04225] | 1.00003 [0.999897, 1.00005] | 1.05 | 0, 0, 552326 | PASS |
| m55r6 | 48, 48 | 24, 24, 24, 24 | 0.964158 [0.957832, 0.973685] | 1.00218 [0.99715, 1.0027] | 1.1 | 1 [0.991853, 1.01879] | 1 [0.989754, 1.00204] | 1.05 | 0, 0, 13964 | PASS |
| m55r14 | 48, 48 | 18, 18, 18, 18 | 0.960229 [0.944167, 0.971419] | 0.992702 [0.989918, 1.00293] | 1.1 | 1.01034 [1.00652, 1.01436] | 1 [0.998522, 1.00151] | 1.05 | 0, 0, 326704 | PASS |
| m80r6 | 48, 48 | 24, 24, 24, 24 | 0.940541 [0.920525, 0.971504] | 1.00157 [0.995804, 1.00295] | 1.1 | 1.01075 [1.00543, 1.01525] | 1.00077 [0.999083, 1.00147] | 1.05 | 0, 0, 429612 | PASS |
| m80r14 | 48, 48 | 15, 15, 15, 15 | 0.938289 [0.930167, 0.951197] | 1.00436 [0.995017, 1.0071] | 1.1 | 1.0016 [1.00114, 1.00255] | 0.999454 [0.999033, 1.00096] | 1.05 | 0, 0, 1950818 | PASS |
| sc126-38 | 48, 48 | 24, 24, 24, 24 | 1.00637 [0.998863, 1.01299] | 0.997537 [0.976729, 1.00312] | 1.1 | 1.00102 [0.994733, 1.00885] | 1.00102 [0.997788, 1.00216] | 1.05 | 0, 0, 72582 | PASS |
| sc126-55 | 48, 48 | 24, 24, 24, 24 | 1.01674 [1.002, 1.02352] | 1.00131 [0.995656, 1.00961] | 1.1 | 1.00487 [0.991014, 1.01485] | 1 [0.997768, 1] | 1.05 | 0, 0, 32726 | PASS |
| tip4p8 | 48, 48 | 24, 24, 24, 24 | 1.00392 [0.994561, 1.02214] | 1.00047 [0.987278, 1.01409] | 1.1 | 1.00912 [0.993711, 1.0302] | 1.00303 [0.993421, 1.01266] | 1.05 | 0, 0, 9732 | PASS |
| tip4p9 | 48, 48 | 24, 24, 24, 24 | 1.02681 [1.01463, 1.03765] | 0.999778 [0.990493, 1.0041] | 1.1 | 1.01114 [0.98995, 1.02041] | 0.99446 [0.993865, 1.00641] | 1.05 | 0, 0, 10644 | PASS |
| tip4p10 | 48, 48 | 24, 24, 24, 24 | 1.04094 [1.02837, 1.05264] | 0.992421 [0.987966, 1.00533] | 1.1 | 1.01278 [1, 1.02] | 1.00319 [0.992701, 1.00347] | 1.05 | 0, 0, 25020 | PASS |
| tip4p11 | 48, 48 | 24, 24, 24, 24 | 1.03802 [1.03391, 1.04706] | 1.00286 [0.997643, 1.00892] | 1.1 | 1.01564 [1.01209, 1.02037] | 0.999588 [0.998912, 1.00094] | 1.05 | 0, 0, 148986 | PASS |
| tip4p12 | 48, 48 | 24, 24, 24, 24 | 1.04354 [1.0347, 1.04789] | 1.00322 [0.989557, 1.01255] | 1.1 | 1.01371 [1.00973, 1.02033] | 0.999404 [0.998504, 1] | 1.05 | 0, 0, 98762 | PASS |
| tip4p13 | 48, 48 | 24, 24, 24, 24 | 1.04053 [1.03586, 1.04568] | 1.00169 [0.995808, 1.00458] | 1.1 | 1.0108 [1.00783, 1.01538] | 1.00059 [0.999867, 1.00131] | 1.05 | 0, 0, 196670 | PASS |
| tip4p14 | 48, 48 | 24, 24, 24, 24 | 1.046 [1.03827, 1.05368] | 1.00534 [0.997977, 1.00831] | 1.1 | 1.01392 [1.01046, 1.01995] | 0.999792 [0.998746, 1.00017] | 1.05 | 0, 0, 146282 | PASS |
| tip4p15 | 48, 48 | 15, 15, 15, 15 | 1.04444 [1.03684, 1.04876] | 1.0017 [0.995866, 1.01784] | 1.1 | 1.00746 [1.00292, 1.01279] | 1.00007 [0.999823, 1.00021] | 1.05 | 0, 0, 887522 | PASS |
| tip4p16 | 48, 48 | 7, 7, 7, 7 | 1.03825 [1.03114, 1.05344] | 1.00395 [0.995723, 1.00514] | 1.1 | 1.00769 [1.00242, 1.00957] | 0.999939 [0.999743, 1.00011] | 1.05 | 0, 0, 1264404 | PASS |
| tip4p17 | 48, 48 | 9, 9, 9, 9 | 1.04009 [1.02864, 1.04367] | 1.00209 [0.992741, 1.0045] | 1.1 | 1.00306 [1.00221, 1.00802] | 1.00002 [0.999845, 1.00018] | 1.05 | 0, 0, 1046088 | PASS |
| tip4p18 | 48, 48 | 5, 5, 5, 5 | 1.03738 [1.03074, 1.04407] | 1.00119 [0.996692, 1.00308] | 1.1 | 1.00601 [1.0018, 1.00773] | 0.999963 [0.999921, 1.00018] | 1.05 | 0, 0, 1288888 | PASS |
| tip4p19 | 48, 48 | 12, 12, 12, 12 | 1.03462 [1.02795, 1.04435] | 0.998372 [0.997096, 1.00963] | 1.1 | 1.00623 [1.00225, 1.00828] | 1.0001 [0.999553, 1.00017] | 1.05 | 0, 0, 1059852 | PASS |
| tip4p20 | 48, 48 | 6, 6, 6, 6 | 1.04517 [1.0321, 1.05326] | 1.00156 [0.988075, 1.00485] | 1.1 | 1.00503 [1.00283, 1.00846] | 0.999984 [0.999906, 1.00006] | 1.05 | 0, 0, 1274852 | PASS |
| tip4p21 | 48, 48 | 1, 1, 1, 1 | 1.03996 [1.01522, 1.04566] | 1.00045 [0.998864, 1.00609] | 1.1 | 1.0054 [1.00295, 1.00645] | 1.00005 [0.999913, 1.00018] | 1.05 | 0, 0, 1506276 | PASS |

### Trajectory differences

None. On all 29 cells and 24 seeds, both B observations and A2 match A.

### Inconclusive rows

None. Every run exited 0 with timing and RssAnon recorded.

### Internal and external peak disagreements

None over 2 MiB.

### blj105n38 RssAnon modes

| Seed | A for B/A | B | A for A2/A | A2 |
| ---: | --- | --- | --- | --- |
| 2000 | ~32 (31.4) | ~28 (27.3) | ~32 (31.4) | ~32 (31.4) |
| 2001 | ~28 (27.4) | ~28 (27.4) | ~28 (27.4) | ~28 (27.4) |
| 2002 | other (22.4) | other (22.4) | other (22.4) | other (22.4) |
| 2003 | ~32 (30.9) | ~32 (30.8) | ~32 (30.9) | ~32 (30.9) |
| 2004 | ~28 (26.7) | ~28 (26.8) | ~28 (26.7) | ~28 (26.7) |
| 2005 | ~28 (27.4) | ~28 (27.5) | ~28 (27.4) | ~28 (27.4) |
| 2006 | ~28 (27.5) | ~28 (27.5) | ~28 (27.5) | ~28 (27.5) |
| 2007 | ~28 (27.3) | ~28 (27.3) | ~28 (27.3) | ~28 (27.3) |
| 2008 | other (22.9) | other (22.9) | other (22.9) | other (23.0) |
| 2009 | ~28 (27.0) | ~32 (31.0) | ~28 (27.0) | ~28 (27.0) |
| 2010 | ~28 (27.0) | ~28 (27.1) | ~28 (27.0) | ~28 (27.0) |
| 2011 | ~32 (31.4) | ~28 (27.5) | ~32 (31.4) | ~32 (31.4) |
| 2012 | ~32 (31.4) | ~32 (31.2) | ~32 (31.4) | ~32 (31.4) |
| 2013 | ~32 (31.3) | ~32 (31.1) | ~32 (31.3) | ~32 (31.3) |
| 2014 | ~28 (27.2) | ~28 (27.2) | ~28 (27.2) | ~28 (27.2) |
| 2015 | ~32 (31.2) | ~32 (31.2) | ~32 (31.2) | ~32 (31.2) |
| 2016 | ~32 (31.2) | ~32 (31.1) | ~32 (31.2) | ~32 (31.2) |
| 2017 | ~28 (27.7) | ~32 (31.4) | ~28 (27.7) | ~28 (27.7) |
| 2018 | ~32 (31.0) | ~32 (30.8) | ~32 (31.0) | ~32 (31.0) |
| 2019 | ~32 (31.5) | ~32 (30.9) | ~32 (31.5) | ~32 (31.5) |
| 2020 | ~32 (30.9) | ~32 (31.3) | ~32 (30.9) | ~32 (30.9) |
| 2021 | ~32 (31.7) | ~28 (27.2) | ~32 (31.7) | ~32 (31.7) |
| 2022 | ~32 (31.1) | ~32 (31.4) | ~32 (31.1) | ~32 (31.1) |
| 2023 | ~28 (27.4) | ~28 (27.4) | ~28 (27.4) | ~28 (27.4) |

| Arm | ~28 MiB | ~32 MiB | Other |
| --- | ---: | ---: | ---: |
| A for B/A | 10 | 12 | 2 |
| B | 11 | 11 | 2 |
| A for A2/A | 10 | 12 | 2 |
| A2 | 10 | 12 | 2 |
