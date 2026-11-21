# basin-bound gate

Three arms, one session, one core. This file states the rule before
any timed row is recorded. The change under test is a pure
performance change, so search trajectories must not move.

- Arm A is develop `9837358ae1027067ff654fac8b48969020853452`, tree
  `15032ef401fb681e230d27c7b315d89f7ac53e4e`.
- Arm A2 is the same commit, built a second time from a second
  worktree into a second target directory. A2 against A is the A/A
  band.
- Arm B is A plus two commits: the block-sum bound that skips basin
  centres (`perf(bias): skip basin centres on a block-sum bound`) and
  one sine and cosine per rotation in symmetrise
  (`perf(symmetrise): take one sine and cosine per rotation`). B is
  `41a4a07d7a4b6e8b5dc50fc74fee4080b66ebf1b`, tree
  `3b430f17ada0387c2ec1ae00e2ca45b8f7abc89f`. The author's own bench
  commit is not part of B.

Release build, rustc 1.88.0, `cargo build --release --locked
--features rgpot-ex`, the same toolchain and flags in every arm. Each
arm has its own worktree and target directory. The five drivers are
the same source bytes on every arm. `h2h_gmin_proof`, `h2h_seed` and
`h2h_sc` are the develop sources. `driver-src/h2h_morse.rs` and
`driver-src/h2h_blj.rs` are the bindings-parity gate copies, placed in
`examples/` of every arm unmodified, so the binaries differ only by
the library they link.

## Harness

`run_gate.py` is the bindings-parity gate harness with the arms
renamed to A, A2 and B. Other changes from that harness, all required
by this gate's protocol:

- m55r6 is added, the one cell of the basin-bound author's bench that
  the panel lacked. m80r14 was already in the panel.
- The four clauses below replace the parity gate's three.
- The harness records its own timing from `os.wait4`, which returns
  the `resource` rusage of the child: `elapsed_s` (monotonic clock),
  `ru_utime_s`, `ru_stime_s` and `ru_maxrss_kb`. No external `time`
  binary is used. `wall_s` and `pot_s` are the driver's in-process
  clocks, and the time outside energy calls is the driver's
  `outside_s`, or `wall_s - pot_s` where the driver does not print it.
- Each row carries `status` and `exit_code`. A non-zero exit, a
  timeout, unparsable output or a pinning fault is recorded as an
  `inconclusive` row. It is never rerun or replaced, and `rows.csv`
  is append-only. The campaign refuses to start on a non-empty
  `rows.csv`.
- `evals` is the driver's `evals_to_hit` whether or not the run hit,
  so energy calls are compared on misses too.
- Before the first row, a preflight runs one real cell and seed per
  arm and driver through the production command (seed 2000 on lj38,
  blj115n38, m55r6, sc126-55 and tip4p8). It aborts the campaign on a
  non-zero exit or an empty timing or RssAnon field. Preflight runs
  are not rows. Their outcomes go to `sessions.jsonl`.
- `sessions.jsonl` journals the session start (arm commits, trees,
  binary sha256, compiler, flags, thread variables), every preflight
  run and the session end.
- Each run log is written as `.tmp`, synced and renamed, and its
  read-back sha256 is the row's `log_sha256`.
- The TIP4P start-file comparison across arms is dropped. Clause (a)
  compares the trajectories themselves.

## Protocol

Seeds 2000 through 2011, twelve seeds. For seed offset k, the order
(A, A2, B) is rotated left by k mod 3, and an odd k reverses that
rotation. Each arm occupies each of the three slots four times. The
three arms of a seed run in that order, then the next seed. Cells run
one after another in the table order below. No quick mode, no
warm-up, and no other timed work shares the session.

The harness and every driver are pinned to core 0 with `taskset -c
0`. Every `*_NUM_THREADS` variable is `1` and the child environment is
otherwise limited to the harness allow-list. Each run logs `taskset
-p` and `Cpus_allowed_list`, and samples peak `RssAnon` from
`/proc/<pid>/status`.

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

A cell passes (`PASS`) only when all four clauses hold. It fails
(`FAIL`) if any clause fails. A cell with any inconclusive row among
its 36 is `INCONCLUSIVE` and cannot pass. Its clauses are still
computed, and (c) and (d) then use the seeds whose three rows are all
conclusive.

(a) Trajectory identity. On every seed the B row equals the A row in
`evals` (energy calls), `charged` and `calls_to_hit`, and `best_bits`
is bit-identical. Any difference fails the cell. Every difference of
B or A2 against A is listed per cell and seed. A2 differences are
reported and do not gate.

(b) No hit loss. No seed where A hits and B misses, and B's hit count
is not below A's.

(c) Time outside energy calls per call, `outside_s / max(1,
charged)`. The upper edge of the B/A bootstrap 95% confidence
interval of the ratio of type-7 medians (10,000 paired seed
resamples, seed `20261121`) is at most `max(1.10, U)`, where `U` is
the upper edge of the A2/A interval computed the same way.

(d) Peak `RssAnon`. The B/A ratio of type-7 medians is at most
`max(1.05, A2/A)`, with A2/A the same ratio for A2. `ru_maxrss` is
reported per cell and never gates.

Also reported per cell: the outside time per call of each arm and its
B/A and A2/A median ratios, so the claimed speedup can be checked, and
wall-clock (`elapsed_s`) B/A and A2/A median ratios. The gate passes
only if every cell passes.

Rows are appended as each arm finishes. `summary.csv`, `hosts.csv` and
the results section below are generated after the last row by
`run_gate.py summarize`.

## Results

Not yet run.
