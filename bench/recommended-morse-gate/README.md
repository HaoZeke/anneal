# Recommended Morse gate

A/B of `Config::recommended` on the four Cambridge Morse systems in
`scripts/ase_nonlj/run_one.py`: `m38r6`, `m38r14`, `m55r6`, `m55r14`.
Seeds 2000-2011. Budgets are the paper budgets in that harness (400,000
at 38 atoms, 1,000,000 at 55). The wall cap is three times the slowest
`for_cluster` wall of revision `9a30f3190b3bb7040c675a904f8388eb68224151`
in `bench/ase-nonlj/rows.csv`, which is the revision `enter_bench.sh` reads:

| system | budget | cap s |
| --- | ---: | ---: |
| m38r6 | 400000 | 10.465089 |
| m38r14 | 400000 | 11.142690 |
| m55r6 | 1000000 | 15.803787 |
| m55r14 | 1000000 | 54.130152 |

Arm A is the library that produced the Morse ASE panel. Those rows cite
`90b88235fe54abace3bde32789582eb6573ed40a`. That object is not in this
repository and not on the remote. The build used here is
`7fda9e44af99a88088c07f712e90281a9bff2728`, the parent of `c3bcdc01`, the
commit that first wrote the citing rows. `hosts.csv` records the revision,
tree, and binary that actually ran. Arm B is `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`.

Each arm is a separate release build of `examples/ase_nonlj` (`cargo +1.88.0
build --release --locked --example ase_nonlj`). The Python driver is the
develop copy of `scripts/ase_nonlj/run_one.py`. If arm A cannot run that
driver, the driver from arm B is used against arm A's library and `hosts.csv`
says so.

## Schedule

Serial, one process at a time, pinned to cpu 0, every thread variable set
to 1. No warm-up and no quick budget. For each seed the arms alternate
ABBA: even offsets run the left arm then the right arm, odd offsets run
the right arm then the left arm. `run_seq` counts every process.

- `AB`: A against B on all four systems.
- `AA`: a second release build of B against the first, on `m38r14` and `m55r6` only.
- `PL`: three layouts of B on those two systems. Only the size and padding
  of `Config` change (`end64`, `start256`, `align1024` in `apply_layout.py`).
  The pad is zero, skipped by serde, and kept by `std::hint::black_box`.
  The layout runs are rotated by seed and pooled with the second build.

## Tests

Hits: two-sided Fisher exact test. Evals to hit and wall time to hit: the
hitting runs only, two-sided exact Mann-Whitney. Medians are the ordinary
median (Hyndman-Fan type 7 at one half). The four systems are one family,
so each A/B p-value is also reported as `min(1, 4p)`.

B loses a cell when it has fewer hits, or a higher median eval count on
hitting runs, and the adjusted p-value is below 0.05. Wall time is
reported and is not part of that rule. A loss on `m38r14` is followed by
`git bisect run` on a short quench-energy prefix: first parent from arm A
to arm B, then the side of that merge when the first bad commit has two
parents. The bisect script restores the example after each step. The
scored binaries are not patched.

Bit identity is equality of the harness record: hit, evals to hit,
charged calls, and the printed final energy. The unchanged harness does
not emit the quench sequence. The bisect prefix does, from
`Config::energy_trace`.

## Run

Build each worktree with `build_arm.sh WORKTREE TARGET_DIR`. Apply one
layout with `apply_layout.py KIND WORKTREE` before building that layout.
Then:

```bash
export LJOS_SEAT=benchbot
export GATE_A_BIN=... GATE_A_REV=... GATE_A_TREE=... GATE_A_WORKTREE=...
export GATE_B_BIN=... GATE_B_REV=... GATE_B_TREE=... GATE_B_WORKTREE=...
export GATE_B2_BIN=... GATE_B2_WORKTREE=...
export GATE_L1_BIN=... GATE_L1_WORKTREE=...
export GATE_L2_BIN=... GATE_L2_WORKTREE=...
export GATE_L3_BIN=... GATE_L3_WORKTREE=...
python3 -m snakemake -s bench/recommended-morse-gate/Snakefile --cores 1
```

`hosts.csv` carries the first 12 hex digits of the SHA-256 of the
machine-id file with its trailing newline removed. Logs stay out of git.

## Result

See `verdict.txt`. B loses `m38r14` on the median eval count (adjusted
p below 0.05). Hits on that cell are 12/12 against 9/12 and are not
significant after the same adjustment. The other three cells do not
lose. The second build and the three layouts match B on the harness
record.

The missing object `90b88235` was not built. Arm A is `7fda9e44`.
Its `m38r6` seed 2000 row matches the recorded panel row. `git bisect
run` on seeds 2001 and 2004, first parent then the side of the merge,
finds `1c1b1238` (`fix: remove structure templates from cluster
search`), which `bc16d35c` merges. The quench prefix matches arm A
through step 18 on seed 2001 and step 25 on seed 2004.
