# no-builders9 gate

Three arms, one session, one core. This file states the rule before
any timed row is recorded.

Arm A and arm A' are separate release builds of develop
`61fa67cf380e46876b569da45e7cc091647a775b`, tree
`cff796c507a8fa1d0bfb28888c3570b61988055e`. Arm B is that revision
plus the sixteen-patch series, tree
`bedec119403af784a4d939805f5bdca3545f542a`.

Release build, rustc 1.88.0, `--locked`. Each arm has its own
worktree and target directory. The five driver sources are the same
bytes on every arm, so the binaries differ by the library they link.
`h2h_gmin_proof`, `h2h_seed` and `h2h_sc` are the develop sources.
`driver-src/h2h_morse.rs` is the series copy of that driver: develop's
copy assigns a field the series removed, inside a branch this gate
does not set, and that copy does not compile against arm B.
`driver-src/h2h_blj.rs` is the Doye-Meyer driver with the budget
subcommand. Develop does not contain that file. Both copies are the
bytes compiled into every arm.

## Rule

A hit is a finite best energy at or below the reference plus 0.001.
A run that errors or does not finish is a miss. Misses are ranked as
positive infinity in the calls test. `calls_to_hit` is `inf` on a
miss.

B fails a cell when any of the following holds.

(a) B has fewer hits than A and the exact one-sided Fisher test of
that 2 by 2 table has p < 0.05. The p-value is the hypergeometric
probability of B hits at or below the observed count, with both
margins fixed.

(b) Calls to the hit are worse for B. The test is a one-sided
Mann-Whitney test that B's calls are larger than A's, with the
tie-corrected normal approximation and the continuity correction,
misses ranked as positive infinity. It fails the cell when p < 0.05
and the ratio of type-7 median calls, B over A, is greater than 1
and greater than the A'/A median ratio. The upper edge of the A/A
band is the larger of 1 and the A'/A median ratio. Two infinite
medians count as a ratio of 1.

(c) The type-7 median of per-row outside seconds divided by charged
calls, B over A, is greater than 1.10, and the same A'/A ratio lies
inside [0.95, 1.05]. A charged count of zero uses a denominator of
1. If A'/A lies outside that interval, (c) does not fail the cell.

Every hit loss is reported, including when the Fisher p is not below
0.05. A hit loss fails the cell only through (a). The gate fails if
any cell fails.

## Protocol

Seeds 2000 through 2011, twelve seeds. For seed offset k, the order
(A, A', B) is rotated left by k mod 3, and an odd k reverses that
rotation. Each arm occupies each of the three slots four times. The
three arms of a seed run in that order, then the next seed. Cells
run one after another. No other timed work shares the session.

The process is pinned to core 0. Every thread variable in the
harness list is 1. No `H2H_*` variable is set. There is no warm-up
and no quick mode. The budget probe checks each non-water driver
before the first row.

Budgets are the paper list: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75
and above. TIP4P uses the gate-b budget of 2e6 for every size from
8 through 21. References and drivers:

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
| m55r14 | h2h_morse | 1000000 | -220.646208 |
| m80r6 | h2h_morse | 4000000 | -378.333471 |
| m80r14 | h2h_morse | 4000000 | -340.811371 |
| sc126-38 | h2h_sc | 400000 | -35419.9804 |
| sc126-55 | h2h_sc | 1000000 | -52884.6806 |
| tip4p8 .. tip4p21 | h2h_seed | 2000000 | data/tip4p_ccd.csv |

Lennard-Jones, binary Lennard-Jones and Morse references are the
ones those drivers print. Sutton-Chen references are the Cambridge
table in `h2h_sc` and `scripts/metal_h2h/run_sc.py`. TIP4P
references are the energies in `data/tip4p_ccd.csv`. The preset for
the cluster drivers is `Config::recommended`. Binary Lennard-Jones
adds two species labels and no diameters. TIP4P uses
`Config::recommended_tip4p` through `h2h_seed`, and stops at the
reference plus 0.001.

Rows are appended as each arm finishes. `summary.csv` is written
after the last row. The author hit counts below are not a result.
They are checked against the rows after the run.

Unchecked author counts: blj115n38 12 to 9, blj105n55 8 to 3,
m80r14 6 to 5, outside time over 1.10 on most cells, lj75 5 to 10,
lj98 7 to 11, blj105n38 2 to 9.

## Results

Not yet run.
