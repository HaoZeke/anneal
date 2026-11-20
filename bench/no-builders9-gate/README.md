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

One session, core 0, twelve seeds, twenty-eight cells, three arms.
1008 rows. The gate fails. Nine cells fail. Nineteen pass. No cell
fails clause (a). The hit losses are blj115n38 (12 to 9, Fisher
p = 0.108696) and m80r14 (6 to 5, Fisher p = 0.5). Both are reported
and neither is below 0.05.

Hits are A/A'/B. Median calls are A and B, misses ranked as positive
infinity. Outside ratios are B/A and A'/A of the median of
outside seconds over charged calls.

| Cell | Hits | Fisher p | Median calls A, B | MW p | Outside B/A, A'/A | Rule | Verdict |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| lj38 | 12/12/12 | 1 | 127840, 91472.5 | 0.950061 | 1.09403, 0.98674 |  | PASS |
| lj75 | 5/5/10 | 0.995289 | inf, 2.49916e+06 | 0.969122 | 1.15881, 0.993358 | c | FAIL |
| lj98 | 7/7/11 | 0.993135 | 3.16681e+06, 990419 | 0.984326 | 1.20399, 1.00696 | c | FAIL |
| lj102 | 6/6/10 | 0.986405 | inf, 960456 | 0.988229 | 1.11492, 0.992338 | c | FAIL |
| lj104 | 5/5/6 | 0.793175 | inf, inf | 0.681525 | 1.17702, 0.99095 | c | FAIL |
| blj105n38 | 2/2/9 | 0.999678 | inf, 345236 | 0.98249 | 1.04904, 0.988093 |  | PASS |
| blj105n55 | 12/12/12 | 1 | 158055, 311804 | 0.030301 | 2.5689, 0.977445 | b, c | FAIL |
| blj115n38 | 12/12/9 | 0.108696 | 82300.5, 149026 | 0.0783355 | 3.59452, 1.00488 | c | FAIL |
| blj115n55 | 0/0/0 | 1 | inf, inf | 1 | 1.64572, 1.00962 | c | FAIL |
| m55r14 | 9/9/10 | 0.841615 | 451562, 458022 | 0.557632 | 1.08806, 0.99802 |  | PASS |
| m80r6 | 12/12/12 | 1 | 62677, 173708 | 0.646334 | 1.1173, 1.00697 | c | FAIL |
| m80r14 | 6/6/5 | 0.5 | inf, inf | 0.611481 | 1.11114, 1.02132 | c | FAIL |
| sc126-38 | 12/12/12 | 1 | 113504, 113504 | 0.511545 | 1.01092, 1.00934 |  | PASS |
| sc126-55 | 12/12/12 | 1 | 18379.5, 18379.5 | 0.511545 | 1.0152, 1.00595 |  | PASS |
| tip4p8 | 12/12/12 | 1 | 6339, 6339 | 0.511545 | 1.00082, 1.00232 |  | PASS |
| tip4p9 | 12/12/12 | 1 | 5839.5, 5839.5 | 0.511545 | 0.980193, 0.976921 |  | PASS |
| tip4p10 | 12/12/12 | 1 | 13138.5, 13138.5 | 0.511545 | 0.994326, 1.00877 |  | PASS |
| tip4p11 | 12/12/12 | 1 | 109288, 109288 | 0.511545 | 0.989373, 0.988807 |  | PASS |
| tip4p12 | 12/12/12 | 1 | 41418.5, 41418.5 | 0.511545 | 0.995089, 1.00399 |  | PASS |
| tip4p13 | 12/12/12 | 1 | 216690, 216690 | 0.511545 | 1.00893, 1.01689 |  | PASS |
| tip4p14 | 12/12/12 | 1 | 135853, 135853 | 0.511545 | 1.00488, 0.991282 |  | PASS |
| tip4p15 | 7/7/7 | 0.659914 | 1.48162e+06, 1.48162e+06 | 0.511971 | 0.961107, 0.974864 |  | PASS |
| tip4p16 | 2/2/2 | 0.704969 | inf, inf | 0.517746 | 1.01227, 1.00029 |  | PASS |
| tip4p17 | 5/5/5 | 0.659914 | inf, inf | 0.512874 | 1.00974, 1.00451 |  | PASS |
| tip4p18 | 2/2/2 | 0.704969 | inf, inf | 0.517746 | 1.01121, 1.0118 |  | PASS |
| tip4p19 | 7/7/7 | 0.659914 | 1.06084e+06, 1.06084e+06 | 0.511971 | 1.00997, 0.986032 |  | PASS |
| tip4p20 | 6/6/6 | 0.657864 | inf, inf | 0.512323 | 0.99464, 0.99251 |  | PASS |
| tip4p21 | 0/0/0 | 1 | inf, inf | 1 | 0.985267, 0.993876 |  | PASS |

Clause (b) fails blj105n55 only: one-sided Mann-Whitney p = 0.030301
and the median-call ratio is 1.97276, above the A'/A ratio of 1.
Clause (c) fails the nine cells marked c. On each of them A'/A is
inside [0.95, 1.05] and B/A is above 1.10.

Author counts. Confirmed: blj115n38 12 to 9, m80r14 6 to 5, lj75 5
to 10, lj98 7 to 11, blj105n38 2 to 9. Not confirmed: blj105n55 8
to 3, measured 12 to 12. Outside time above 1.10 on most cells is
not confirmed: 9 of 28 cells.

The blj105n38 count gap between the two published drivers is in
`harness-diff.md`. Rows for that measurement are
`discrepancy-rows.csv`.
