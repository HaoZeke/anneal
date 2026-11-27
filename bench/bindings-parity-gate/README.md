# bindings-parity gate

Three arms, one session, one core. This file states the rule before
any timed row is recorded.

Arm A and arm A' are separate release builds of develop
`61fa67cf380e46876b569da45e7cc091647a775b`, tree
`cff796c507a8fa1d0bfb28888c3570b61988055e`. Arm B is
`9595acc8813364747157de5acee8512c5cde0a70` (`feat/bindings-parity`,
twenty commits on develop), tree
`e8840116daa00d8d832f5c93938ac8552968d29d`.

Release build, rustc 1.88.0, `--locked`. Each arm has its own
worktree and target directory. The five driver sources are the same
bytes on every arm, so the binaries differ by the library they link.
`h2h_gmin_proof`, `h2h_seed` and `h2h_sc` are the develop sources.
`driver-src/h2h_morse.rs` and `driver-src/h2h_blj.rs` are the copies
from the three-arm gate harness, compiled unmodified into every arm.
No driver adapter is required for arm B: the Python binding changes
(`cluster_search` taking a `Config`) live in `src/python.rs` and
`python/anneal/`, while the five Rust drivers call
`anneal_core::methods::cluster_hopping::run_with_gradient` directly
with `&Config`, whose signature and search settings are unchanged
across arms A, A' and B.

## Rule

A hit is re-derived from `best` against the cell reference at
tolerance 0.001 (`best` finite and at or below `reference + 0.001`),
never read from the `hit` column. A run that errors, times out or
produces a non-finite energy is a miss. Misses are ranked as
positive infinity in the calls test. `calls_to_hit` is `inf` on a
miss.

A cell passes (`PASS`) only when all three clauses hold, and fails
(`FAIL`) if any clause fails:

(a) Energy calls (`charged` and `calls_to_hit`) are identical to A
per seed and best energies are bit-identical (`best_bits`), or else
hits are not lower than A under the exact Fisher test (`hits_b >=
hits_a` and one-sided Fisher `p >= 0.05`, with any miss or
non-finite energy counted as a miss and hits re-derived from `best`
against the reference).

(b) Time outside energy calls per call (`outside_s / max(1,
charged)`): the B/A bootstrap 95% confidence interval upper edge of
the type-7 median ratio (10,000 paired seed resamples, seed
`20261120`) is at most `max(1.10, A'/A band upper edge)`, where the
A'/A band upper edge is the larger of `max(A'/A, A/A')` and the
A'/A bootstrap 95% confidence interval upper edge.

(c) Peak `RssAnon` from `/proc/<pid>/status`: the B/A ratio of type-7
medians lies inside the A'/A band or is at most `1.05`. `ru_maxrss`
from `getrusage` is recorded and reported per cell but is not gated
on.

Wall-clock B/A and A'/A median ratios are also reported per cell.
Any trajectory difference between B and A or between A' and A is
listed per cell and seed. The gate fails if any cell fails.

## Protocol

Seeds 2000 through 2011, twelve seeds. For seed offset k, the order
(A, A', B) is rotated left by k mod 3, and an odd k reverses that
rotation. Each arm occupies each of the three slots four times. The
three arms of a seed run in that order, then the next seed. Cells
run one after another. No other timed work shares the session.

The process is pinned to core 0 with `taskset -c 0`. Every
`*_NUM_THREADS` variable (`OMP`, `OPENBLAS`, `MKL`, `RAYON`, `TBB`
and the rest of the harness thread list) is `1`. No `H2H_*` variable
is set. Each run logs `taskset -p` and `Cpus_allowed_list` from
`/proc/<pid>/status`, samples peak `RssAnon` from
`/proc/<pid>/status`, and records `ru_maxrss` from `wait4`. There is
no warm-up and no quick mode. The budget probe checks each non-water
driver before the first row.

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

Rows are appended as each arm finishes. `summary.csv`, `hosts.csv`
and the results section below are generated after the last row by
`run_gate.py summarize`.

## Results

One session, core 0, twelve seeds, twenty-eight cells, three arms.
1008 rows. Gate verdict: **PASS**. 28 cells pass, 0 fail.
Clause (a) failures: none. Clause (b) failures: none. Clause (c) failures: none. Hit losses: none.

Hits are A/A'/B (re-derived from `best` against the reference). `Traj` is `traj_match_ba/traj_match_apa` over the 12 seeds. Outside ratios are B/A (with bootstrap 95% CI) and A'/A of the median outside seconds over charged calls. Memory ratios are peak `RssAnon` (gating) and `ru_maxrss` (reported, non-gating) B/A and A'/A medians. Wall ratios are B/A and A'/A of median wall-clock seconds.

| Cell | Traj | Hits | Fisher p | Med calls A, B | Outside B/A [95% CI], A'/A | RssAnon B/A, A'/A | ru_maxrss B/A, A'/A | Wall B/A, A'/A | Rule | Verdict |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 12/12 | 12/12/12 | 1 | 127840, 127840 | 0.994843 [0.975288, 1.01016], 1.00676 | 1.00474, 1.0058 | 1, 1 | 0.993586, 1.00903 |  | PASS |
| lj75 | 12/12 | 5/5/5 | 0.659914 | inf, inf | 1.0079 [0.967085, 1.03178], 1.00399 | 0.999716, 0.999866 | 0.998554, 0.998988 | 1.00523, 1.00484 |  | PASS |
| lj98 | 12/12 | 7/7/7 | 0.659914 | 3.16681e+06, 3.16681e+06 | 1.01748 [0.970446, 1.0625], 0.995355 | 1.00071, 1.00015 | 0.999581, 1.00039 | 0.977226, 0.994312 |  | PASS |
| lj102 | 12/12 | 6/6/6 | 0.657864 | inf, inf | 0.997471 [0.97465, 1.01805], 1.01983 | 1.00056, 1.00016 | 0.999835, 0.999284 | 1.00397, 1.01219 |  | PASS |
| lj104 | 12/12 | 5/5/5 | 0.659914 | inf, inf | 1.01387 [0.959817, 1.03871], 1.0237 | 0.997705, 0.997493 | 1.00001, 0.99991 | 0.997825, 0.998445 |  | PASS |
| blj105n38 | 12/12 | 2/2/2 | 0.704969 | inf, inf | 1.05094 [0.979323, 1.07016], 0.995133 | 1.00007, 0.999933 | 1.00189, 1.00034 | 1.03143, 1.00976 |  | PASS |
| blj105n55 | 12/12 | 12/12/12 | 1 | 158055, 158055 | 1.01255 [0.999417, 1.03174], 1.01293 | 0.999919, 1 | 1.00022, 1.00118 | 1.00859, 1.00685 |  | PASS |
| blj115n38 | 12/12 | 12/12/12 | 1 | 82300.5, 82300.5 | 1.04055 [0.978787, 1.0524], 1.03521 | 1.03545, 1.03554 | 0.998789, 0.997365 | 1.00914, 1.01337 |  | PASS |
| blj115n55 | 12/12 | 0/0/0 | 1 | inf, inf | 0.975276 [0.95453, 1.05968], 0.976134 | 1.00003, 1 | 0.99883, 0.999303 | 0.975107, 0.979413 |  | PASS |
| m55r14 | 12/12 | 9/9/9 | 0.679797 | 451562, 451562 | 1.00189 [0.948109, 1.02594], 1.02002 | 0.997416, 1.00152 | 1, 1 | 0.956677, 0.967497 |  | PASS |
| m80r6 | 12/12 | 12/12/12 | 1 | 62677, 62677 | 0.972677 [0.953677, 1.00656], 0.979489 | 1.00044, 1.00044 | 1, 1 | 0.99701, 1.00289 |  | PASS |
| m80r14 | 12/12 | 6/6/6 | 0.657864 | inf, inf | 0.980246 [0.961329, 1.03414], 0.983446 | 1, 0.999979 | 0.999538, 0.999969 | 0.996645, 0.977857 |  | PASS |
| sc126-38 | 12/12 | 12/12/12 | 1 | 113504, 113504 | 0.99561 [0.977328, 1.02314], 1.01193 | 1, 0.997863 | 1, 1 | 1.00496, 0.987541 |  | PASS |
| sc126-55 | 12/12 | 12/12/12 | 1 | 18379.5, 18379.5 | 1.01939 [0.985172, 1.02881], 1.03582 | 1.00539, 1 | 1, 1 | 1.00843, 1.00092 |  | PASS |
| tip4p8 | 12/12 | 12/12/12 | 1 | 6339, 6339 | 1.01617 [0.990727, 1.03905], 1.04216 | 1.00296, 1.00296 | 1, 1 | 1.0174, 1.02671 |  | PASS |
| tip4p9 | 12/12 | 12/12/12 | 1 | 5839.5, 5839.5 | 0.9799 [0.966285, 1.01246], 1.00405 | 1.00585, 1.00585 | 1, 1 | 1.00018, 1.01055 |  | PASS |
| tip4p10 | 12/12 | 12/12/12 | 1 | 13138.5, 13138.5 | 0.99428 [0.969742, 1.05052], 0.996133 | 1.00373, 1.01119 | 1, 1 | 0.993573, 1.00549 |  | PASS |
| tip4p11 | 12/12 | 12/12/12 | 1 | 109288, 109288 | 0.985661 [0.971251, 1.05116], 0.991243 | 0.987009, 1.00087 | 1, 1 | 0.989371, 0.998857 |  | PASS |
| tip4p12 | 12/12 | 12/12/12 | 1 | 41418.5, 41418.5 | 1.00325 [0.990365, 1.02254], 1.017 | 1.00321, 1.05074 | 1, 1 | 1.00009, 1.00824 |  | PASS |
| tip4p13 | 12/12 | 12/12/12 | 1 | 216690, 216690 | 0.985522 [0.960196, 1.00396], 0.991896 | 1.00329, 1.0009 | 1, 1 | 1.0016, 1.00458 |  | PASS |
| tip4p14 | 12/12 | 12/12/12 | 1 | 135853, 135853 | 0.989925 [0.954341, 1.01043], 1.00244 | 0.996285, 0.994338 | 1, 1 | 0.998895, 1.00434 |  | PASS |
| tip4p15 | 12/12 | 7/7/7 | 0.659914 | 1.48162e+06, 1.48162e+06 | 0.997362 [0.968938, 1.01626], 1.0128 | 0.999855, 1.00079 | 0.996985, 0.999837 | 1.00399, 1.02316 |  | PASS |
| tip4p16 | 12/12 | 2/2/2 | 0.704969 | inf, inf | 0.999236 [0.989258, 1.0228], 1.00463 | 0.999613, 0.999742 | 0.999012, 0.999259 | 1.00194, 0.99832 |  | PASS |
| tip4p17 | 12/12 | 5/5/5 | 0.659914 | inf, inf | 0.98886 [0.97503, 0.998164], 0.987544 | 0.999916, 1.00002 | 0.998292, 0.999176 | 0.994254, 1.00559 |  | PASS |
| tip4p18 | 12/12 | 2/2/2 | 0.704969 | inf, inf | 0.990366 [0.981759, 1.00504], 1.01149 | 0.999868, 0.999831 | 0.998432, 0.999892 | 0.996232, 0.996898 |  | PASS |
| tip4p19 | 12/12 | 7/7/7 | 0.659914 | 1.06084e+06, 1.06084e+06 | 0.999884 [0.984699, 1.01198], 1.0006 | 0.999704, 1.00062 | 0.998904, 1.00224 | 0.996123, 0.998348 |  | PASS |
| tip4p20 | 12/12 | 6/6/6 | 0.657864 | inf, inf | 0.988371 [0.930028, 0.998054], 1.0002 | 0.999621, 1.0004 | 0.99797, 0.999221 | 1.00118, 0.9981 |  | PASS |
| tip4p21 | 12/12 | 0/0/0 | 1 | inf, inf | 0.994934 [0.984671, 1.00068], 0.999621 | 0.999813, 0.99992 | 0.999236, 1.0003 | 0.998655, 0.999649 |  | PASS |

### Trajectory Differences

None: all 28 cells x 12 seeds (336 seed pairs) are bit-identical in `best_bits`, `charged` and `calls_to_hit` between B and A and between A' and A.
