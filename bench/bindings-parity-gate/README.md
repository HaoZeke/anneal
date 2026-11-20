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

Not yet run.
