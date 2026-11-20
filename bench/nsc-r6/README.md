# Restraint hops without the speculative plain quench

Three arms of the recommended preset, run on each code revision in turn:

| Arm | Revision | Tree | Code |
| --- | --- | --- | --- |
| A | `61fa67cf` | `cff796c5` | develop before the size-constant removal |
| B | `7fff9f2e` | `113a3420` | the surface at every size, the speculative plain quench, the release of a failed restraint block |
| C | this branch | `37e6d6fc` | B without the speculative plain quench |

C's `code_revision` in `rows.csv` is the commit as it was run. Applying the
patch again gives a new commit id with the same tree, so `code_tree` is the
identifier that holds.

Two entries per arm. `cluster_search::search` is the library and Python
default, run through `nsc_reach.rs` in this directory (copy it to
`examples/` to build). `h2h_seed recommended` is the shipped example. Both
use `Config::recommended(n)`. Systems and budgets are the paper budgets:
LJ103 at 4,000,000 calls, Morse 38 at rho 6 at 400,000, Morse 55 at rho 14
at 1,000,000. Seeds 2000 to 2011. Every run stops at the reference plus
1e-3. Release build, one thread per run, six runs at a time on one 8-core
x86_64 virtual machine; the `host` column names the machine by kind. The
216 commands are in `jobs.tsv`, where `A/`, `B/` and `C/` stand for each
arm's `target/release/examples`.

A hit is a best energy within 1e-3 of the reference. `evals_to_hit` is the
charged count at the first improvement within that tolerance; a miss counts
at the budget. Fisher is the two-sided exact test on hits, Mann-Whitney the
two-sided test on `evals_to_hit`. `python3 bench/nsc-r6/summary.py`
recomputes the table from `rows.csv`.

| path | system | A hits | B hits | C hits | A median | B median | C median | Fisher B/A | MW B/A | Fisher C/A | MW C/A | Fisher C/B | MW C/B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_search::search | lj103 | 4/12 | 11/12 | 11/12 | 4000000 | 17171 | 34577 | 0.00942 | 0.00122 | 0.00942 | 0.0058 | 1 | 0.0495 |
| cluster_search::search | m38r6 | 7/12 | 12/12 | 12/12 | 346723 | 102527 | 71151 | 0.0373 | 0.00373 | 0.0373 | 0.00448 | 1 | 0.453 |
| cluster_search::search | m55r14 | 1/12 | 8/12 | 11/12 | 1000000 | 692602 | 366737 | 0.00942 | 0.0116 | 0.000107 | 0.00077 | 0.317 | 0.0722 |
| h2h_seed recommended | lj103 | 8/12 | 12/12 | 11/12 | 2581470 | 24886 | 68294 | 0.0932 | 9.4e-05 | 0.317 | 0.00233 | 1 | 0.0226 |
| h2h_seed recommended | m38r6 | 4/12 | 12/12 | 12/12 | 400000 | 96526 | 83882 | 0.00135 | 0.0317 | 0.00135 | 0.0253 | 1 | 0.371 |
| h2h_seed recommended | m55r14 | 3/12 | 8/12 | 6/12 | 1000000 | 525140 | 883255 | 0.0995 | 0.059 | 0.4 | 0.259 | 0.68 | 0.338 |

C is slower than B on LJ103 on both entries: the median calls to a hit
doubles on `cluster_search::search` (Mann-Whitney 0.0495) and nearly
triples on `h2h_seed` (0.0226), and `h2h_seed` misses one more seed. On
Morse 55 at rho 14 the entries disagree: `cluster_search::search` gains
three hits, `h2h_seed` loses two. Neither hit difference is significant.
Morse 38 at rho 6 is level on hits with lower medians that are not
significant.

The Morse 55 tolerance admits a second minimum. Relaxing each
`cluster_search::search` hit to convergence, B reaches the reference
-220.646208 on 4 seeds and -220.645856 on 4; C reaches the reference on 4
and -220.645856 on 7; A reaches the reference on its one hit. Counted
against the reference minimum alone, B and C are 4/12 each.
`h2h_seed` does not print the relaxed check.

Arm B here reproduces the earlier run of the same code (tree `64b12739`,
which differs from `113a3420` only by `examples/ase_native_eval.rs`) on all
72 of its seeds.
