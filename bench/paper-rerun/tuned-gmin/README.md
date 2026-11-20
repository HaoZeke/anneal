# Tuned GMIN rerun

Fresh measurement of the printed GMIN cells whose per-cluster settings
are recorded in this tree and whose earlier harness commit is not.
The base is 61fa67cf380e46876b569da45e7cc091647a775b.
Budgets are PAPER_BUDGETS in scripts/gmin_panel/run_panel.py.
Seeds of a run cell are 2000-2011.
Arm order inside a seed is ABBA when both codes of that system run.

A hit is energy_hits in scripts/target_tol.py: best <= reference + 0.001.
DEFAULT_HIT_TOL is 0.001. A miss is stored at the budget.
calls_median is the Hyndman-Fan type 7 median of evals_to_hit.
outside_per_charged_median is the type 7 median of outside_s/charged.
fisher_p is the two-sided exact test on the hit counts.
mannwhitney_p ranks a full-budget miss after every hit.
Measured rows are rows-r2.csv.
rows.csv is the earlier anneal-only pass.

GMIN is the build in scripts/gmin_panel/build_gmin.sh.
The pin is SVNREV 9cbec1a446352bc9ef064bd2b1a5767194655919.
The tarball sha256 is ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776.
The script refuses a tarball whose SVNREV or sha256 differs.
ALIGN, MPI and OpenMP are off. The link is the reference BLAS and LAPACK.
Anneal code_revision is the base commit. GMIN code_revision is the pin.

host e06d886d760e0a59870126b963636ff6408201594f2686f72deb66816fccadb3
session 20261120T072444Z
harness_revision f009acc96bb95023eeb6e394e863630c752a5d05
harness_blob c48f4e7bae4c3ae607829c40693d415bf3cecaed714a56c2f76dcc35ed9d2a8d

| cell | code | status | hits/n | calls median | outside/charged | fisher p | mann-whitney p | printed hits/n | printed calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| au55 | anneal | run | 4/12 | 1000000 | 9.4620265e-06 | 0.31677 | 0.207632 | 4/12 | 1000000 |
| au55 | gmin | run | 1/12 | 1000000 | 3.3173225e-06 | 0.31677 | 0.207632 | 1/12 | 1000000 |
| m55r6 | anneal | run | 12/12 | 22596 | 5.02688557e-06 | 1 | 0.930987 | 12/12 | 92049.5 |
| m55r6 | gmin | run | 12/12 | 37047.5 | 2.770448239e-06 | 1 | 0.930987 | 12/12 | 37047.5 |
| m55r14 | anneal | run | 9/12 | 451561.5 | 4.41893706e-06 | 0.000336519 | 0.000383392 | 7/12 | 730604 |
| m55r14 | gmin | run | 0/12 | 1000000 | 2.4612255e-06 | 0.000336519 | 0.000383392 | 0/12 | 1000000 |
| m80r6 | anneal | run | 12/12 | 62677 | 1.237066804e-05 | 1 | 0.260236 | 2/12 | 4000000 |
| m80r6 | gmin | run | 12/12 | 268934 | 3.918998905e-06 | 1 | 0.260236 | 12/12 | 268934 |
| m80r14 | anneal | run | 6/12 | 3901429 | 9.653637213e-06 | 0.01373 | 0.0070541 | 9/12 | 1338932.5 |
| m80r14 | gmin | run | 0/12 | 4000000 | 3.52032025e-06 | 0.01373 | 0.0070541 | 0/12 | 4000000 |
| lj75-sym | gmin | unrunnable |  |  |  |  |  | 200/200 | 33122 |
| lj98-early | gmin | unrunnable |  |  |  |  |  | 64/64 | 158540.5 |
| lj98-early | full | unrunnable |  |  |  |  |  | 24/64 | 2000000 |
| lj38 | gmin | not_tuned |  |  |  |  |  | 7/12 | 389399 |
| lj55 | gmin | not_tuned |  |  |  |  |  | 12/12 | 12764.5 |
| lj75 | gmin | not_tuned |  |  |  |  |  | 0/12 | 4000000 |
| lj98 | gmin | not_tuned |  |  |  |  |  | 0/12 | 4000000 |
| lj102 | gmin | not_tuned |  |  |  |  |  | 5/12 | 4000000 |
| lj103 | gmin | not_tuned |  |  |  |  |  | 3/12 | 4000000 |
| lj104 | gmin | not_tuned |  |  |  |  |  | 6/12 | 3696118 |
| lj150 | gmin | not_tuned |  |  |  |  |  | 9/12 | 3129518.5 |
| tip4p | gmin | not_tuned |  |  |  |  |  | 8 0 0 0 0/12 |  |

status run was executed. status unrunnable was not executed.
lj75-sym is missing the OptBench LJ75 SYMMETRISE keyword file and coords.1 through coords.200.
lj98-early is missing the GMIN keyword file and the start generator for seeds 0-63.
status not_tuned is a printed GMIN count under one shared input,
which the manuscript separates from a per-cluster input.
