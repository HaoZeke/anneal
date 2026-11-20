# nm3 land9 panel gate

Arm A is develop `61fa67cf380e46876b569da45e7cc091647a775b`. Arm B is that
tree plus the three patches that drop motif-driven search moves, drop the
motif order caps, and reject self-mapping turns in symmetry detection.
The gate record is this directory on develop. It does not include arm B's
commits.

Both arms are release builds, each from its own worktree. The run reads
the two binary directories and the two source trees from the command
line. A row stores the sha256 of the binary that produced it.

## Run

One process at a time, pinned with `taskset -c 0`. The process is fresh
for every seed. Nothing is discarded as a warm-up, and no quick flag is
passed.

Every thread-count variable is `1`: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`GOTO_NUM_THREADS`, `OMP_THREAD_LIMIT`, `MKL_DOMAIN_NUM_THREADS`,
`JULIA_NUM_THREADS`, `TBB_NUM_THREADS`, `POLARS_MAX_THREADS`,
`OPENMP_NUM_THREADS`, and any inherited variable whose name ends in
`_NUM_THREADS`, `_MAX_THREADS`, or `_MAXIMUM_THREADS`. `OMP_DYNAMIC` and
`MKL_DYNAMIC` are `FALSE`. Variables named
`H2H_*` are unset, so each binary keeps the harness preset.

Seeds are 2000 through 2011 (n = 12). Within a seed the arm order is
ABBA: an even seed runs A then B, an odd seed runs B then A. Cells are
not interleaved. The order block below is the whole schedule.

The A/A band runs A against A, labelled A1 and A2, on LJ75, LJ38, and
LJ98. Even seeds run A1 then A2. Odd seeds run A2 then A1. Both labels
use the arm A binary. The band is the outside-time noise.

Budgets and references are the cells block. They are the paper-harness
values: `scripts/gmin_panel/run_panel.py` for Lennard-Jones, Gupta, and
Morse; `scripts/paper_0120/run_panel.py` for the extra LJ75 seed driver;
`scripts/metal_h2h/run_sc.py` for Sutton-Chen; `scripts/tip4p_h2h/run_h2h.py`
for TIP4P. Sutton-Chen uses this gate's twelve seeds. Its budgets stay
the harness budgets. The run refuses to start unless each `run` binary
prints that budget and reference, on both arms.

LJ103 is the `h2h_gmin_proof` system of that name. TIP4P25 is not a cell:
the harness has no fixed header for it and scores it against the session
minimum. This gate does not invent a reference. GFN2 cells are not in
this gate. The Sutton-Chen 9-6, 78-atom how-to uses other seeds and other
arms, so it is not a cell here.

A hit is `energy_hits` from `scripts/target_tol.py` at the default
tolerance: the recorded best is at or below the reference plus 0.001.
A miss stores `evals_to_hit` as the budget.

## Record

Rows are appended under `<cell>/rows.csv` and are not rewritten. A
finished `(cell, arm, seed)` is left in place. Columns:

```
cell,core,affinity,machine_id,binary_sha256,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s
```

`core` is 0. `affinity` is the process CPU list, which must be `0`.
`machine_id` is the sha256 of the machine id, not a hostname.
`inside_s` is time inside the potential. `outside_s` is the rest of the
wall. `charged` is the number of charged evaluations.

`<cell>/summary.csv` is computed from those rows. The directory
`summary.csv` repeats every cell. Columns:

```
cell,n,budget,reference,hits_a,hits_b,evals_median_a,evals_median_b,fisher_p,mannwhitney_p,outside_per_call_median_a,outside_per_call_median_b,outside_ratio,aa_n,aa_outside_per_call_median_a1,aa_outside_per_call_median_a2,aa_outside_ratio,verdict,reasons
```

Medians are Hyndman-Fan type 7. For n = 12 that is the mean of the 6th
and 7th ordered values. The eval median uses `evals_to_hit` as stored.

## Verdict

A cell with fewer than 12 A rows or 12 B rows, or with any other seed
set, is `INCOMPLETE`.

Otherwise the cell is `FAIL` when any of these holds, and `PASS` when
none do.

- Exact two-sided Fisher on the hit counts, and only when B has fewer
  hits than A. `FAIL` when that p is below 0.05. The p-value sums the
  tables with the observed margins whose probability is at most the
  observed table's probability.
- One-sided Mann-Whitney on the scores, alternative that B is worse.
  A hit scores `evals_to_hit`. A miss scores `budget + 1`, so a miss
  ranks worse than a hit that spent the whole budget. The statistic is
  the tie-corrected normal approximation with the continuity correction,
  the paper harness Mann-Whitney, on the upper tail. `FAIL` when that p
  is below 0.05.
- Outside time per call. Each row contributes `outside_s / charged`.
  The ratio is the type 7 median of B divided by the type 7 median of A.
  `FAIL` when the ratio is above 1.1. The A/A band reports the same
  ratio of A2 over A1 and is not itself a fail rule.

`reasons` lists the failed rules among `fisher`, `mannwhitney`, and
`outside`. The summary also reports each arm's hits, eval median, and
the two p-values.

The summary refuses a row whose core is not 0 or whose affinity is not
0, a row whose machine id is not one 64-digit hex digest shared by the
cell, or an arm whose binary sha256 is not constant. Arm A, A1, and A2
must share a sha256. Arm B must have a different one.

```
order
ab lj75 lj98 lj102 lj103 lj104
aa lj75 lj38 lj98
ab lj38 lj55 lj150 lj75-seed au55 m55r6 m55r14 m80r6 m80r14 sc126-38 sc126-55 sc126-80 sc108-38 sc108-55 sc108-80 sc96-38 sc96-55 sc96-80 tip4p8 tip4p15 tip4p20 tip4p21
```

```
cells
cell,family,binary,budget,reference
lj38,lj,h2h_gmin_proof,400000,-173.928427
lj55,lj,h2h_gmin_proof,1000000,-279.248470
lj75,lj,h2h_gmin_proof,4000000,-397.492331
lj98,lj,h2h_gmin_proof,4000000,-543.665361
lj102,lj,h2h_gmin_proof,4000000,-569.363652
lj103,lj,h2h_gmin_proof,4000000,-575.766131
lj104,lj,h2h_gmin_proof,4000000,-582.086642
lj150,lj,h2h_gmin_proof,4000000,-893.310258
lj75-seed,lj-seed,h2h_seed,4000000,-397.492331
au55,gupta,h2h_metal,1000000,-191.03085280
m55r6,morse,h2h_morse,1000000,-250.286609
m55r14,morse,h2h_morse,1000000,-220.646208
m80r6,morse,h2h_morse,4000000,-378.333471
m80r14,morse,h2h_morse,4000000,-340.811371
sc126-38,sc,h2h_sc,400000,-35419.9804
sc126-55,sc,h2h_sc,1000000,-52884.6806
sc126-80,sc,h2h_sc,1000000,-78414.6271
sc108-38,sc,h2h_sc,400000,-10117.2454
sc108-55,sc,h2h_sc,1000000,-14814.5225
sc108-80,sc,h2h_sc,1000000,-21819.9952
sc96-38,sc,h2h_sc,400000,-8917.7056
sc96-55,sc,h2h_sc,1000000,-13217.8963
sc96-80,sc,h2h_sc,1000000,-19560.8966
tip4p8,tip4p,h2h_seed,2000000,-305.5183221676
tip4p15,tip4p,h2h_seed,2000000,-628.3722525077
tip4p20,tip4p,h2h_seed,2000000,-872.9887544952
tip4p21,tip4p,h2h_seed,2000000,-916.7074302940
```
