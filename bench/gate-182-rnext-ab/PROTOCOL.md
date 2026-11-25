# PR 182 r-next A/B Gate Protocol

This protocol is committed before any driver build or measurement row.
No library source is edited on either arm. The Rust driver is the
escape4 gate driver, byte for byte, built against each arm.

## Question

Arm B changes the warm-relax cut in the packing audit. After a
relaxation, the ledger is closed when it has stopped or has no calls
left. Arm A treats that closure as a cut only when the requested step
count was shortened. Arm B also treats it as a cut when the relax was
refused at least one further evaluation, including when the requested
step count fit. A cut is handed back and its quench waits. That
deferral can move calls-to-hit.

The measured binaries are release builds of the named commits, not a
cherry-pick. On the Lennard-Jones search path used here, that cut
predicate is the library change under test.

## Arms and Builds

- **Arm A**: `d41f0ae293884d1173d6ab1a7201e66749d3848a`
  (tree `a88dffd0f9c49cbafe4cbcf0a8fd77b154ee7fcf`).
- **Arm B**: `af23f1f7f977d5b4a42e50c2a8993233b40853f4`
  (tree `b11f6a576c257000a04e5c2b9f271be4fb74d202`).
- **Arm A2**: a second independent `--release` build of arm A, in its
  own detached worktree and a fresh `CARGO_TARGET_DIR`. A2 is the A/A
  band for outside time and RSS. It is the same search as A.

Each arm is built with `cargo build --release` of
`bench/gate-182-rnext-ab/driver` inside a detached worktree of that
arm's commit. The recorded identity of a build is the commit and tree
that were compiled. A and A2 share that commit and tree and are
distinct binaries. No other cargo profile or rustc flag is set.

## Execution

All measured rows run serially on one machine, pinned to CPU core 0
(`taskset -c 0`), one thread per process. The runner sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`RAYON_NUM_THREADS`, `NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
`BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `GOTO_NUM_THREADS`,
`OMP_THREAD_LIMIT`, `MKL_DOMAIN_NUM_THREADS`, and `OPENMP_NUM_THREADS`
to `1`, and sets `OMP_DYNAMIC` and `MKL_DYNAMIC` to `FALSE`.

The 15 cells, seeds, and budgets are the escape4 gate cells. Each cell
has `n = 12` rounds. Multi-seed cells use seeds `2000..2011`. The two
LJ98 short cells repeat seed `7` for all 12 rounds.

Within a round the three arms run in counterbalanced ABBA order. The
A-versus-B order across rounds `0..11` is A, B, B, A, repeated three
times. A2 takes the remaining slot so that each arm stands in each of
the three positions four times:

- `0`: `A, B, A2`
- `1`: `B, A, A2`
- `2`: `B, A2, A`
- `3`: `A, A2, B`
- `4`: `A2, A, B`
- `5`: `A2, B, A`
- `6`: `B, A, A2`
- `7`: `A, B, A2`
- `8`: `A, A2, B`
- `9`: `B, A2, A`
- `10`: `A2, B, A`
- `11`: `A2, A, B`

`inner_reps` is chosen on arm A for that round and then held fixed for
B and A2. Cells with budget `4000000` use one rep. Every other cell
raises the rep count until the calibration wall is at least `0.075 s`.
A recorded row is kept only when its wall is at least `0.050 s`.

## Cells

The hit threshold for every size `N` is `reference + 1e-3`
(`DEFAULT_HIT_TOL`), using the Cambridge Cluster Database references
LJ38 `-173.928427`, LJ55 `-279.248470`, LJ75 `-397.492331`,
LJ98 `-543.665361`, LJ102 `-569.363652`, LJ103 `-575.766131`,
LJ104 `-582.086642`, LJ150 `-893.310258`.

1. **LJ98 short walk** (`cluster_search::search`,
   `Config::recommended(98)`, budget `2500`, seed `7`, 12 repeats):
   - `lj98_short_d0`: `packing_deferral = 0`
   - `lj98_short_d4`: default `packing_deferral` (`4`)
2. **LJ98 trajectory-pin walk** (`run_with_gradient`,
   `WarmLbfgs::default()`, scalar SIMD path, budget `5000`,
   seeds `2000..2011`):
   - `lj98_traj_d0`: `packing_deferral = 0`
   - `lj98_traj_d4`: default `packing_deferral` (`4`)
3. **LJ38 controls** (seeds `2000..2011`):
   - `lj38_short_d0`: `cluster_search::search`, budget `2500`,
     `packing_deferral = 0`
   - `lj38_short_d4`: `cluster_search::search`, budget `2500`,
     default `packing_deferral` (`4`)
   - `lj38_d0`: `run_with_gradient` with target stop, budget `400000`,
     `packing_deferral = 0`
   - `lj38_d4`: `run_with_gradient` with target stop, budget `400000`,
     default `packing_deferral` (`4`)
4. **LJ panel** (`run_with_gradient`, `WarmLbfgs::default()`,
   `Config::recommended(N)`, default `packing_deferral = 4`, target
   stop, seeds `2000..2011`):
   - `lj55`: budget `1000000`
   - `lj75`, `lj98`, `lj102`, `lj103`, `lj104`, `lj150`: budget
     `4000000`

`lj38_d4` is both the LJ38 deferral-4 control and the panel cell.

## Rows

Rows are appended after each run to `bench/gate-182-rnext-ab/rows.csv`:

`arm,sha,cell,seed,deferral,hit,calls_to_hit,best_energy,hops,`
`inner_reps,outside_s,rss_anon_kib,core`

- `calls_to_hit` is the charged calls at the first hit, or `budget + 1`
  on a miss.
- `outside_s` is `max(0, wall_total - pot_total) / inner_reps`, where
  `pot_total` sums `PairPotential::value_and_gradient` time inside the
  timed span.
- `rss_anon_kib` is the peak in-process `RssAnon` from
  `/proc/self/status`, sampled at startup, every 4096 evaluations, and
  after the timed loop.
- `core` is `0`.

## Bar

For each cell (`n = 12` paired rounds), using CPython `random.Random`
seed `20261123` and Hyndman-Fan type-7 quantiles. The generator
advances once per cell in the cell order below, and within a cell in
this order: calls ratio `B/A`, outside `B/A`, outside `A2/A`, RSS
`B/A`, RSS `A2/A`. Each ratio CI is 10000 bootstrap resamples of the
paired type-7 median, with the 95% interval at the type-7 2.5% and
97.5% quantiles of those medians.

- **Hits.** Two-sided and one-sided (`A > B`) Fisher exact test on the
  2x2 hit table. A hit loss is `hits_B < hits_A`. The p-value is
  reported and does not excuse a lower hit count.
- **Calls-to-hit.** A miss counts as `budget + 1`. Report the type-7
  median of each arm, the two-sided and one-sided (`B` slower)
  Mann-Whitney U (tie correction, continuity correction `0.5`), and
  the paired median ratio `B/A` with its 95% CI. A calls-to-hit loss
  is any of: median `B` greater than median `A`; one-sided MWU
  `p < 0.05`; or the ratio CI lower bound strictly above `1 + 1e-9`.
  A lower bound that clears 1 by at most `1e-9` is numerical noise and
  is not a loss.
- **Outside time and RSS.** Report the paired median ratio `B/A` with
  its 95% CI next to the `A2/A` 95% CI. These are context only.

Best energies are reported. A paired gap at `1e-9`, and agreement
inside the `1e-3` hit window, is not a verdict input.

A cell **FAILS** only on a hit loss or a calls-to-hit loss. The gate
**FAILS** if any cell fails. Computed verdicts are not relabeled.
