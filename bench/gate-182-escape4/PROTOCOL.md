# PR 182 Escape-4 Performance Gate Protocol

This protocol is committed before any driver build or measurement row.
No library source edits are permitted on either arm.

## Arms and Builds

- **Arm A**: `cd6995e53ffe0e065fe3e4e6b63d2a9c481b904f`
  (`origin/bench/gate-pr182-screen-escape3`).
- **Arm B**: `d41f0ae293884d1173d6ab1a7201e66749d3848a`
  (tree `a88dffd0f9c49cbafe4cbcf0a8fd77b154ee7fcf`).
- **Arm A2**: second independent release build of Arm A in a fresh
  target directory.
- **Arm B2**: second independent release build of Arm B in a fresh
  target directory.

Each arm is built in its own detached worktree with its own
`CARGO_TARGET_DIR` in `--release` mode. A shared Rust driver under
`bench/gate-182-escape4/driver/` is used identically across all four
builds with zero edits to library sources.

## Execution Order and Isolation

All runs execute serially on this VM, pinned to CPU core `0` via
`taskset -c 0`, in one session, with single-thread environment settings
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`RAYON_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`,
`BLIS_NUM_THREADS=1`).

Each cell runs 12 rounds (`r = 0..11`). For multi-seed cells, round `r`
uses seed `2000 + r` (`2000..2011`). For the fixed-seed short-walk cell
(`seed = 7`), all 12 rounds repeat `seed = 7`.

Within each round, the four arms run in counterbalanced ABBA order:

- `r % 4 == 0`: `A, B, B2, A2`
- `r % 4 == 1`: `B, A, A2, B2`
- `r % 4 == 2`: `A2, B2, B, A`
- `r % 4 == 3`: `B2, A2, A, B`

## Cells

Hit threshold for every Lennard-Jones size `N` is `reference + 1e-3`,
using the standard Cambridge Cluster Database reference energies:
LJ38 `-173.928427`, LJ55 `-279.248470`, LJ75 `-397.492331`,
LJ98 `-543.665361`, LJ102 `-569.363652`, LJ103 `-575.766131`,
LJ104 `-582.086642`, LJ150 `-893.310258`.

1. **LJ98 short walk** (`cluster_search::search`, `Config::recommended(98)`,
   budget `2500`, fixed `seed = 7`, 12 timed repeats):
   - `lj98_short_d0`: `packing_deferral = 0`
   - `lj98_short_d4`: default `packing_deferral` (`4`)

2. **LJ98 trajectory-pin walk** (`run_with_gradient`, `WarmLbfgs::default()`,
   `rgmin::vecops::force_simd_path("scalar")`, `Config::recommended(98)`,
   budget `5000`, seeds `2000..2011`):
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
     default `packing_deferral` (`4`) (also the LJ38 panel cell)

4. **Standard LJ panel cells** (`run_with_gradient`, `WarmLbfgs::default()`,
   `Config::recommended(N)`, default `packing_deferral = 4`, target stop
   at `reference + 1e-3`, seeds `2000..2011`, default budgets):
   - `lj38_d4`: `N = 38`, budget `400000`
   - `lj55`: `N = 55`, budget `1000000`
   - `lj75`: `N = 75`, budget `4000000`
   - `lj98`: `N = 98`, budget `4000000`
   - `lj102`: `N = 102`, budget `4000000`
   - `lj103`: `N = 103`, budget `4000000`
   - `lj104`: `N = 104`, budget `4000000`
   - `lj150`: `N = 150`, budget `4000000`

## Measurement and CSV Schema

Rows are appended immediately after each run to
`bench/gate-182-escape4/rows.csv` with columns:

`arm,sha,cell,seed,deferral,hit,calls_to_hit,best_energy,hops,inner_reps,outside_s,rss_anon_kib,core`

- `inner_reps`: repeat count of the search loop within one timed span,
  calibrated per `(cell, round)` so total timed wall duration is at
  least `50 ms` (`0.050 s`), and held identical across `A`, `B`, `A2`,
  and `B2` for that `(cell, round)`.
- `outside_s`: `max(0.0, wall_total - pot_total) / inner_reps` in seconds,
  where `wall_total` spans all `inner_reps` runs and `pot_total` sums
  `PairPotential::value_and_gradient` evaluations inside that timed span.
- `calls_to_hit`: charged calls at first hit when `hit = 1`, or
  `budget + 1` when `hit = 0`.
- `rss_anon_kib`: peak in-process `RssAnon` in KiB observed from
  `/proc/self/status` across startup, periodic sampling during search,
  and immediately after the timed loop prior to dropping outcomes.
- `core`: pinned CPU core index (`0`).

## Statistical Analysis and Pass/Fail Bar

For each cell (`n = 12` paired rounds):

- **Hits**: two-sided (and one-sided `A > B`) Fisher exact test on
  `hits_B` vs `hits_A`.
- **Calls to hit**: two-sided (and one-sided `B > A`) Mann-Whitney U
  test with tie correction and continuity correction `0.5`, entering
  every miss as `budget + 1`.
- **Outside time and RSS ratios**: paired round ratios `B / A` and
  `A2 / A` (`10000` bootstrap resamples of the Hyndman-Fan type-7
  median with PRNG seed `20261123`), reporting the type-7 `2.5%` and
  `97.5%` quantiles as the 95% CI.
- **Real regression rule**: a ratio increase is real only if its lower
  95% CI bound is strictly above `1.0` and strictly above the `A2 / A`
  upper 95% CI bound.

**Cell PASS bar** (all conditions required):

1. No hit loss (`hits_B >= hits_A`).
2. No calls-to-hit loss (neither a statistically significant increase
   with `mwu_b_slower_p < 0.05` nor a higher median `calls_to_hit`
   for `B` than `A`).
3. `B / A` upper 95% CI on `outside_s` is `<= 1.10`.
4. `B / A` upper 95% CI on `rss_anon_kib` is `<= 1.10`.

Overall gate verdict is `PASS` if and only if every cell passes;
otherwise `FAIL`. Verdicts are never relabeled.
