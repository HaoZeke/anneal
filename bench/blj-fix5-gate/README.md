# PR #95 Gate (`fix/blj-fix3` without `with_species` `budget_window` forcing)

Paired serial benchmark comparing:

- **Arm A**: `develop` (`21cd57b6eb0b4d79ff86a5769632b65f3e6ccd82`)
- **Arm B**: `15f38ac242101fa9c5ffa528566cfd485af0f157` + `anneal-blj-fix5.mbox` (tree `208b1b1d3e8ec79a40d8fdea069a4871fbb1d821`)

## Panels and sample sizes

1. **BLJ panel (`blj`)**: seven Doye--Meyer binary Lennard-Jones clusters (`blj105n38`, `blj105n55`, `blj115n38`, `blj115n55`, `blj130n38`, `blj130n55`, `blj130n98`) using `Config::recommended(n).with_species(labels, None)` (labels `200`/`201`, no diameters), `PAPER_BUDGETS` (`400_000` for `n=38`, `1_000_000` for `n=55`, `4_000_000` for `n=98`), and the `h2h_blj` harness specification against the Cambridge Cluster Database references (`HIT_TOL = 0.001`).
   - Upfront sample-size declaration (prior to running): `blj130n98` uses `n = 6` (`seeds 2000-2005`) because prior BLJ98 measurements at 4,000,000 evaluations run ~1735 s + 333 s per seed (~6.9 h wall for 2 arms x 12). All other BLJ systems use `n = 12` (`seeds 2000-2011`).
2. **Binary water via `search_binary` (`binary_water`)**: `lj100-binary` (`n_a=42`, `n_b=58`, `sigma_aa=1.0`, `sigma_ab=1.15`, `sigma_bb=1.3`, `budget=2_000_000`, reference `-604.7963065717`, `HIT_TOL=0.001`) via `anneal_core::binary_water::search_binary` on the default `recommended` preset (`communicating = false`, `--features rgsaddle`), using optbench starts `coords.0`–`coords.11` paired with `seeds 2000-2011` (`n = 12`). On Arm B, `pot_s` is timed directly via `ClusterKernels::binary_lj`; on Arm A (`21cd57b6`, which has no `ClusterKernels::binary_lj` hook and no potential callback in `search_binary`), `pot_s` is `charged * post-run single-eval latency` of `BinaryLennardJones::value_and_gradient`.
3. **Rigid-body water (`water_gmin`)**: existing `bench/water-gmin` TIP4P panel (`tip4p8`, `tip4p12`, `tip4p16`, `tip4p20`, `tip4p21`), `Config::recommended_tip4p(n)`, `budget = 2_000_000`, `n = 12` (`seeds 2000-2011`) via `examples/h2h_seed`.

## Protocol

- Serial execution in one session under `unshare --uts` with hostname `bench`, `taskset -c 0`, and `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`.
- Arms A and B are interleaved per seed, flipping order (`A, B` vs `B, A`) on alternating seeds, with a fresh process per run and append-only rows in `rows.csv`.
- Two-sided Fisher exact test on hits and two-sided tie-corrected Mann--Whitney U test on all-seed `evals_to_hit` (misses entered at budget). Medians are Hyndman--Fan type 7.

## Results

<!-- gate table: start -->
| Panel | System | n | Budget | Hits A | Hits B | Fisher p | Calls A | Calls B | MW p | Inside A/B (s) | Outside A/B (s) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blj | blj105n38 | 12 | 400000 | 1/12 | 2/12 | 1 | 400000 | 400000 | 0.5139 | 1.382 / 1.987 | 28.497 / 3.901 | PASS |
| blj | blj105n55 | 12 | 1000000 | 12/12 | 12/12 | 1 | 204047 | 158055 | 0.3408 | 1.566 / 1.752 | 12.524 / 0.651 | PASS |
| blj | blj115n38 | 12 | 400000 | 9/12 | 12/12 | 0.2174 | 138863.5 | 82300.5 | 0.4699 | 0.506 / 0.444 | 1.676 / 0.298 | PASS |
| blj | blj115n55 | 12 | 1000000 | 0/12 | 0/12 | 1 | 1000000 | 1000000 | 1 | 7.301 / 10.426 | 263.9 / 25.618 | PASS |
| blj | blj130n38 | 12 | 400000 | 4/12 | 11/12 | 9.42e-03 | 400000 | 134072.5 | 0.0802 | 1.385 / 0.666 | 18.827 / 0.681 | PASS |
| blj | blj130n55 | 12 | 1000000 | 4/12 | 9/12 | 0.0995 | 1000000 | 633042 | 0.3467 | 7.269 / 6.697 | 111.8 / 9.301 | PASS |
| blj | blj130n98 | 6 | 4000000 | 0/6 | 0/6 | 1 | 4000000 | 4000000 | 1 | 92.060 / 130.9 | 1980.8 / 57.177 | PASS |
| binary_water | lj100-binary | 12 | 2000000 | 0/12 | 0/12 | 1 | 2000000 | 2000000 | 1 | 47.731 / 68.998 | 47.581 / 50.179 | PASS |
| water_gmin | tip4p8 | 12 | 2000000 | 12/12 | 12/12 | 1 | 6339 | 6339 | 1 | 0.012 / 0.012 | 0.008 / 0.008 | PASS |
| water_gmin | tip4p12 | 12 | 2000000 | 12/12 | 12/12 | 1 | 41418.5 | 41418.5 | 1 | 0.150 / 0.150 | 0.071 / 0.071 | PASS |
| water_gmin | tip4p16 | 12 | 2000000 | 2/12 | 2/12 | 1 | 2000000 | 2000000 | 1 | 11.907 / 11.901 | 4.283 / 4.214 | PASS |
| water_gmin | tip4p20 | 12 | 2000000 | 6/12 | 6/12 | 1 | 1686249.5 | 1686249.5 | 1 | 14.817 / 14.896 | 4.131 / 4.127 | PASS |
| water_gmin | tip4p21 | 12 | 2000000 | 0/12 | 0/12 | 1 | 2000000 | 2000000 | 1 | 19.675 / 19.526 | 6.144 / 6.052 | PASS |
<!-- gate table: end -->

## Verdict

<!-- gate verdict: start -->
**Verdict: PASS**

Hit drops: none.

Call increases: none.
<!-- gate verdict: end -->
