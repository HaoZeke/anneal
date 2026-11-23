# 156 r-next3 performance and parity gate protocol

This gate evaluates PR #156 `r-next3` against `develop` on the five
`r-next2` FAIL cells (`n = 24`) and eight control cells (`n = 12`).
`r-next3` stops `csm::project` from keeping a match row per orientation
and removes the per-op solve cache and the uniqueness flag, so solves
that the cache previously skipped are recomputed.

## Arms

- `A`: `develop` commit `5abb4dedec6cd2e95e7b17472639529fd139b278`
- `A2`: second independent release build of `5abb4dedec6cd2e95e7b17472639529fd139b278`
  for the A/A noise floor band
- `B`: `5abb4ded` + `anneal-156-rnext3.mbox` applied with `git am -3`,
  commit `5ebffa4a29a20e71bb3e265f27bec8dd0b321c89`,
  tree `4a05a694d5882a89ae8776a4efac93d4b6ec6d48`
- `R`: `r-next2` (`pull/156/head` `52c2937fc59bb22dc78b443326a1b54c7cde816d`
  merged with `5abb4ded`),
  commit `2042b1b6d70be2adaf577f2ddf886aee1937cd48`,
  tree `5ddfb3f4cc5ae1c07d9055b7d87860f0681d6448`

## Build and runtime rules

- Each arm is built with:
  `cargo build --release --locked --features rgpot-ex --example h2h_gmin_proof --example h2h_seed --example h2h_metal --example h2h_morse --example h2h_sc`
- `RGPOT_SC_LIB` is one shared library built from `rgpot` commit
  `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e` shared across all arms.
- Serial execution on one core: `taskset -c 0`, one process at a time,
  all `*_NUM_THREADS`, `*_MAX_THREADS`, and `*_MAXIMUM_THREADS` set to `1`,
  `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`, no `--quick`.
- ABBA arm interleaving per seed:
  - even seeds: `A`, `B`, `R`, `A2`
  - odd seeds: `A2`, `R`, `B`, `A`
- In-process `LD_PRELOAD` probe (`probe.c`) measures `RssAnon` peak via
  `/proc/self/status` and counts `SIGPROF` ticks landing inside
  `anneal_core` `#[target_feature]` vector functions.
- Rows are appended to `<cell>/rows.csv` with LF (`\n`) endings and never
  rewritten.

## Schedule and cells

```
order
run sc96-38 sc96-55 lj38 lj55 tip4p8 sc126-55 m55r14 au55 lj75 lj98 sc108-80 lj103 lj104
```

```
cells
cell,group,n,family,binary,budget,reference
sc96-38,fail_rnext2,24,sc,h2h_sc,400000,-8917.7056
sc96-55,fail_rnext2,24,sc,h2h_sc,1000000,-13217.8963
lj38,control,12,lj,h2h_gmin_proof,400000,-173.928427
lj55,control,12,lj,h2h_gmin_proof,1000000,-279.248470
tip4p8,control,12,tip4p,h2h_seed,2000000,-305.5183221676
sc126-55,control,12,sc,h2h_sc,1000000,-52884.6806
m55r14,control,12,morse,h2h_morse,1000000,-220.646208
au55,fail_rnext2,24,gupta,h2h_metal,1000000,-191.03085280
lj75,control,12,lj,h2h_gmin_proof,4000000,-397.492331
lj98,control,12,lj,h2h_gmin_proof,4000000,-543.665361
sc108-80,control,12,sc,h2h_sc,1000000,-21819.9952
lj103,fail_rnext2,24,lj,h2h_gmin_proof,4000000,-575.766131
lj104,fail_rnext2,24,lj,h2h_gmin_proof,4000000,-582.086642
```

- `n = 24`: seeds `2000` through `2023` inclusive.
- `n = 12`: seeds `2000` through `2011` inclusive.

## Metrics and verdict rules

Per cell:
- Hits for `A`, `B`, `R`, `A2`; exact two-sided Fisher p-value (`B` vs `A`);
  one-sided tie-corrected Mann-Whitney U p-value (`B` worse than `A`,
  a miss scored as `budget + 1`).
- `B/A` median outside-time-per-call (`outside_s / charged`) ratio with
  seed-paired bootstrap 95% CI (10000 resamples, fixed seed `20261123`,
  Hyndman-Fan type 7 quantiles). Total `outside_s` ratio and CI are also
  recorded.
- `B/A` median `rss_anon_peak_kb` ratio with the same seed-paired bootstrap
  95% CI.
- `A2/A` band median and bootstrap 95% CI for outside time per call and
  `rss_anon_peak_kb`.
- Per-seed identity check (`best` and `parity_digest`) for `B` vs `R` and
  `B` vs `A`.
- Probe sanity: `rss_anon_peak_kb > 0` and `rss_reads >= 2` on every row;
  `vector_ranges > 0` on every row and nonzero `vector_ticks` on the
  vector-kernel path.

Verdict:
- A cell is `INCOMPLETE` until all `n` seeds have completed across all
  four arms (`A`, `B`, `R`, `A2`).
- Otherwise a cell is `PASS` only if:
  1. No hit loss (`not (hits_b < hits_a and fisher_p < 0.05)` and `mw_p >= 0.05`),
  2. Outside-time CI upper bound `outside_ratio_hi <= 1.10`, and
  3. RSS CI upper bound `rss_ratio_hi <= 1.10`.
- Each CI upper bound is also checked against `max(1.10, aa_upper)` where
  `aa_upper` is the cell's `A2/A` bootstrap 95% CI upper bound.
- Overall gate verdict is `PASS` only if every cell is `PASS`.
