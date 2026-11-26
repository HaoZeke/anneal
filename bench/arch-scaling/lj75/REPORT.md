# LJ75 Worker-Scaling Report

## Environment and Provenance

- Base revision: `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b` (`develop`)
- Source invariant: `git diff -- src/` is empty.
- Hardware: `8` logical cores (`nproc = 8`, `Intel(R) Xeon(R) Processor`), Linux `CLK_TCK = 100`.
- Total oracle-call budget: `4_000_000` (`PAPER_BUDGETS["lj75"]` in `scripts/paper_0120/run_panel.py:47`), split equally as `4_000_000 / p` per replica (`EnsembleConfig::budgets()`).
- Target threshold: `-397.492331 + 1.0e-3 = -397.491331` (`scripts/paper_0120/run_panel.py:57-58`).
- Entry point: `anneal_core::methods::ensemble::run_ensemble` (`src/methods/ensemble.rs:523`), spawning `p` OS threads inside a `rayon::ThreadPoolBuilder::new().num_threads(p)` pool, `Config::communicating(75)`, `HistoryMode::Shared`, `HistoryMembership::Accepted`.
- Driver: `examples/arch_lj75_bench.rs` (built in `--release` with `install_rgpot_kernels!()` and a memoized `SortedPairs` witness cache so history admission does not bottleneck on repeated descriptor sorting).

## Completed Rungs (`p = 2, 4, 8, 16`, 12 Seeds Each = 48 Runs)

Misses are coded as `4_000_001` in `calls_to_hit`. Confidence intervals use 10,000 percentile bootstrap resamples (PRNG seed `20261123`). Two-sided Fisher's exact test and tie-corrected Mann-Whitney U test compare each rung against `p = 2` (the smallest completed non-oversubscribed rung).

| Workers (`p`) | Per-Replica Budget | Hits | Hit Rate [95% CI] | Calls-to-Hit Med [95% CI] | Wall-to-Hit Med s [95% CI] | Full-Budget Wall Med s [95% CI] | CPU Med s [95% CI] | Peak RssAnon Med KiB | Rel Eff vs `p=2` | Fisher `p` vs `p=2` | MWU `p` vs `p=2` | Oversubscribed |
| ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 2,000,000 | 6/12 | 0.500 [0.250; 0.750] | 3,866,876 [1,571,770; 4,000,001] | 138.695 [37.024; 317.934] | 429.181 [370.829; 454.602] | 762.70 [728.49; 827.38] | 588,640 | 1.000 | 1.0000 | 1.0000 | false |
| 4 | 1,000,000 | 7/12 | 0.583 [0.333; 0.833] | 3,595,776 [1,324,869; 4,000,001] | 73.700 [42.672; 134.714] | 216.017 [198.052; 250.767] | 767.46 [657.75; 847.25] | 595,162 | 0.993 | 1.0000 | 0.8794 | false |
| 8 | 500,000 | 3/12 | 0.250 [0.000; 0.500] | 4,000,001 [2,778,598; 4,000,001] | 40.468 [37.268; 40.665] | 119.405 [106.223; 133.133] | 780.69 [713.19; 789.73] | 658,384 | 0.899 | 0.4003 | 0.2734 | false |
| 16 | 250,000 | 4/12 | 0.333 [0.083; 0.583] | 4,000,001 [3,028,252; 4,000,001] | 40.528 [4.127; 78.479] | 107.386 [105.570; 113.787] | 794.12 [773.24; 808.49] | 764,282 | NA | 0.6802 | 0.6288 | true |

### Additional Pairwise Comparisons

- `p = 4` vs `p = 8`: Fisher `p = 0.2138`, MWU `U = 49.5`, `p = 0.1561`.
- `p = 4` vs `p = 16`: Fisher `p = 0.4136`, MWU `U = 58.0`, `p = 0.3956`.
- `p = 8` vs `p = 16`: Fisher `p = 1.0000`, MWU `U = 78.0`, `p = 0.6926`.

## Ledger Invariant

Across all 48 completed runs, `sum(replica_charged) == ledger_sum == total_oracle_calls == 4_000_000`. Because `run_ensemble` divides `budget` equally (`4_000_000 / p` per replica) and runs to budget exhaustion rather than stopping early on first hit, higher `p` reduces per-replica search depth while proportionally reducing wall time up to the 8-core hardware limit (`1.99x` from `p=2` to `p=4` at `99.3%` relative efficiency; `3.59x` from `p=2` to `p=8` at `89.9%` relative efficiency). Oversubscribing to `p = 16` yields only `1.11x` wall speedup over `p = 8` (`107.386 s` vs `119.405 s`) while raising peak `RssAnon` from `658 MiB` to `764 MiB`.

## Unstarted Rung Projection (`p = 1`)

The `p = 1` rung (`comm-p1`, `1` worker at `4_000_000` calls, `taskset -c 0`) was not started because total wall time across the completed rungs (`p = 8` ~25 min, `p = 16` ~22 min, `p = 4` ~46 min, `p = 2` ~86 min) plus the companion warm-start suite (~45 min) reached ~3.7 h, and adding `p = 1` would exceed the ~4 h session cap.

Empirical projection for `p = 1` (`comm-p1`):

- From measured process CPU time across `p = 2..16` (`762.70 s` at `p = 2`, `767.46 s` at `p = 4`, `780.69 s` at `p = 8`, `794.12 s` at `p = 16`) extrapolated linearly to zero cross-thread contention at `p = 1`: `~750 s` (`~12.5 min`) median wall/CPU time per 4M-call run, or `~150 min` (`~2.50 h`) for 12 seeds.
- Cross-check from the single-thread calibration probe at `400,000` calls (`wall_s = 68.85 s`, `cpu_s = 68.83 s`, `peak_rss_anon_kib = 53,348 KiB` before witness memoization): scaling by `10x` to `4,000,000` calls gives an upper bound of `~689 s` plus witness growth (~`~750 s` total per seed, `~2.5 h` for 12 seeds).
- Including `p = 1` (`~2.50 h`) with the completed suite (`~3.73 h`) would have totaled `~6.23 h`.

## Follow-up Session (`p = 1` Measured)

Bootstrap intervals are 10,000 percentile resamples with seed `20261123`. A miss is coded `calls_to_hit = 4_000_001`. Fisher's exact test and the tie-corrected Mann-Whitney U test are two-sided against measured `p = 1`. Efficiency is `T1 / (p * Tp)` from hit-conditioned wall-to-hit medians. `p = 16` ran oversubscribed (`16 > nproc`, `nproc = 8`) and is excluded from efficiency.

### What Finished

- Release binary: `examples/arch_lj75_bench` at `5abbdf46a81ccbc78a04cdfd453d3f4eb4951326`. `src/` matches `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b`.
- New rows record `revision=bc850d9958410d8956b7a046abf775b4ea529b58`, `nproc=8`, `cpu_model=Intel(R) Xeon(R) Processor`, pinned cores, `oversubscribed=false`, and in-process peak `RssAnon`.
- `p = 1` (`session=primary`, core `0`, budget `4_000_000` on the one replica) finished seeds `2000..=2011`. Hits `9/12`, hit rate `0.750` `[0.500; 1.000]`. Full-budget wall sum `9,661.9 s` (`2.684 h`); median full-budget wall `792.159 s`; median CPU `792.08 s`; median peak `RssAnon` `596,098 KiB`.
- Seed `2000` ran `p = 1` (`836.940 s`) and then one same-VM `p = 2` control (`session=control`, cores `0-1`, `541.596 s`, miss). Twelve times that `p = 1` wall plus the original `p = 2` wall sum (`5,082.165 s`) is `4.202 h`, past the about-4 h cap, so seeds `2001..=2011` ran as `p = 1` only.

### Hits, Time to Hit, and Efficiency Against the Original Rungs

Hit counts: `p = 1` `9/12`, `p = 2` `6/12`, `p = 4` `7/12`, `p = 8` `3/12`, `p = 16` `4/12`.

| `p` | Session | Hits | Wall-to-Hit Med s [95% CI] | Calls-to-Hit Med [95% CI] | Fisher `p` vs `p=1` | MWU `U` | MWU `p` vs `p=1` | Efficiency [95% CI] | Oversubscribed |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | primary | 9/12 | 312.868 [129.631; 573.828] | 2,065,049 [1,190,956; 3,745,096] | | | | 1.000 (definition) | false |
| 2 | original | 6/12 | 138.695 [37.024; 317.934] | 3,866,876 [1,571,770; 4,000,001] | 0.4003 | 89.0 | 0.3278 | 1.128 [0.296; 4.577] | false |
| 4 | original | 7/12 | 73.700 [42.672; 134.714] | 3,595,776 [1,324,869; 4,000,001] | 0.6668 | 87.5 | 0.3776 | 1.061 [0.269; 2.238] | false |
| 8 | original | 3/12 | 40.468 [37.268; 40.665] | 4,000,001 [2,778,598; 4,000,001] | 0.0391 | 105.5 | 0.0417 | 0.966 [0.429; 1.764] | false |
| 16 | original | 4/12 | 40.528 [4.127; 78.479] | 4,000,001 [3,028,252; 4,000,001] | 0.0995 | 100.0 | 0.0950 | excluded | true |

`p = 8` has fewer hits than `p = 1` (`3/12` vs `9/12`, Fisher `p = 0.0391`) and higher calls-to-hit (MWU `U = 105.5`, `p = 0.0417`). For `p = 2`, `p = 4`, and `p = 16`, both tests against `p = 1` have `p >= 0.095`.

Efficiency against the original rungs is `1.128` `[0.296; 4.577]` at `p = 2`, `1.061` `[0.269; 2.238]` at `p = 4`, and `0.966` `[0.429; 1.764]` at `p = 8`. The intervals are wide and each contains `1`. `p = 1` is `1` by definition. `p = 16` is excluded from efficiency because it ran oversubscribed (`p > nproc`).

### Same-VM `p = 2` Control

The control rung has seed `2000` only, so efficiency against the control is omitted. This session does not decide whether the original-session `p = 2` and the control `p = 2` agree.

Seed `2000` missed in both sessions: original wall `553.267 s`, control wall `541.596 s`, ratio `0.979`. A 12-seed control projected from the original `p = 2` walls is `1.412 h` (`5,082.165 s`); scaling by `0.979` gives `1.382 h`. The 11 seeds still unrun are `1.258 h` unscaled and `1.231 h` scaled. Measured `p = 1` plus a full control is `4.096 h` unscaled and `4.066 h` scaled.

### Ledger

All 12 `p = 1` rows and the one control row have `sum(replica_charged) == ledger_sum == total_oracle_calls == 4_000_000`.
