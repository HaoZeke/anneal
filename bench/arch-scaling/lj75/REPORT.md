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
