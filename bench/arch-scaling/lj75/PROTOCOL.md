# LJ75 Worker Scaling Protocol

## Scope and Revision

- Base revision: `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b` (`develop`)
- Source tree invariant: `src/` is unmodified relative to base.
- System: `lj75` (`PairPotential::lennard_jones(75)` in `src/potentials.rs`, with `install_rgpot_kernels!()`).
- Reference minimum and hit tolerance: `-397.492331` with `DEFAULT_HIT_TOL = 1.0e-3` (`scripts/paper_0120/run_panel.py:57-58`, `src/target_tol.rs:4`).
- Total oracle-call budget per run: `4_000_000`, read by subscript from `PAPER_BUDGETS["lj75"]` in `scripts/paper_0120/run_panel.py:47`.

## Worker Mapping and Pinning

- Worker rungs: `p in {1, 2, 4, 8, 16}`.
- Seeds: `2000..=2011` (12 seeds per rung).
- Each rung runs `anneal_core::methods::ensemble::run_ensemble` (`src/methods/ensemble.rs:523`) inside a `rayon::ThreadPoolBuilder::new().num_threads(p)` pool with:
  - `cfg = Config::communicating(75)` (`src/methods/cluster_hopping/config.rs:1972`)
  - `replicas = p`
  - `budget = 4_000_000` split equally across replicas via `EnsembleConfig::budgets()`
  - `history = HistoryMode::Shared` (`shared = true`)
  - `membership = HistoryMembership::Accepted`
  - `checkpoint_interval = 1_000`
  - `target = Some(-397.492331 + 1.0e-3)`
  - `ExactStructureWitness`: memoized `SortedPairs { n_points: 75 }` at tolerance `CALIBRATION_IRA_TOLERANCE` (`1.0e-4`) behind `SerializedWitness`
- Note on `src/methods/ensemble.rs`: `run_ensemble` is the public ensemble entry point in `src/methods/ensemble.rs` and executes the `p` replicas concurrently on `p` OS threads via `std::thread::scope` (the symbol `run_minima_hopping_ensemble` is a private NVE-MD helper in `examples/lj_joint_optimum.rs:834` that hardcodes `Config::for_cluster`).
- Process pinning via `taskset -c`: for `p <= nproc`, pin to cores `0..(p - 1)` (`oversubscribed=false`); for `p > nproc`, pin to all available cores `0..(nproc - 1)` (`oversubscribed=true`) and exclude from parallel efficiency.

## Measurement and Execution Order

- Driver: `examples/arch_lj75_bench.rs` built in `--release` mode.
- Every timed sample must exceed `50 ms`.
- Memory: in-process peak `RssAnon` in KiB sampled every `10 ms` from `/proc/self/status`.
- CPU time: in-process `utime + stime` delta from `/proc/self/stat` divided by `CLK_TCK` (`100`).
- Ledger accounting: `report.aggregate_charged`, per-replica charged calls (`replica_charged`), and `ledger_sum` (`sum_i replica_charged[i] == total_oracle_calls`).
- Rows in `rows.csv` are append-only.
- Schedule rule: if projected wall time for all 60 runs plus warm-start exceeds ~4 h, execute the cheapest rungs first (`p = 8, 16, 4, ...`), commit completed rungs, and report empirical projections for remaining rungs without starting them.

## Statistical Summary

- Hit-rate comparison: two-sided Fisher's exact test.
- Calls-to-hit comparison: two-sided tie-corrected Mann-Whitney U test with misses coded as `budget + 1 = 4_000_001`.
- Confidence intervals: 10,000-resample percentile bootstrap 95% CIs with PRNG seed `20261123`.
- Parallel efficiency: `T1 / (p * Tp)` from wall-to-hit medians (omitting oversubscribed rungs `p > nproc`), with bootstrap 95% CIs when `T1` is measured.

## Follow-up: `p = 1` and Same-VM `p = 2` Control

- Binary: `examples/arch_lj75_bench`, release build of `5abbdf46a81ccbc78a04cdfd453d3f4eb4951326`. `src/` stays identical to `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b`.
- Budget: `load_budgets()["lj75"]`, which is `PAPER_BUDGETS["lj75"]` and must be `4_000_000`.
- Seeds: `2000..=2011`. One `p = 1` run and one repeated `p = 2` run per seed. No shortened budget.
- Command, unchanged: `taskset -c <cores> arch_lj75_bench scaling <p> <seed> <budget>`.
- Pinning: `p = 1` uses core `0`; `p = 2` uses cores `0-1`. Both have `oversubscribed=false` when `nproc >= 2`.
- Order: serial ABBA. `--rungs 1,2 --rotate` runs even seeds as `p = 1` then `p = 2` and odd seeds as `p = 2` then `p = 1`, so the stream is `1, 2, 2, 1` repeated.
- Launch: `python3 bench/arch-scaling/lj75/run_scaling.py --rungs 1,2 --rotate --session primary --control-workers 2 --control-session control`.
- Ledger: append-only. The header gains one trailing field, `session`. Existing data lines are not edited. `p = 1` rows use `session=primary`. Repeated `p = 2` rows keep `arm=comm-p2` and use `session=control`. They do not replace the original `p = 2` rows, whose `session` is blank.
- Each new row keeps every existing column and records `revision` (`git rev-parse HEAD` at launch), `nproc`, `cpu_model`, `pinned_cores`, `oversubscribed`, and in-process peak `RssAnon`.
- If the projected wall of both rungs exceeds about 4 h, finish `p = 1` (`--rungs 1 --session primary`), commit those rows, and report the `p = 2` control as a projection instead of starting it.
- Efficiency uses hit-conditioned wall-to-hit medians: `T1 / (p * Tp)` with a 10,000-resample percentile bootstrap 95% CI (seed `20261123`). Compute it (a) against the original rungs and (b) against the same-VM `p = 2` control. `p = 16` ran with `p > nproc` and is excluded. Hits use a two-sided Fisher exact test. Calls-to-hit uses a two-sided tie-corrected Mann-Whitney U test with a miss coded as `4_000_001`.
