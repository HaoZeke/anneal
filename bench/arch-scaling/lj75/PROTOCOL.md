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
