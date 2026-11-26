# Communicating vs Independent vs Single-Chain LJ Ensemble Protocol

Baseline revision: `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b`

This protocol is locked and committed before any benchmark run or data artifact.

## 1. Systems, References, and Budgets

Total oracle-call budgets are read directly by subscript from `PAPER_BUDGETS` in `scripts/paper_0120/run_panel.py` (lines 44–56):

- `lj38`: `n_points = 38`, `PAPER_BUDGETS["lj38"] = 400_000`, reference `-173.928427`
- `lj75`: `n_points = 75`, `PAPER_BUDGETS["lj75"] = 4_000_000`, reference `-397.492331`
- `lj98`: `n_points = 98`, `PAPER_BUDGETS["lj98"] = 4_000_000`, reference `-543.665361`

Potential: reduced-unit Lennard-Jones via `PairPotential::lennard_jones(n_points)` with `anneal_core::install_rgpot_kernels!()`.

Target-hit predicate: `anneal_core::energy_hits(best, reference, anneal_core::SEARCH_HIT_TOL)` (`SEARCH_HIT_TOL = 1.0e-4` in `src/target_tol.rs`), mirrored in analysis scripts by `scripts/target_tol.py::energy_hits`.

## 2. Arm Definitions and Code Mapping

Ensemble size is `K = 4` for multi-replica arms. Each replica uses `Config::communicating(n_points)` (`src/methods/cluster_hopping/config.rs:1972`) and calls `anneal_core::methods::ensemble::run_ensemble` (`src/methods/ensemble.rs:523`), which splits `EnsembleConfig.budget` equally across replicas via `EnsembleConfig::budgets()` and derives deterministic per-replica seeds via `EnsembleConfig::replica_seeds(seed)`.

(Note: `run_minima_hopping_ensemble` in `examples/lj_joint_optimum.rs:834` is a private NVE-MD example helper; `src/methods/ensemble.rs::run_ensemble` is the library ensemble driver accepting `Config::communicating(n_points)` and `HistoryMode::Shared` / `HistoryMode::Private`.)

Common `EnsembleConfig` parameters across arms:

- `membership = HistoryMembership::Accepted`
- `shared_bias = None`
- `gossip = None`
- `two_choice_stall = None`
- `checkpoint_interval = 1000`
- `target = Some(reference + anneal_core::SEARCH_HIT_TOL)`

Common `EnsembleProblem` parameters across arms:

- `descriptor = anneal_core::catalog::lj::descriptor_space()` (`src/catalog/lj.rs:140`)
- `context = StructureContext::new(Some(vec![18; n_points]), None, Some(format!("lj-reduced-n{n_points}")))`
- `witness = sorted_pairs_witness(n_points, anneal_core::catalog::lj::CALIBRATION_IRA_TOLERANCE)` (`SerializedWitness` over `SortedPairs`)
- `same_family`: `!anneal_core::catalog::different_packing_family(a, b)`
- `certificate = 1e-5`, `polish_below = 1e-3`, `callbacks_per_objective = 1`

Arm configurations:

1. `comm`: `replicas = 4`, `history = HistoryMode::Shared` (one shared `Mutex<MinimumHistory>` across the 4 replicas).
2. `indep`: `replicas = 4`, `history = HistoryMode::Private` (four isolated `MinimumHistory` instances, one per replica).
3. `single`: `replicas = 1`, `history = HistoryMode::Private` (1 replica holding the full `budget`).

## 3. Shared Minima Store and Charged Ledger

- **Shared minima store**: Since `run_ensemble` maintains an in-memory `MinimumHistory` and does not write to disk, the driver opens `MinimaCorpus::open(&scratch_dir, lj::system_signature(n_points)?.digest())` (`src/minima_db.rs`) in an isolated temporary scratch directory outside committed repository paths and calls `MinimaCorpus::record(...)` with the distinct quenched minima (`best_state` / `final_state`) returned by the ensemble's replicas, recording the appended count (`corpus_appended`).
- **Build & ledger**: The binary is built with `--release --locked --features bank-rpc` (`Cargo.toml:83`). Because `run_ensemble` accounts work on per-replica `cluster_hopping::Ledger` instances rather than `CooperativeRun` (`src/cooperative_search.rs`), the driver reports the ensemble's own charged counters from `EnsembleReport`: `aggregate_charged` (verified equal to `sum(replica.charged)` and `sum(replica.outcome.charged)`), together with summed mechanism counters (`hops`, `screened_out`, `returned`, `basins`, `history_obs`, `history_new`, `history_refused`, `history_s`, `orbits_count`, `symmetrised_count`, `continuous_sym_count`, `visit_same`, `visit_known`, `visit_new`) and `first_target_calls`.

## 4. Seeds, Scheduling, Pinning, and Telemetry

- **Seeds**: `2000..=2011` (12 seeds per system).
- **System order**: Cheapest first (`lj38`, then `lj75`, then `lj98`).
- **Time-budget guard**: Before starting a system (`lj75` or `lj98`), a short sub-budget calibration measures per-call wall time; if elapsed campaign wall time plus projected full-budget wall time for the next system exceeds the 4-hour (`14,400 s`) campaign budget, execution stops cleanly before starting that system and reports the projection in `REPORT.md`.
- **Arm ordering**: ABBA-balanced per consecutive seed pair `(2k, 2k+1)` cycling all 6 permutations of `(comm, indep, single)` across the 12 seeds:
  - `(2000, 2001)`: `[comm, indep, single]`, `[single, indep, comm]`
  - `(2002, 2003)`: `[indep, comm, single]`, `[single, comm, indep]`
  - `(2004, 2005)`: `[comm, single, indep]`, `[indep, single, comm]`
  - `(2006, 2007)`: `[comm, indep, single]`, `[single, indep, comm]`
  - `(2008, 2009)`: `[indep, comm, single]`, `[single, comm, indep]`
  - `(2010, 2011)`: `[comm, single, indep]`, `[indep, single, comm]`
- **Process isolation & pinning**: Each `(system, arm, seed)` tuple runs serially in its own fresh process pinned to core 0 via `taskset -c 0` with `RAYON_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.
- **Resource & timing metrics**:
  - `wall_s`: monotonic `Instant::now()` elapsed seconds over the ensemble invocation (minimum valid sample duration >= 50 ms).
  - `cpu_s`: `getrusage(RUSAGE_SELF)` user + system CPU seconds delta across the run.
  - `peak_rss_anon_kib` and `vm_hwm_kib`: in-process `/proc/self/status` `RssAnon` sampled every 10 ms during execution (plus start/end) and final `VmHWM`.
  - `calls_to_hit`: `first_target_calls` when `hit == true`, else `budget + 1`.
- **Atomic persistence**: Each completed run appends one validated CSV row to `bench/arch-chains/lj/rows.csv` immediately and fsyncs before starting the next run.

## 5. Statistical Analysis

From `rows.csv`, `bench/arch-chains/lj/run.py --summarize` computes:

- `success_vs_calls.csv`: empirical hit fraction across call thresholds in `[0, budget]` for each `(system, arm)`.
- Pairwise comparisons (`comm` vs `indep`, `comm` vs `single`, `indep` vs `single`):
  - Two-sided Fisher's exact test on hit counts (`scripts/paired_stats.py::fisher_exact`).
  - Two-sided Mann-Whitney U test (normal approximation with tie and continuity corrections) on `calls_to_hit` with misses entered at `budget + 1`.
  - 10,000-resample bootstrap 95% confidence intervals (deterministic PRNG seed `20261123`) for hit-rate difference and median `calls_to_hit`, `wall_s`, `cpu_s`, and `peak_rss_anon_kib`.

## 6. Driver location

The single-run driver lives at `bench/arch-chains/lj/arch_chains_lj.rs`.
Cargo registers example name `arch_chains_lj` with that path and
`required-features = ["bank-rpc"]`. The example name is unchanged, so
the build command is still:

- `cargo build --release --features bank-rpc --example arch_chains_lj`

## 7. Sharding

The lj75 comparison covers seeds 2000 through 2011, arms `comm`,
`indep`, and `single`, and budget `4_000_000`. The global minimum is
`-397.4923309829`. Measurement used 4 shards of 3 seeds, serial per
VM, pooled after the four shards finished:

- seeds 2000, 2001, 2002
- seeds 2003, 2004, 2005
- seeds 2006, 2007, 2008
- seeds 2009, 2010, 2011

Rows are appended in that shard order. An earlier identical line is
not repeated. This record pools the finished shards.

Each shard stays expressible on the unified runner:

- `python3 bench/arch-chains/lj/run.py --systems lj75 --seeds 2000,2001,2002 --no-cap`
- `python3 bench/arch-chains/lj/run.py --systems lj75 --seeds 2003,2004,2005 --no-cap`
- `python3 bench/arch-chains/lj/run.py --systems lj75 --seeds 2006,2007,2008 --no-cap`
- `python3 bench/arch-chains/lj/run.py --systems lj75 --seeds 2009,2010,2011 --no-cap`

## 8. Unified runner filters

`--systems` and `--seeds` take comma-separated subsets. Surrounding
whitespace is ignored. Selected systems stay in `SYSTEMS` order. Each
selected seed keeps the `SEED_ARM_ORDER` entry of its original index
when the list is a subset or is written out of order. An omitted or
blank flag keeps the full list. `--no-cap` skips calibration and the
4-hour cap check. Unknown names, non-integer seeds, and empty tokens
are errors.

A finished run appends its rows, then rewrites `success_vs_calls.csv`
and `REPORT.md` from every row on disk.

The consolidated series is committed on develop `c425635bf564a989b180adc3311ded09066f73d6`. The measured
rows come from the shard series rooted at the baseline revision above.
