# LJ75 Warm-Start vs Cold-Start Protocol

## Scope and Revision

- Base revision: `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b` (`develop`)
- Source tree invariant: `src/` is unmodified relative to base.
- System: `lj75` (`PairPotential::lennard_jones(75)` in `src/potentials.rs`, with `install_rgpot_kernels!()`).
- Reference minimum and hit tolerance: `-397.492331` with `DEFAULT_HIT_TOL = 1.0e-3` (`scripts/paper_0120/run_panel.py:57-58`, `src/target_tol.rs:4`).
- Call budget per run: `4_000_000`, read by subscript from `PAPER_BUDGETS["lj75"]` in `scripts/paper_0120/run_panel.py:47`.

## Corpus Construction

- Build a `MinimaCorpus` (`src/minima_db.rs`) keyed by `system_signature(75).digest()` (`src/catalog/lj.rs:163`) from 12 single-chain runs (`seeds 1000..=1011`) using `Config::communicating(75)` and `cluster_hopping::run_with_gradient` at full `4_000_000` budget, pinned to core `0` (`taskset -c 0`).
- Record every `QuenchStatus::Validated` quench boundary and final `best_state` per seed via `MinimaCorpus::record`.
- Record the resulting corpus size and SHA-256 digest; commit the corpus archive only if its total size is at most `5 MB`.

## Paired Evaluation (Warm vs Cold)

- Evaluation seeds: `2000..=2011` (12 seeds per arm).
- Single-chain configuration: `Config::communicating(75)` via `cluster_hopping::run_with_gradient` with `ledger.stop_below(-397.492331 + 1.0e-3)`, pinned to core `0` (`taskset -c 0`).
- `cold` arm: starts from `random_cluster_in_radius(75, cfg.start_radius(), cfg.min_separation, &mut rng)` seeded by `seed`.
- `warm` arm: reads `corpus.minima("lj75", cfg.temperature)` and seeds `start` from the lowest-energy stored minimum strictly above the hit threshold (`energy > -397.492331 + 1.0e-3`) so no run begins already at the target minimum.
- Ordering: ABBA across seeds (`i % 4 in {0, 3}` runs `cold` then `warm`; `i % 4 in {1, 2}` runs `warm` then `cold`).
- Rows in `rows.csv` are append-only; timed samples must exceed `50 ms`.

## Statistical Summary

- Hit comparison: two-sided Fisher's exact test (and paired McNemar exact test reported alongside).
- Calls-to-hit comparison: two-sided tie-corrected Mann-Whitney U test with misses coded as `budget + 1 = 4_000_001`.
- Confidence intervals: 10,000-resample percentile bootstrap 95% CIs with PRNG seed `20261123`.
