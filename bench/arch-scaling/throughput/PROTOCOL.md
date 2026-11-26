# Throughput scaling

This file is the measurement contract. Rows are appended only after it is
committed. Nothing under `src/` is edited. The release binary is built from
that tree plus the driver in `bench/arch-scaling/throughput/arch_throughput.rs` (example `arch_throughput`, `required-features = ["bank-rpc"]`).

## Question

How wall time and oracle throughput scale with replica count when a shared
communicating ensemble spends a fixed call budget to exhaustion.

## Systems and budget

Every cell spends `1000000` oracle calls. The search does not stop when the
best energy hits the reference. The count is the Au55 entry of
`PAPER_BUDGETS` in `scripts/paper_0120/run_panel.py` line 53. The same total
is applied to LJ75. The LJ75 entry on line 47 of that file is `4_000_000`
and is not the budget of this measurement. The driver refuses to start if
those two lines move.

Au55 uses `system_spec("au55")` in `examples/h2h_metal.rs` lines 79-84:
55 gold atoms, reference `-191.03085280`, copper count 0. The potential is
`Gupta::copper_gold(0, 55)`. The chain is `Config::communicating(55)` with
that species vector and no per-atom sigma.

LJ75 uses the reduced Lennard-Jones cluster of 75 atoms. The reference
`-397.492331` is `LJ75_REFERENCE` in `scripts/paper_0120/run_panel.py` line
57, and the same number is the 75-atom arm of `reference` in
`examples/lj_cluster_search.rs` line 1295. The potential is
`PairPotential::lennard_jones(75)`. The chain is
`Config::communicating(75)` with no species vector.

`Config::communicating` leaves `replicas` at 1, which is what
`EnsembleConfig::validate` requires of the chain.

A hit flag is `energy_hits(best, reference, DEFAULT_HIT_TOL)` with
`DEFAULT_HIT_TOL` equal to `1e-3`. The ensemble target is
`reference + 1e-3`. The flag is recorded and is not a stopping rule.

## Runner

The public runner is `run_ensemble` in `src/methods/ensemble.rs`. Each cell
calls it with:

- the communicating chain above
- `EnsembleConfig.replicas = p`
- `EnsembleConfig.budget = 1000000`
- `HistoryMode::Shared`
- `HistoryMembership::Accepted`
- no shared bias, no gossip, and no two-choice restart
- `checkpoint_interval = 1`
- `callbacks_per_objective = 1`
- certificate `1e-5` and polish bound `1e-3`

`EnsembleConfig::budgets` splits the total across replicas. For 1, 2, 4
and 8 the split is exact, so each replica receives `1000000 / p` calls.
One replica is the whole budget.

The ensemble already shares one `MinimumHistory`. `MinimaCorpus` is not
opened. `CooperativeRun` is not built, and the `bank-rpc` feature is off.

Identity is a sorted all-pairs distance vector. Two structures match when
every sorted distance agrees within `1e-3`.

## Workers and threads

`p` is the replica count. `run_ensemble` starts those replicas with
`std::thread::scope`, one operating-system thread each. It does not build
a rayon pool, and it does not run the replicas one after another. The
driver pins the calling thread with `sched_setaffinity` to cores
`0..p-1` before the scope, so the replica threads inherit that mask.
`RAYON_NUM_THREADS` and `OMP_NUM_THREADS` are 1.

Cells of this campaign are themselves serial: one ensemble returns before
the next starts.

`nproc` is the online processor count from `sysconf`, read before any pin
and checked against `/proc/cpuinfo`. A cell with `p > nproc` still runs,
its mask is the online set, and `oversubscribed` is true. That cell is
left out of efficiency.

## Ledger and timing

`wall_s` runs from immediately before `run_ensemble` to its return. It
must be at least 50 ms or the row is refused.

Each objective return increments one counter. `ledger_objective` is that
count. `ledger_charged` is `EnsembleReport.aggregate_charged`.
`ledger_ensemble` is the sum of the per-replica `charged` fields. The
driver writes a row only when those three are equal. `oracle_calls` is
that common total. `calls_minus_budget` is `oracle_calls - 1000000`.
`calls_per_s` is `oracle_calls / wall_s`.

A nonzero `calls_minus_budget` is a deviation and is reported. It does not
stop the remaining cells.

## Design

Systems are `au55` and `lj75`. Seeds are 2000 through 2011. Worker counts
are 1, 2, 4 and 8, all on the shared arm (`arm = comm`). There is no
16-worker rung.

For seed `2000 + i` with `i` even, the worker order is 1, 2, 4, 8 and,
within a worker count, Au55 runs before LJ75. With `i` odd, the worker
order is 8, 4, 2, 1 and LJ75 runs before Au55. A cell already stored for
the same system, arm, seed, and worker count is left in place.

The launch sets `THROUGHPUT_DEADLINE_SECS` to 14400. Once that many
seconds have passed, the driver finishes the cell it is in and does not
start another. An unset deadline does not stop the ladder.

`probe` times a caller-chosen call count and does not append a row.

## Statistics

For each system and worker count, over the stored seeds whose three ledger
fields agree:

- Median wall time, with a bootstrap 95% interval.
- Geometric-mean wall time, with a bootstrap 95% interval.
- Median and geometric-mean calls per second, each with a bootstrap 95%
  interval. The rate of a seed is its own `oracle_calls / wall_s`.
- Parallel efficiency `T1 / (p * Tp)` from the median wall times. The
  1-worker cell is the baseline. Oversubscribed cells are omitted. The
  interval resamples seed indices jointly, recomputes both medians, and
  takes the 2.5 and 97.5 percentiles of the ratio.

The geometric mean is the exponential of the arithmetic mean of the
logs. The median of an even sample is the mean of the two central order
statistics. Percentiles interpolate linearly on the sorted draws, with
position `p * (n - 1)`.

Bootstrap draws use Python `random.Random` seeded with 20261123 and 10000
resamples. One generator serves every interval, in this order: systems
`au55` then `lj75`; within a system, worker counts ascending; within a
count, the median wall, the geometric-mean wall, the median rate, then
the geometric-mean rate. Efficiency follows, for the same system order
and for each non-oversubscribed count above 1, ascending. The 1-worker
efficiency is 1 and draws no samples.

Hit counts are printed and are not part of the intervals.

## Rows

`rows-throughput.csv` is append-only. The header is written once:

```
system,arm,seed,workers,wall_s,oracle_calls,call_budget,calls_minus_budget,calls_per_s,ledger_objective,ledger_charged,ledger_ensemble,hit,best_energy,revision,nproc,cpu_model,pinned_cores,oversubscribed
```

`revision` is `git rev-parse HEAD` of the tree that built the binary.
`pinned_cores` is the affinity mask read back after the pin. `cpu_model`
is the processor model string.
