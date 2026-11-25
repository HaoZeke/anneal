# Au55 worker scaling

This file is the measurement contract. Rows are appended only after it is
committed. Nothing under `src/` is edited. The release binary is built from
that tree plus the driver in `examples/au55_arch_scaling.rs`.

## Question

How the wall time and the oracle-call count to the Gupta Au55 minimum scale
when the communicating ensemble grows from 1 to 16 replicas, at one fixed
total oracle-call budget.

## System and budget

The total oracle-call budget is the module-level `PAPER_BUDGETS` entry for
`au55` in `scripts/paper_0120/run_panel.py`, read by subscript. At this
revision that entry is line 53 and the value is 1000000. The driver reads
that line and refuses to run if the token is not `1_000_000`.

The builder and the reference energy are `system_spec("au55")` in
`examples/h2h_metal.rs` lines 79-84: 55 gold atoms, reference
`-191.03085280`, copper count 0. The potential is
`Gupta::copper_gold(0, 55)`. The search configuration is
`Config::communicating(55)` with that species vector and no per-atom sigma.
`Config::communicating` leaves `replicas` at 1, which is what
`EnsembleConfig::validate` requires of the chain.

A hit is `energy_hits(best, reference, DEFAULT_HIT_TOL)` with
`DEFAULT_HIT_TOL` equal to `1e-3`. The ensemble target is
`reference + 1e-3`, and its own check uses a zero tolerance, so the two
tests are the same inequality.

## Runner

`run_minima_hopping_ensemble` is a private function in
`examples/lj_joint_optimum.rs`. It takes a pair potential, not
`Config::communicating`, and not the Gupta potential. The public runner in
`src/methods/ensemble.rs` is `run_ensemble`. Every measured cell calls
`run_ensemble` with:

- `Config::communicating(55)` as above
- `EnsembleConfig.replicas = p`
- `EnsembleConfig.budget = 1000000`
- `HistoryMode::Shared` (the shared arm)
- `HistoryMembership::Accepted`
- no shared bias, no gossip, and no two-choice restart
- `checkpoint_interval = 1`
- `callbacks_per_objective = 1`
- certificate `1e-5` and polish bound `1e-3`, the reduced-unit values named
  on `validated_relax`

`EnsembleConfig::budgets` divides the total budget across replicas, with
the remainder on the low indices. For 1, 2, 4, 8 and 16 the split is exact,
so each replica receives `1000000 / p` calls. One replica is the whole
budget.

The ensemble already stores minima in one `MinimumHistory` behind a lock
and gives every replica a `SharedMinimumHistory` on that history.
`MinimaCorpus` is not opened. `CooperativeRun` and `CooperativeLedger` are
not built, and the `bank-rpc` feature is off.

Identity for the shared history is a sorted all-pairs distance vector.
Two structures match when every sorted distance agrees within `1e-3`.

## Workers, threads, and pins

`p` is the replica count. `run_ensemble` starts those replicas with
`std::thread::scope`, one operating-system thread each. It does not build
a rayon pool, and it does not run the replicas one after another. Before
the scope, the driver pins the calling thread with `sched_setaffinity` to
cores `0..p-1`. Replica threads and the memory sampler inherit that mask.
`RAYON_NUM_THREADS` is set to 1 so a library pool cannot add workers on
top of the replica threads. `OMP_NUM_THREADS` is set to 1 for the same
reason.

`nproc` is the online processor count, read with `sysconf` before any pin,
and checked against `/proc/cpuinfo`. A rung with `p > nproc` still runs.
Its affinity mask is the online set, `oversubscribed` is true, and the rung
is left out of parallel efficiency.

## Ledger

The oracle ledger has one term, `charged`, taken from
`EnsembleReport.aggregate_charged`. `ledger_sum` is the sum of the
per-replica `charged` fields. The driver refuses a row when those two
differ. `oracle_calls` is that same total. Objective returns are stamped
in completion order; their count must equal `charged` or the row is
refused.

These ensemble counters are recorded and are not oracle calls, so they are
not added into `ledger_sum`: shared-history minima, accepted identities,
visits, observations, refusals, exchange published, exchange delivered,
bias deposits published, and two-choice restarts. With the channels above,
the exchange, bias, and two-choice counters stay at zero.

## Timing and memory

`wall_s` is the elapsed time from immediately before `run_ensemble` to its
return. `cpu_s` is the process user plus system time over that same
interval, from `/proc/self/stat`. A row is kept only when `wall_s` is at
least 50 ms.

On a hit, `calls_to_hit` is `EnsembleReport.first_target_calls`, the
aggregate charged count at the checkpoint where the best energy first met
the target. `wall_to_hit_s` is the stamp of objective completion number
`calls_to_hit` on the same clock as `wall_s`. On a miss, `calls_to_hit` is
`call_budget + 1` and `wall_to_hit_s` is empty. The search does not stop
at the hit; the budget is spent.

`peak_rss_anon` is the maximum `RssAnon` value, in bytes, sampled inside
the process from `/proc/self/status` about every 10 ms during the call,
including one sample before and one after.

There is no shortened measurement mode. A separate `probe` command may
time a caller-chosen call count and does not append a row.

## Design

Seeds are 2000 through 2011. Rungs are 1, 2, 4, 8, 16 workers, all on the
shared arm (`arm = comm`). For seed `2000 + i` the rung order is the
worker list rotated left by `i mod 5`, so each rung is first for some
seeds and last for others. A cell already stored for the same system, arm,
seed, and worker count is left in place and not run again.

## Statistics

For each rung, over the twelve seeds:

- Hits, with a bootstrap 95% interval on the hit count.
- Fisher's exact test, two-sided, of that rung's hits against the
  1-worker hits.
- Calls to hit, with a miss entered as `budget + 1`. The reported centre
  is the median. The interval is a bootstrap 95% interval of that median.
  The Mann-Whitney U test against the 1-worker sample is two-sided, with
  the tie-corrected normal approximation and a continuity correction of
  one half.
- Wall time to hit, median over the seeds that hit, with a bootstrap 95%
  interval. Misses are absent from this median.
- Parallel efficiency `T1 / (p * Tp)` from those wall-to-hit medians.
  The 1-worker rung is the baseline. Oversubscribed rungs are omitted.
  The interval resamples seed indices jointly, recomputes the two
  medians from the hits in the draw, and takes the 2.5 and 97.5
  percentiles of the ratio. A draw with no hit on either rung is dropped.

Bootstrap draws use Python `random.Random` seeded with 20261123, 10000
resamples. One generator serves every interval, in this order: worker
counts ascending, and within a count the hit count, then the calls-to-hit
median, then the wall-to-hit median; then efficiency for each
non-oversubscribed count above 1, ascending.

## Rows

`rows.csv` is append-only. The header is written once:

```
system,arm,seed,workers,oracle_calls,call_budget,hit,calls_to_hit,wall_to_hit_s,best_energy,wall_s,cpu_s,peak_rss_anon,ledger_charged,ledger_sum,hist_minima,hist_accepted,hist_visits,hist_observations,hist_refusals,exchange_published,exchange_delivered,bias_published,two_choice_restarts,objective_calls,revision,nproc,cpu_model,pinned_cores,oversubscribed
```

`revision` is `git rev-parse HEAD` of the tree that built the binary.
`pinned_cores` is the affinity mask read back after the pin. `cpu_model`
is the processor model string.
