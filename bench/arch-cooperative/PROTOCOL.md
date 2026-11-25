# Au55 cooperative arm

This file is the measurement protocol. It is committed before any row.

## Arm

System `au55`. Arm `cooperative`. One ensemble seed runs four chains
at once. The chains are threads in one process. A second process is
the catalog server, reached over a loopback ephemeral TCP port. The
release build is `--features "bank-rpc ira"`. `src/` is unchanged.
The linked shape library is commit
`f395c83a7cfc4c5858f60bcee9b02ce30bd42a61` of
`https://github.com/mammasmias/IterativeRotationsAssignments`,
compiled with GNU Fortran 13.3.0. `libira.so` sha256
`e0ffd8ed18611827056ce7d0e54a884cf5fdd88d2b32db7def6fa3a89631c6b9`.
`libira.a` sha256
`99b9343cce5195f04759055a7be665b70d004d10bafb657730f57f03108eb099`.

The search preset is `Config::communicating(55)` with Gupta
`copper_gold(0, 55)` species and no diameter rescaling. The run
then sets `shape_keyed` to true. That field is public and the
preset leaves it false; `keying` stays `Distances`, so the flag
selects shape keying. The replica count on that preset is held at
1. The four chains are four threads,
each running that single-replica preset, rather than one internal
ladder.
`basin_stride` is 8, because the Gupta surface does not read the
basin index. `polish_records` equals `relax_steps`. The reference
energy is `-191.03085280`. A hit is `energy_hits` at
`DEFAULT_HIT_TOL` (0.001).

## Budget and seeds

The call budget is `PAPER_BUDGETS["au55"]` from
`scripts/paper_0120/run_panel.py`. That total is split evenly across
the four chains. Seeds are 2000 through 2011, one seed at a time, in
ascending order. This delivery is one arm, so there is no second arm
to alternate. The process is pinned to one CPU. Thread pools are
held at one thread. There is no quick mode.

Replica 0 uses the ensemble seed for both the start draw and the
search stream, as two streams from the same seed, matching the metal
driver. Replica `r` uses `seed + r * 0x9E3779B9` (wrapping). A short
start fails the seed.

## Ledger

Each chain has a `CooperativeRun` of one replica and a hop ledger
with the same per-chain budget. A shared stop flag closes every
chain once any quenched energy is within tolerance of the reference.
`stop_below` uses the reference plus the tolerance, because that
helper tests the threshold with a zero tolerance.

Potential calls are taken from the hop ledger. Each relaxation
records one boundary with `record_quench_boundary`. The gradient is
the last charged evaluation when that point is the returned state
and its norm is at most `1e-3`; otherwise the boundary is rejected.

- a validated quench boundary is `AcceptedQuench`
- a rejected quench boundary is `RejectedQuench`
- any remaining calls in the interval are `AuxiliaryEvaluation`

Zero-call boundaries are not relabeled. Descriptor construction that
precedes an offer or a policy query is `DescriptorEvaluation` with
zero calls. An offer or policy result of local fallback or disabled
sharing is `RpcFallback` with zero calls. Adopting a deeper catalog
sample records `RemoteProposal` with zero calls.
`FreshValidation`, `Retry`, `BasinEscape`, `SaddleRide` and
`LocalProposal` stay at zero events. Server-side fresh checks sit
outside the replica budget, and this driver does not run those
mechanisms.

The hop loop can stop inside a slice. The uncheckpointed suffix of
`Ledger::quench_boundaries` is classified the same way, then each
chain flushes. `charge_sum` is the sum of charged calls over every
`ChargeKind`. `ensemble_total` is the sum of the four
`CooperativeLedger::ensemble_total` values. `oracle_calls` is the
sum of the hop ledgers. `server_aggregate` is the observer
`aggregate_charged` after the flushes. `terms_match` is 1 only when
those four numbers are equal. A mismatch is stored and reported.

## Catalog

The server is `CatalogServer` with `with_scientific_state`. The
signature is Gupta gold (atomic number 79), coordinate dimension
165, length and energy scales 1, universal descriptor schema
`anneal-universal-pes` version 2. The validator reference is 55
atoms on a line at 3 angstrom spacing, minimum separation 1,
maximum gradient norm `1e-3`, matching the search record gradient.
Catalog capacity is 64 and the census radius is 0.35. No state
directory, clock, quorum, or halving. The coordinator refuses a
scientific state that has no exact structure witness. The feature
set is `bank-rpc` and `ira`. The server passes the
`ExactStructureWitness` function implementation: two coordinate
vectors match when every component differs by at most `1e-4`.

Checkpoint interval is 500 charged calls. A validated boundary with
a finite energy and gradient norm at most `1e-3` is offered. Each
slice requests an incumbent catalog sample twice, the same poll the
existing catalog worker uses. A sample whose energy is deeper than
the local best by more than the hit tolerance replaces the live
state. Population epochs are not joined.

## Clock, memory, and rows

`calls_to_hit` is the sum of the four published hop-ledger spends at
the first quench whose returned energy hits. A miss is
`call_budget + 1`, not the amount spent. If the best energy hits but
no quench snapshot was taken, `calls_to_hit` falls back to the end
charged sum and `hit_clock` is `end`. Otherwise `hit_clock` is
`quench` or `miss`.

`wall_s` is the ensemble wall from chain start through join. A row
shorter than 50 ms is rejected. `cpu_s` is the parent user plus
system time plus the catalog process user plus system time over that
window. `peak_rss_anon` is the maximum resident anonymous memory, in
bytes, of the ensemble process and the catalog process, sampled
about every 100 ms. The sampler reads `RssAnon` from the rollup
when that field is present, and otherwise the rollup `Anonymous`
total.

Rows are appended to `bench/arch-cooperative/au55/rows.csv` and are
not rewritten. A seed already stored for the same system, arm, seed
and revision is skipped. The revision is the commit that was HEAD
when the harness started. `nproc` is the processor count in
cpuinfo, not the one pinned CPU. The CPU model is the first model
name, with commas turned into spaces.

Each chain's best state is appended to a shared minima corpus after
the joins. The corpus is not part of the recorded rows.

The row columns are:

```text
system,arm,seed,chains,oracle_calls,call_budget,hit,calls_to_hit,
hit_clock,best_energy,wall_s,cpu_s,peak_rss_anon,
local_proposal_events,local_proposal_calls,
remote_proposal_events,remote_proposal_calls,
descriptor_evaluation_events,descriptor_evaluation_calls,
accepted_quench_events,accepted_quench_calls,
rejected_quench_events,rejected_quench_calls,
fresh_validation_events,fresh_validation_calls,
retry_events,retry_calls,
rpc_fallback_events,rpc_fallback_calls,
auxiliary_evaluation_events,auxiliary_evaluation_calls,
basin_escape_events,basin_escape_calls,
saddle_ride_events,saddle_ride_calls,
charge_sum,ensemble_total,server_aggregate,terms_match,
binary,features,shape_keyed,libira_sha256,
revision,nproc,cpu_model
```

`binary` is `arch_cooperative`. `features` is `bank-rpc ira`.
`shape_keyed` is `true`. `libira_sha256` is the sha256 of the
linked `libira.so`.

`rows-bankrpc-only.csv` keeps twelve Au55 rows from a `bank-rpc`
build with `shape_keyed` false. That file is not a paper cell.
Its values are unchanged.

## Comparison

Hits use a two-sided Fisher exact test against the stored au55 rows
in `bench/paper-0120/rows.csv` whose driver is `h2h_metal` and whose
code is `anneal`, restricted to the seeds this arm finished. Calls
to the hit use the tie-corrected Mann-Whitney normal approximation
with a continuity correction. Stored misses, which were entered at
the budget, are recoded to `budget + 1` before that test, and so are
misses here. Bootstrap intervals are percentile 95 percent, 10000
resamples, generator seed 20261123, on the hit rate and on the
median calls to the hit.
