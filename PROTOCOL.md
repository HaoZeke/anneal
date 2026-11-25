# Cooperative architecture protocol

## Scope

This protocol measures the `lj75` CooperativeRun arm at revision
`3ccb2ad4fb930270f0fe95568ff09a5d0127d81b`. The source tree under
`src/` remains unchanged. The release examples are built with the
`bank-rpc` feature.

The exact Cargo feature set is `bank-rpc ira`. The `ira` feature is
required by the out-of-process catalog coordinator target; enabling it
does not change the CooperativeRun arm.

IRA is built outside this repository from
`https://github.com/mammasmias/IterativeRotationsAssignments` at
`f395c83a7cfc4c5858f60bcee9b02ce30bd42a61`. The compiler is
`GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`.
The SHA-256 digests are:

- `libira.so`:
  `17d08d412418128ff5a345501b2e671c03694913ee5c1d5e6b7de8014babbd90`
- `libira.a`:
  `076fdd3ec4d8a875dabe88d9222701312a9edad5e4edf19ba6e891db72a06191`

The sole measured arm uses four communicating chains and one
out-of-process catalog coordinator. The chain configuration is
`Config::communicating(75)` with `shape_keyed` explicitly enabled by
the `shape` run option. Every seed receives the `lj75` entry in
`PAPER_BUDGETS` as one aggregate oracle-call budget, divided equally
between the four chains. The shared coordinator receives the same
aggregate budget.

## Schedule

The measured seeds are 2000 through 2011, in ascending order. The four
chain seeds for ensemble seed `s` are
`s + replica * 0x9e3779b97f4a7c15`, matching the repository ensemble
seed expansion. Seeds run serially. The coordinator and four chains of
one seed run concurrently because that concurrency defines the arm.

There is only one requested arm, so there is no distinct arm pair to
order as ABBA. No comparison arm is introduced.

The coordinator is pinned to CPU 4 and replicas 0 through 3 are pinned
to CPUs 0 through 3. Thread-count environment variables are fixed at
one. The catalog capacity is 30, census radius is 0.1, checkpoint slice
is 500 charged calls, and population interval is 50,000 charged calls.

## Measurements

One append-only CSV row is written after each completed ensemble seed.
The row records the system, arm, seed, chain count, worker count,
aggregate oracle calls, call budget, hit status, catalog-observed
aggregate calls to the first target encounter, best energy, wall time,
CPU time, and peak anonymous resident memory.

CPU time is the sum of user and system CPU time for the coordinator and
all chains. Memory is sampled from each live process's `RssAnon` entry;
the row stores the peak sum across the five measured processes. Wall
time uses a monotonic clock around the complete ensemble.

Every `ChargeKind` records both event count and charged-call count.
The row also stores the sum of all named charged-call terms, the
workers' summed `ensemble_total`, and the coordinator's final aggregate
charge. Any disagreement makes the run fail before a row is appended.

A hit is an energy within the repository search tolerance of
-397.492331. Calls to hit are the earliest aggregate ledger value on a
catalog slice that reports a best energy meeting that rule. A miss is
entered as budget plus one for calls-to-hit statistics.

Each timed sample must run for at least 50 ms. Rows name the source
revision, logical processor count, and CPU model. Raw process output is
temporary and is not committed.

## Analysis

Summary confidence intervals use 10,000 bootstrap resamples with seed
20261123. The report gives a hit-rate interval, a calls-to-hit median
interval with misses entered as budget plus one, wall-time interval,
and ledger-sum interval.

Fisher's exact test and the Mann-Whitney U test require a second arm.
Because this task requests only CooperativeRun and names no baseline,
pairwise tests are reported as unavailable rather than computed against
an invented comparator.
