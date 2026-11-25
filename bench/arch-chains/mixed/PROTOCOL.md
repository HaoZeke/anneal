# Communicating chains, independent chains, and one chain

K is 4. Each seed spends one total oracle-call budget. The three arms
differ in how that budget is divided and whether minima are shared.

## Arms

The runner is `run_ensemble` in `src/methods/ensemble.rs`. A private
function named `run_minima_hopping_ensemble` lives in
`examples/lj_joint_optimum.rs`. It takes a Lennard-Jones pair potential
and does not take `Config`, so it cannot evaluate these systems. The
public ensemble in `src/methods/ensemble.rs` does take `Config`.

`comm` sets `replicas = 4` and `HistoryMode::Shared`: one
`MinimumHistory` behind a lock, and every replica uses that history.
`indep` uses the same replica count with `HistoryMode::Private`: one
history per replica and no shared visits. `single` sets `replicas = 1`
and `HistoryMode::Private`, and that replica receives the whole budget.
`EnsembleConfig::budgets` splits the aggregate budget equally, with the
remainder on the low replica indices.

Shared bias, gossip, and the two-choice restart stay off on every arm,
so the history mode is the only channel that changes. Membership is
`HistoryMembership::Accepted`. Checkpoints are every 1000 charged calls.
That interval is the resolution of `calls_to_hit`.

`MinimumHistory` is the shared minima store when the history mode is
shared. `MinimaCorpus` in `src/minima_db.rs` is a durable corpus opened
by path. This ensemble does not read or write it, and a second store is
not added. `CooperativeRun` is not on this path. The oracle ledger is
the ensemble charge counter: `aggregate_charged`, also recorded as
`ledger_sum`. Replica spent counts are recorded as
`ledger_replica_sum`. The remaining `ledger_*` columns are the
ensemble's history and exchange counters. They are not charge-kind
addends, and they are not summed into `ledger_sum`.

## Budgets and systems

Au55 uses `PAPER_BUDGETS["au55"]` in `scripts/paper_0120/run_panel.py`
line 53, which is 1000000 calls. The builder and reference are
`system_spec("au55")` in `examples/h2h_metal.rs` lines 79-84: 55 atoms,
reference energy -191.03085280, pure gold. The potential is
`Gupta::copper_gold(0, 55)`. Each replica's configuration is
`Config::communicating(55)` with those gold labels attached. A hit is
`energy_hits` at `DEFAULT_HIT_TOL` (1e-3) from `src/target_tol.rs` line 4.

TIP4P (H2O)20 uses `PAPER_BUDGETS[20]` in
`scripts/tip4p_h2h/run_campaign.py` line 30, which is 2000000 calls.
The reference is the same file line 31, -872.9887544952, which is also
`data/tip4p_ccd.csv` line 20. Each replica uses
`Config::communicating_tip4p(20)`. The rigid geometry is then set to
the water geometry installed by `examples/h2h_seed.rs` `water_config`,
and the objective is the rigid TIP4P potential that driver evaluates.
The hit window is the same 1e-3.

The ensemble quench certifies on the Euclidean gradient norm and only
keeps polishing while that norm sits between the certificate and
`polish_below`. A rigid chain records a minimum when the
root-mean-square force is at or below `record_gradient` (1e-3 on this
preset). For water the certificate is half of that root-mean-square,
written as a Euclidean norm over the 120 rigid coordinates, and
`polish_below` is 1e6, so a quench that is still loose after the
configured 80 steps continues until the record window. Au55 keeps the
certificate at `record_gradient` and polishes below ten times that
value. The 80-step preset already reaches it.

The openconf ligand is dropped from this panel.
The openconf ligand (rq3) was not run: rgpot has no MMFF potential, and run_ensemble needs an energy and gradient.

Au55 identity is the sorted pair-distance witness used when the shape
matcher is not linked, with radius 0.02. TIP4P identity matches
lab-frame oxygen and hydrogen sites of the rigid chart within 0.05
angstrom, hydrogens interchangeable inside a molecule. The descriptor
passed to the history is the universal descriptor at the configuration
length scale. It orders witness checks and does not decide identity.

## Measurement

Seeds are 2000 through 2011. Within a seed the arm order is ABBA: an
even offset from 2000 runs comm, indep, single, and an odd offset runs
single, indep, comm. One arm runs at a time. The process is pinned with
`taskset -c 0`. Replica threads are the ensemble's own threads on that
one CPU. `nproc` is the machine's online processor count, not the
affinity mask. No run uses a quick flag. Wall time and process CPU time
cover the ensemble call. Peak memory is the highest `RssAnon` value
read from the process status during that call. A miss stores
`calls_to_hit` as budget + 1.

Rows are appended to `rows.csv` and are not rewritten. After the rows
for a system are complete, `success_vs_calls.csv` is the cumulative
fraction of seeds that have hit by each aggregate call count, and
`REPORT.md` gives hits, the calls-to-hit median with a bootstrap 95%
interval, ledger sums, and the tests. The hit test is Fisher exact. The
calls-to-hit test is Mann-Whitney, with a miss scored as budget + 1.
Bootstrap intervals use 10000 resamples and seed 20261123.
