# (H2O)8 full-budget rows

## Seeds and budget

Shard A is seeds 2000-2005. Shard B is seeds 2006-2011.
The budget is 151738 calls. A miss is scored as 151739.

## calls_to_first_hit

`calls_to_first_hit` is the first finite objective evaluation at or below the harness threshold, or 151739.
The threshold is reference `-40.681336013150` Eh plus `1e-6` Eh, times `27.211386245988`.
Polish evaluations count. The comparison is `energy <= threshold`.
The index is `first_hit_call` from `account.patch` applied to `examples/gfn2_mol_h2h.rs` at `b35601b7`, not the kit base.
`run.py` leaves `H2O_TARGET_EV` unset, so `Ledger::stop_below` (`src/methods/cluster_hopping.rs:797` at that commit) is not installed.
That function only stores a threshold. The quench watch (`src/methods/cluster_search.rs:1043`) and the polish stop (`src/methods/cluster_search.rs:1982`) read it after an evaluation, so the crossing index is the same whether or not the stop is armed. The next evaluation can differ.

Seeds 2001, 2005, and 2009 were rerun on this definition with `H2H_POLISH_ON_IMPROVE=1`, config `985afef7`, and `H2O_HIT_EV` set to the threshold.
With `H2O_TARGET_EV` set, `first_hit_call` was 627, 28376, and 20714, and charged calls were 1276, 30338, and 22666.
Without `H2O_TARGET_EV`, the per-call log matched through that index and the next line differed (628, 28377, 20715). Seed 2005 then ran the full budget and recorded `first_hit_call` 28376.
Those three indices are the `calls_to_hit` values in `bench/octamer-matched/h2o8/poi/results.csv`.

The previous column is not that crossing. On the reproduced seed 2005 trace, call 5432 is 0.576 eV above the threshold and an earlier call is already lower. On seed 2009, call 5531 is 1.012 eV above the threshold and call 3170 is already lower. Seed 2001's previous 629 is two evaluations after the crossing.
Seeds other than 2001, 2005, and 2009 were not rerun. Their column is the octamer-matched `calls_to_hit` for the same definition.

seed octamer-matched calls_to_hit calls_to_first_hit
2000 5874 5874
2001 627 627
2002 15513 15513
2003 6132 6132
2004 4070 4070
2005 28376 28376
2006 6254 6254
2007 6258 6258
2008 4388 4388
2009 20714 20714
2010 8009 8009
2011 3951 3951

## Harness

Shard A used `python3 bench/pareto-h2o8/run.py --shard A --binary BINARY --engine ENGINE --libdir LIBDIR`.
Shard B used `python3 bench/pareto-h2o8/run.py --shard B --binary BINARY --xtb-engine ENGINE --xtb-lib LIBDIR`.
Both set `H2H_POLISH_ON_IMPROVE=1` and `H2O_HIT_EV`, leave `H2O_TARGET_EV` unset, and read `first_hit_call`.
The binary includes `account.patch` on `examples/gfn2_mol_h2h.rs` at `b35601b7`.
The host label in `rows-A.csv` column `host_id_hash` is `vm-A`.
The host label in `rows-B.csv` column `host_id` is `vm0`.

`abba.py` calls the shard B anneal functions and `crest_run.py`.

## ABBA

For each seed 2006-2011 the order is anneal, CREST, CREST, anneal.
The 24 rows in `rows-abba.csv` ran serially on one VM on core 0.
The host column is `vm0`.
Anneal wall_s sum 15639.847620, median 1301.8362105.
CREST wall_s sum 11695.050000, median 945.870000.

## CREST geometry

`crest_run.py` compares seed 17000's `total_calls` with `RECORDED_17000` 151538, the value in `bench/h2o8-poi-vs-crest/inputs.csv`.
An optimization cycle at or below the threshold with no optimized geometry is not a hit. The event records `geometry_missing` when such a cycle was skipped.
No committed row was scored by that function. `inputs.csv` records source commits `42e708289` and `7ad8eab67`. `rows-abba.csv` has no call column.

## Methods

`python3 bench/pareto-h2o8/analyze.py` writes `rows.csv` and `summary.json`.
`rows.csv` has seeds 2000-2011 and a `shard` column. Shard A `best_eV` is empty because `rows-A.csv` has no `best_eV` column.
The arms are methods. Anneal is seeds 2000-2011 from `rows-A.csv` and `rows-B.csv`, n=12.
CREST first-hit calls are the 12 rows with `source` crest in `bench/h2o8-poi-vs-crest/inputs.csv`.
CREST seeds are 17000-17011. Those seeds differ from 2000-2011. The arms are not paired by seed.
The CREST call column is `first_hit_call`.
A hit is a recorded energy at or below reference `-40.681336013150` Eh plus `1e-6` Eh.
`run.py` and `analyze.py` both apply that test in eV: `energy <= reference * 27.211386245988 + 1e-6 * 27.211386245988`.
That is `energy_hits` against `REFERENCE_EV` and `TOL_EV` in `bench/octamer-matched/h2o8/run.py`.
The octamer driver also requires an intact geometry. These tables classify the stored `best_Eh`.
No committed row changes class. The anneal rows, `rows-abba.csv`, and the CREST rows in `bench/h2o8-poi-vs-crest/inputs.csv` keep the stored hit or miss.
A miss in the call median is scored as budget+1.
Anneal hits 12. Median 6193.0.
CREST hits 7. Median 141621.5.
Two-sided Fisher exact table [[12, 0], [7, 5]]. p_value 0.037267080745341616. odds_ratio_infinite true.
Two-sided exact Mann-Whitney U_anneal 0.0. p_value 7.39602301050679e-07.
Median ratio anneal over CREST 0.04372923602701567. Bootstrap 95% CI 0.028926273599831243 to 0.09657737850843338, 10000 draws, numpy default_rng seed 20261123.

## Seed stability

Shard A and shard B are the same anneal method on different seeds.
This block is a seed-stability check.
Shard A hits 6. Median 6003.0.
Shard B hits 6. Median 6256.0.
