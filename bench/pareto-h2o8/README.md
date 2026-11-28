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

`rows-abba.csv` covers seeds 2000-2011.
Each seed is anneal, CREST, CREST, anneal, and each arm has two repeats.
Seeds 2000-2005 keep the wall time, cpu time, and final energy from this kit.
Seeds 2006-2011 keep those three fields from the rows already on this branch.
Seeds 2000-2005 use `host_tag` `da5ae9b9-9259-4a28-95e3-53ff0ab39756`.
That tag is the uuid4 from r2 `run.py` lines 174-182.
Seeds 2006-2011 keep the branch label `vm0`.
Those blocks are different VMs and different sessions, so wall times across the two blocks are not directly comparable.
CREST `wall_s` for 2006-2011 is `/usr/bin/time` elapsed wall clock at 0.01 s resolution (`crest_run.py` reads `Elapsed (wall clock)`).
CREST `wall_s` for 2000-2005 is a monotonic-clock reading.
`summary.json` key `abba_wall` reports `wall_s` and `cpu_s` inside each `host_tag`.
The two tags are not added together.
For `da5ae9b9-9259-4a28-95e3-53ff0ab39756`, anneal has n=12, wall_s_sum 15841.426931, wall_s_median 1317.0779865, cpu_s_sum 15839.816479, cpu_s_median 1316.914142, and CREST has n=12, wall_s_sum 12295.704538, wall_s_median 1069.591688, cpu_s_sum 12227.130000, cpu_s_median 1064.065000.
For `vm0`, anneal has n=12, wall_s_sum 15639.847620, wall_s_median 1301.8362105, cpu_s_sum 15638.099226, cpu_s_median 1301.694488, and CREST has n=12, wall_s_sum 11695.050000, wall_s_median 945.870000, cpu_s_sum 11646.220000, cpu_s_median 939.580000.

`total_calls` and `exit_code` for anneal seeds 2000-2005 are `energy_calls` and the process exit from each run's `run.txt`.
Those files are in `logs/anneal/`.
`logs/anneal-abba.txt` is those two fields, one line per run.
Those 12 runs spent 151738 calls and exited 0.
Seeds 2006-2011 have no ABBA anneal call log in this tree, so those two cells are `NA`.
Anneal `first_hit_call` is `calls_to_first_hit` in `rows-A.csv` or `rows-B.csv` for that seed, both repeats.
That column is the one from `f65da7d4`.
It is not the `ctoh` field on the campaign `SEED` lines.

CREST `total_calls` and `exit_code` for seeds 2000-2005 are the detail lines in `logs/campaign.txt`.
Seed 2001 left before the budget: repeat 1 finished at 146864 calls with exit 0, and repeat 2 finished at 99267 calls with exit 0.
Seeds 2006-2011 have no CREST log here, so `total_calls`, `first_hit_call`, and `exit_code` are `NA`.
CREST `first_hit_call` is `NA` on every row, including 2000-2005.
`logs/crest/*.txt` is the CREST stdout, and it does not list an energy for each xtb evaluation.
The compile banner's user@host token in those twelve files is replaced with `HOST`.
The wrapper event log stores optimization-cycle energies and one MD-block minimum, not an energy for every MD step, so that file is not a call index either.
The first evaluation at or below `-40.681336013150 + 1e-6` Eh is therefore not taken from those logs.
A scored miss would be 151739. These logs cannot score it.

A hit in this table is `best_Eh` at or below `-40.681336013150 + 1e-6` Eh.
Across 12 seeds and both repeats, 24 rows per arm: anneal has 24 hits, 0 misses, and 0 early exits in the 12 logged runs.
CREST has 10 hits and 14 misses.
Early exits, meaning `exit_code` 0 and `total_calls` below 151738, are the two CREST seed 2001 rows above.
The other CREST rows with a log reached 151738 calls.

## CREST geometry

`crest_run.py` compares seed 17000's `total_calls` with `RECORDED_17000` 151538, the value in `bench/h2o8-poi-vs-crest/inputs.csv`.
An optimization cycle at or below the threshold with no optimized geometry is not a hit. The event records `geometry_missing` when such a cycle was skipped.
No committed row was scored by that function. `inputs.csv` records source commits `42e708289` and `7ad8eab67`.
The ABBA call columns are the ones in the ABBA section.

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
