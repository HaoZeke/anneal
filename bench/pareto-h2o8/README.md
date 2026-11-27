# (H2O)8 full-budget rows

## Seeds and budget

Shard A is seeds 2000-2005. Shard B is seeds 2006-2011.
The budget is 151738 calls. A miss is scored as 151739.

## Harness

Shard A used `python3 bench/pareto-h2o8/run.py --shard A --binary BINARY --engine ENGINE --libdir LIBDIR`.
That path compiles the force-call logger, sets `H2H_POLISH_ON_IMPROVE=1`, and does not set `H2O_HIT_EV`.
`calls_to_first_hit` is the first logged finite force call at or below the harness threshold, or 151739.
The harness threshold is reference `-40.681336013150` Eh plus `1e-6` Eh.
The host label in `rows-A.csv` column `host_id_hash` is `vm-A`.

Shard B used `python3 bench/pareto-h2o8/run.py --shard B --binary BINARY --xtb-engine ENGINE --xtb-lib LIBDIR`.
That path runs the binary with `account.patch` applied, sets `H2H_POLISH_ON_IMPROVE=1` and `H2O_HIT_EV`, and reads `first_hit_call`.
The host label in `rows-B.csv` column `host_id` is `vm0`.

`abba.py` calls the shard B anneal functions and `crest_run.py`.

## ABBA

For each seed 2006-2011 the order is anneal, CREST, CREST, anneal.
The 24 rows in `rows-abba.csv` ran serially on one VM on core 0.
The host column is `vm0`.
Anneal wall_s sum 15639.847620, median 1301.8362105.
CREST wall_s sum 11695.050000, median 945.870000.

## rows-B and the stored stop-at-hit POI indices

`calls_to_first_hit` in `rows-B.csv` and `calls_to_hit` in `bench/octamer-matched/h2o8/poi/results.csv`:

seed 2006 rows-B 6379 POI 6254 best_Eh -40.681336013361 -40.681336013361
seed 2007 rows-B 7822 POI 6258 best_Eh -40.681336013361 -40.681336013361
seed 2008 rows-B 2925 POI 4388 best_Eh -40.681336013361 -40.681336013361
seed 2009 rows-B 5531 POI 20714 best_Eh -40.681336013361 -40.681336013361
seed 2010 rows-B 4451 POI 8009 best_Eh -40.681336013361 -40.681336013361
seed 2011 rows-B 3489 POI 3951 best_Eh -40.681336013361 -40.681336013361

The rows-B first-hit indices differ from the stored stop-at-hit POI indices. The final energies match.

## Methods

`python3 bench/pareto-h2o8/analyze.py` writes `rows.csv` and `summary.json`.
`rows.csv` has seeds 2000-2011 and a `shard` column. Shard A `best_eV` is empty because `rows-A.csv` has no `best_eV` column.
The arms are methods. Anneal is seeds 2000-2011 from `rows-A.csv` and `rows-B.csv`, n=12.
CREST first-hit calls are the 12 rows with `source` crest in `bench/h2o8-poi-vs-crest/inputs.csv`.
CREST seeds are 17000-17011. Those seeds differ from 2000-2011. The arms are not paired by seed.
The CREST call column is `first_hit_call`.
A hit is `best_Eh` within `1e-6` Eh of `-40.681336013361`. A miss in the call median is scored as budget+1.
Anneal hits 12. Median 4435.0.
CREST hits 7. Median 141621.5.
Two-sided Fisher exact table [[12, 0], [7, 5]]. p_value 0.037267080745341616. odds_ratio_infinite true.
Two-sided exact Mann-Whitney U_anneal 0.0. p_value 7.39602301050679e-07.
Median ratio anneal over CREST 0.03131586658805337. Bootstrap 95% CI 0.022996687923278462 to 0.05766491602153667, 10000 draws, numpy default_rng seed 20261123.

## Seed stability

Shard A and shard B are the same anneal method on different seeds.
This block is a seed-stability check.
Shard A hits 6. Median 4368.5.
Shard B hits 6. Median 4991.0.
