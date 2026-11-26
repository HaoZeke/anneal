# Polish-on-improve versus CREST, (H2O)8

Anneal rows are `bench/octamer-matched/h2o8/poi/results.csv` at `42e708289a718d3d916b46e5ae43f26cf0a677c1`.
Seeds 2000-2011. The call column is `calls_to_hit`.
Hits 12/12. Wilson 95% CI 0.757505993345 to 1.
Median 6193.

CREST rows are `bench/h2o8-crest-firsthit/rows.csv` at `7ad8eab67d61f1d5835a9ce7b67e7b2cb2b4ede9`.
The table contains seeds 17000-17011, n=12.
The call column is `first_hit_call`. Misses stay at the stored value 151739.
Hits 7/12. Wilson 95% CI 0.31951131255 to 0.806739686341.
Median 141621.5.

The hit counts use a two-sided Fisher exact test.
Table of hits and misses, anneal then CREST: [[12, 0], [7, 5]].
Odds ratio infinite. p = 0.0372670807453.

The call counts use a two-sided exact Mann-Whitney U test, anneal against CREST.
U = 0. p = 7.39602301051e-07.

The median ratio is the CREST median divided by the anneal median.
Ratio 22.8679961247.
Percentile bootstrap 95% CI 10.3543916334 to 34.5706472197, 10000 resamples, numpy default_rng seed 20261123.

These figures are the values written by `bench/h2o8-poi-vs-crest/stats.py` from `inputs.csv`.
