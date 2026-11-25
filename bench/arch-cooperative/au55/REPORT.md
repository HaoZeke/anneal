# Au55 cooperative rows

Revision `76bb4afddba992f0f8758094610ec80564e0a81f`. Seeds finished: 12.
Hits: 0 of 12.
Hit rate 0.0000, bootstrap 95 percent interval [0.0000, 0.0000].
Calls to hit, median 1000001.0, bootstrap 95 percent interval [1000001.0, 1000001.0]. A miss is the budget plus one (1000001).
Wall sum 1036.306 s. Mean wall 86.359 s.
CPU sum 1035.950 s.
Oracle-call sum 12000000.
Ledger charge sum 12000000.
Ensemble total sum 12000000.
Server aggregate sum 12000000.

Per seed:

| seed | hit | calls_to_hit | wall_s | charge_sum | terms_match |
| --- | --- | --- | --- | --- | --- |
| 2000 | 0 | 1000001 | 88.351155 | 1000000 | 1 |
| 2001 | 0 | 1000001 | 91.088684 | 1000000 | 1 |
| 2002 | 0 | 1000001 | 89.778436 | 1000000 | 1 |
| 2003 | 0 | 1000001 | 84.355983 | 1000000 | 1 |
| 2004 | 0 | 1000001 | 85.337632 | 1000000 | 1 |
| 2005 | 0 | 1000001 | 84.532128 | 1000000 | 1 |
| 2006 | 0 | 1000001 | 86.670724 | 1000000 | 1 |
| 2007 | 0 | 1000001 | 86.878356 | 1000000 | 1 |
| 2008 | 0 | 1000001 | 86.490810 | 1000000 | 1 |
| 2009 | 0 | 1000001 | 84.165557 | 1000000 | 1 |
| 2010 | 0 | 1000001 | 84.229893 | 1000000 | 1 |
| 2011 | 0 | 1000001 | 84.426667 | 1000000 | 1 |

Named charge terms match the charged total on every finished seed.

Against stored au55 `h2h_metal` rows for the same seeds: hits 0 versus 6, Fisher two-sided p 0.01373.
Calls to hit, Mann-Whitney two-sided p 0.007054. Stored misses were recoded from the budget to the budget plus one.

Intervals use 10000 resamples and generator seed 20261123. The hit-rate interval is drawn first, then the calls-to-hit median. Fisher and Mann-Whitney are computed in this harness without an external statistics package.
