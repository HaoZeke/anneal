# Au55 cooperative rows

Revision `b1eef78e89bd34d1accabed74aa9addb71e100bd`. Seeds finished: 12.
Hits: 1 of 12.
Hit rate 0.0833, bootstrap 95 percent interval [0.0000, 0.2500].
Calls to hit, median 1000001.0, bootstrap 95 percent interval [1000001.0, 1000001.0]. A miss is the budget plus one (1000001).
Wall sum 1128.483 s. Mean wall 94.040 s.
CPU sum 1128.299 s.
Oracle-call sum 11423090.
Ledger charge sum 11423090.
Ensemble total sum 11423090.
Server aggregate sum 11423090.
Binary arch_cooperative. Features bank-rpc ira. shape_keyed true.
libira.so sha256 e0ffd8ed18611827056ce7d0e54a884cf5fdd88d2b32db7def6fa3a89631c6b9.

Per seed:

| seed | hit | calls_to_hit | wall_s | charge_sum | terms_match |
| --- | --- | --- | --- | --- | --- |
| 2000 | 0 | 1000001 | 96.846759 | 1000000 | 1 |
| 2001 | 0 | 1000001 | 94.178854 | 1000000 | 1 |
| 2002 | 0 | 1000001 | 93.496107 | 1000000 | 1 |
| 2003 | 0 | 1000001 | 94.545111 | 1000000 | 1 |
| 2004 | 0 | 1000001 | 98.937346 | 1000000 | 1 |
| 2005 | 0 | 1000001 | 100.152458 | 1000000 | 1 |
| 2006 | 0 | 1000001 | 97.198794 | 1000000 | 1 |
| 2007 | 0 | 1000001 | 101.056456 | 1000000 | 1 |
| 2008 | 0 | 1000001 | 108.880452 | 1000000 | 1 |
| 2009 | 0 | 1000001 | 101.534491 | 1000000 | 1 |
| 2010 | 1 | 423078 | 44.248190 | 423090 | 1 |
| 2011 | 0 | 1000001 | 97.407766 | 1000000 | 1 |

Named charge terms match the charged total on every finished seed.

Against stored au55 `h2h_metal` rows for the same seeds: hits 1 versus 6, Fisher two-sided p 0.06865.
Calls to hit, Mann-Whitney two-sided p 0.03398. Stored misses were recoded from the budget to the budget plus one.

`rows-bankrpc-only.csv` keeps twelve Au55 rows from a `bank-rpc`
build with `shape_keyed` false. That file is not a paper cell.
Its values are unchanged.

Intervals use 10000 resamples and generator seed 20261123. The hit-rate interval is drawn first, then the calls-to-hit median. Fisher and Mann-Whitney are computed in this harness without an external statistics package.
