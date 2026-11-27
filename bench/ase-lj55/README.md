Copyright (c) anneal developers

LJ55, seeds 2000-2011, maximum budget 1,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -279.24847000; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 0a08925987e7463fa8f02067ad2881a9.
The recorded wall caps are recommended 4.220427 s; basin hopping 4.220427 s; minima hopping 4.220427 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -279.248470 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 1 | -279.24847046 | -0.00000046 | 0.787921 | 1.000000 | 1.000000 | 0.546228 | 0.241693 |
| recommended | 12/12 | 1 | -279.24847046 | -0.00000046 | 0.051215 | 0.065000 | 15.384565 | 0.025901 | 0.027197 |
| basin hopping | 1/12 | — | -271.21690949 | 8.03156051 | 4.220457 | 5.356450 | 0.186691 | 1.652929 | 2.564113 |
| minima hopping | 0/12 | — | -266.05409411 | 13.19437588 | 4.442467 | 5.638218 | 0.177361 | 0.178083 | 4.261282 |

Basin hopping stopped at the wall cap on 11/12 seeds; those rows charged 200,184-245,800 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 18,558-24,077 calls.

The Fisher cells are blank where the stops differ, and there is no paired test.
Wall ratios are omitted where the arms do not share a stop.

| comparison | hits | Fisher p |
| --- | --- | --- |
| recommended against basin hopping | 12/12 vs 1/12 | — |
| recommended against minima hopping | 12/12 vs 0/12 | — |

The Fisher cells above are blank where the arms do not share a stop.
Medians are type 7. Each wall is that arm's own median. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
