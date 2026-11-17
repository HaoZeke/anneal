Copyright (c) anneal developers

LJ150, seeds 2000-2011, maximum budget 4,000,000, host bench, one core.
The reference minimum is -893.31025800; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The ASE wall cap is 544.418154 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -884.65296483 | 8.65729317 | 177.839157 | 1.000000 | 1.000000 | 132.630578 | 45.213123 |
| basin hopping | 0/12 | 1 | 72 | 1 | -868.96761602 | 24.34264198 | 213.520749 | 1.200640 | 0.832889 | 167.078759 | 46.468266 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -862.56043539 | 30.74982261 | 571.325791 | 3.212598 | 0.311275 | 2.452872 | 567.757919 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 41,939-97,217 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
