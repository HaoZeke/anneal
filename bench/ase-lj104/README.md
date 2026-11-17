Copyright (c) anneal developers

LJ104, seeds 2000-2011, maximum budget 4,000,000, host bench, one core.
The reference minimum is -582.08664200; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The ASE wall cap is 280.907064 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -580.80432897 | 1.28231303 | 88.960797 | 1.000000 | 1.000000 | 63.958464 | 25.016078 |
| basin hopping | 0/12 | 1 | 72 | 1 | -575.36385980 | 6.72278220 | 127.172397 | 1.429533 | 0.699529 | 83.116758 | 43.981369 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -570.25110190 | 11.83554010 | 287.850950 | 3.235706 | 0.309052 | 1.361099 | 286.357127 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 40,554-71,659 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
