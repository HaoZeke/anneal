Copyright (c) anneal developers

LJ55, seeds 2000-2011, maximum budget 1,000,000, host bench, one core.
The reference minimum is -279.24847000; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625, 3654866310baf4ef63853130608ab65392429ff6.
The ASE wall cap is 4.424556 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 1 | 72 | 1 | -279.24847046 | -0.00000046 | 0.823500 | 1.000000 | 1.000000 | 0.562777 | 0.260722 |
| basin hopping | 1/12 | 9.61483e-06 | not estimated | not estimated | -271.21690949 | 8.03156051 | 4.424591 | 5.372912 | 0.186119 | 1.738571 | 2.681977 |
| minima hopping | 0/12 | 7.39602e-07 | not estimated | not estimated | -266.05409411 | 13.19437588 | 4.488125 | 5.450064 | 0.183484 | 0.183773 | 4.307671 |

Basin hopping stopped at the wall cap on 11/12 seeds; those rows charged 75,325-257,144 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 18,558-24,077 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
