Copyright (c) anneal developers

LJ55, seeds 2000-2011, maximum budget 1,000,000, host bench, one core.
The reference minimum is -279.24847000; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625, b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac, 3654866310baf4ef63853130608ab65392429ff6.
The recorded wall caps are recommended 4.424556 s; basin hopping 4.424556 s; minima hopping 4.424556 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -279.248470 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 1 | 72 | 1 | -279.24847046 | -0.00000046 | 0.823500 | 1.000000 | 1.000000 | 0.562777 | 0.260722 |
| recommended | 12/12 | 1 | 9 | 0.000308045 | -279.24847046 | -0.00000046 | 0.127701 | 0.155072 | 6.448628 | 0.055039 | 0.074915 |
| basin hopping | 1/12 | 9.61483e-06 | not estimated | not estimated | -271.21690949 | 8.03156051 | 4.424591 | 5.372912 | 0.186119 | 1.738571 | 2.681977 |
| minima hopping | 0/12 | 7.39602e-07 | not estimated | not estimated | -266.05409411 | 13.19437588 | 4.488125 | 5.450064 | 0.183484 | 0.183773 | 4.307671 |

Basin hopping stopped at the wall cap on 11/12 seeds; those rows charged 75,325-257,144 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 18,558-24,077 calls.

Fisher and Mann-Whitney below compare recommended with each ASE arm.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 12/12 vs 1/12 | 9.61483e-06 | not estimated | not estimated | 0.028862 | 34.647913 |
| recommended against minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.028453 | 35.145437 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
