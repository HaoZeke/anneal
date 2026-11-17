Copyright (c) anneal developers

LJ98, seeds 2000-2011, maximum budget 4,000,000, host bench, one core.
The reference minimum is -543.66536100; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The ASE wall cap is 246.546693 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 4/12 | 1 | 72 | 1 | -543.54672557 | 0.11863543 | 80.934059 | 1.000000 | 1.000000 | 57.146890 | 23.659054 |
| basin hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -536.02030046 | 7.64506053 | 118.307105 | 1.461772 | 0.684101 | 74.478849 | 43.673459 |
| minima hopping | 0/12 | 0.0931677 | not estimated | not estimated | -532.04115348 | 11.62420752 | 254.781554 | 3.148014 | 0.317661 | 1.270590 | 253.521140 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 55,033-80,878 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
