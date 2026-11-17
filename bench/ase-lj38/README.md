Copyright (c) anneal developers

LJ38, seeds 2000-2011, maximum budget 400,000, host bench, one core.
The reference minimum is -173.92842700; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625.
The ASE wall cap is 4.653972 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10/12 | 1 | 72 | 1 | -173.92842659 | 0.00000041 | 0.981375 | 1.000000 | 1.000000 | 0.574202 | 0.407172 |
| basin hopping | 0/12 | 6.73038e-05 | not estimated | not estimated | -171.10081694 | 2.82761006 | 4.654009 | 4.742337 | 0.210866 | 1.319512 | 3.334485 |
| minima hopping | 0/12 | 6.73038e-05 | not estimated | not estimated | -170.08494879 | 3.84347821 | 4.864397 | 4.956718 | 0.201746 | 0.138811 | 4.728948 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 319,156-346,870 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 19,112-30,792 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
