Copyright (c) anneal developers

LJ38, seeds 2000-2011, maximum budget 400,000, host bench, one core.
The reference minimum is -173.92842700; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 5a7f629d79be88b9d3f0cb2c6665062bda4ac625, b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
The recorded wall caps are recommended 4.653972 s; basin hopping 4.653972 s; minima hopping 4.653972 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -173.928427 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10/12 | 1 | -173.92842659 | 0.00000041 | 0.981375 | 1.000000 | 1.000000 | 0.574202 | 0.407172 |
| recommended | 12/12 | 0.478261 | -173.92842659 | 0.00000041 | 0.066933 | 0.068203 | 14.662152 | 0.026147 | 0.042388 |
| basin hopping | 0/12 | 6.73038e-05 | -171.10081694 | 2.82761006 | 4.654009 | 4.742337 | 0.210866 | 1.319512 | 3.334485 |
| minima hopping | 0/12 | 6.73038e-05 | -170.08494879 | 3.84347821 | 4.864397 | 4.956718 | 0.201746 | 0.138811 | 4.728948 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 319,156-346,870 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 19,112-30,792 calls.

Fisher below compares recommended with each ASE arm.

| comparison | hits | Fisher p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- |
| recommended against basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.014382 | 69.532873 |
| recommended against minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.013760 | 72.676159 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
