Copyright (c) anneal developers

Au55, seeds 2000-2011, maximum budget 1,000,000, host bench, one core.
The reference minimum is -191.03085280; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision fc1af066a915a80eb46a3bc478caaf155cc9b29f, b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac.
The recorded wall caps are recommended 96.650229 s; basin hopping 112.374012 s; minima hopping 112.374012 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -191.03085280 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -190.81083051 | 0.22002228 | 34.628695 | 1.000000 | 1.000000 | 28.582308 | 6.172918 |
| recommended | 0/12 | 1 | 72 | 1 | -190.98331173 | 0.04754107 | 33.663735 | 0.972134 | 1.028665 | 26.379868 | 7.288144 |
| basin hopping | 0/12 | 1 | not estimated | not estimated | -189.82092796 | 1.20992484 | 112.374357 | 3.245123 | 0.308155 | 16.719311 | 95.655017 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -190.29395447 | 0.73689833 | 115.191930 | 3.326488 | 0.300617 | 2.475493 | 112.830887 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 496,722-609,996 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 72,762-88,823 calls.

Fisher and Mann-Whitney below compare recommended with each ASE arm.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.299568 | 3.338143 |
| recommended against minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.292240 | 3.421840 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
