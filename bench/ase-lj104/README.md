Copyright (c) anneal developers

LJ104, seeds 2000-2011, maximum budget 4,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -582.08664200; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 8e3c2b1b1df44c8389476848df82bee0.
The recorded wall caps are recommended 290.981259 s; basin hopping 290.981259 s; minima hopping 290.981259 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -582.086642 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | -580.80432897 | 1.28231303 | 91.371704 | 1.000000 | 1.000000 | 65.059184 | 25.949275 |
| recommended | 5/12 | — | -582.03842906 | 0.04821294 | 101.740789 | 1.113482 | 0.898083 | 63.784653 | 37.765714 |
| basin hopping | 0/12 | 1 | -575.36385980 | 6.72278220 | 128.484760 | 1.406177 | 0.711148 | 83.898952 | 44.635166 |
| minima hopping | 0/12 | — | -570.33284099 | 11.75380101 | 305.203218 | 3.340238 | 0.299380 | 1.384291 | 303.822900 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 40,554-75,123 calls.

The Fisher cells are blank where the stops differ, and there is no paired test.
Wall ratios are omitted where the arms do not share a stop.

| comparison | hits | Fisher p |
| --- | --- | --- |
| recommended against basin hopping | 5/12 vs 0/12 | — |
| recommended against minima hopping | 5/12 vs 0/12 | — |

The Fisher cells above are blank where the arms do not share a stop.
Medians are type 7. Each wall is that arm's own median. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
