Copyright (c) anneal developers

LJ150, seeds 2000-2011, maximum budget 4,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -893.31025800; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 5ac25c49007c7f93848e588f0e1325eba9e31e44d0cb51a3ee63154467bbe48f.
The recorded wall caps are recommended 667.242957 s; basin hopping 667.242957 s; minima hopping 667.242957 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -893.310258 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | -884.65296483 | 8.65729317 | 208.091856 | 1.000000 | 1.000000 | 154.073584 | 53.702757 |
| recommended | 8/12 | — | -893.31025788 | 0.00000012 | 59.386426 | — | — | 37.834226 | 21.552200 |
| basin hopping | 0/12 | 1 | -868.96761602 | 24.34264198 | 216.066392 | 1.038322 | 0.963092 | 169.079825 | 47.233752 |
| minima hopping | 0/12 | — | -863.40304193 | 29.90721607 | 712.521453 | — | — | 2.716632 | 709.665526 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 43,061-98,927 calls.

Ratios and tests are printed only for same-stop pairs.

| comparison | hits | Fisher p |
| --- | --- | --- |
| recommended against basin hopping | 8/12 vs 0/12 | — |
| recommended against minima hopping | 8/12 vs 0/12 | — |

The Fisher cells above are blank where the arms do not share a stop.
Medians are type 7. Each wall is that arm's own median. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
