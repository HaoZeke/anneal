Copyright (c) anneal developers

LJ150, seeds 2000-2011, maximum budget 4,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -893.31025800; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Each row also records harness_blob 7d084438b8bf626895204067126d06e96a994db9, ase_version 3.29.0, and session 5ac25c49007c7f93848e588f0e1325eba9e31e44d0cb51a3ee63154467bbe48f.
The recorded wall caps are recommended 667.242957 s; basin hopping 667.242957 s; minima hopping 667.242957 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -893.310258 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -884.65296483 | 8.65729317 | 208.091856 | 1.000000 | 1.000000 | 154.073584 | 53.702757 |
| recommended | 8/12 | 0.00134608 | 24 | 0.00108433 | -893.31025788 | 0.00000012 | 59.386426 | 0.285386 | 3.504031 | 37.834226 | 21.552200 |
| basin hopping | 0/12 | 1 | 72 | 1 | -868.96761602 | 24.34264198 | 216.066392 | 1.038322 | 0.963092 | 169.079825 | 47.233752 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -863.40304193 | 29.90721607 | 712.521453 | 3.424072 | 0.292050 | 2.716632 | 709.665526 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 43,061-98,927 calls.

Fisher and Mann-Whitney below compare recommended with each ASE arm.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 8/12 vs 0/12 | 0.00134608 | 24 | 0.00108433 | 0.274853 | 3.638313 |
| recommended against minima hopping | 8/12 vs 0/12 | 0.00134608 | not estimated | not estimated | 0.083347 | 11.998052 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
