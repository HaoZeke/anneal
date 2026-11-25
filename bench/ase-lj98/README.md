Copyright (c) anneal developers

LJ98, seeds 2000-2011, maximum budget 4,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -543.66536100; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 8e3c2b1b1df44c8389476848df82bee0.
The recorded wall caps are recommended 288.988608 s; basin hopping 288.988608 s; minima hopping 288.988608 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -543.665361 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 4/12 | 1 | 72 | 1 | -543.54672557 | 0.11863543 | 84.630861 | 1.000000 | 1.000000 | 59.056659 | 25.579015 |
| recommended | 7/12 | 0.413649 | 48 | 0.139186 | -543.66536077 | 0.00000023 | 78.918537 | 0.932503 | 1.072383 | 47.071047 | 31.134047 |
| basin hopping | 0/12 | 0.0931677 | 96 | 0.0366874 | -536.02030046 | 7.64506053 | 123.012927 | 1.453523 | 0.687983 | 77.440966 | 45.571962 |
| minima hopping | 0/12 | 0.0931677 | not estimated | not estimated | -532.04115348 | 11.62420752 | 299.160840 | 3.534891 | 0.282894 | 1.395366 | 297.686307 |

Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 56,362-80,878 calls.

Fisher and Mann-Whitney below compare recommended with each ASE arm.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 7/12 vs 0/12 | 0.00457666 | 30 | 0.00285559 | 0.641547 | 1.558733 |
| recommended against minima hopping | 7/12 vs 0/12 | 0.00457666 | not estimated | not estimated | 0.263800 | 3.790755 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
