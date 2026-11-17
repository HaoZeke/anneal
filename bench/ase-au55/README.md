Copyright (c) anneal developers

Au55, seeds 2000-2011, maximum budget 1,000,000, host bench, one core.
The reference minimum is -191.03085280; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 61b5c9a01b2002223f9c1e7024863b112001ebf7.
The anneal, basin-hopping and minima-hopping rows of `rows.csv` at 33bed75f are kept
and superseded by `rows-for-cluster.csv`; `superseded.csv` lists each of them.
This report reads the recommended rows of `rows.csv` and every row of `rows-for-cluster.csv`.
The recorded wall caps are recommended 122.695098 s; basin hopping 122.695098 s; minima hopping 122.695098 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | gap to -191.03085280 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -190.81083051 | 0.22002228 | 36.060303 | 1.000000 | 1.000000 | 29.540093 | 6.382390 |
| recommended | 0/12 | 1 | 72 | 1 | -190.98001732 | 0.05083548 | 36.069230 | 1.000248 | 0.999753 | 28.296631 | 7.801004 |
| basin hopping | 0/12 | 1 | not estimated | not estimated | -189.83438389 | 1.19646891 | 122.696096 | 3.402525 | 0.293899 | 18.332622 | 104.363608 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -190.29395447 | 0.73689833 | 126.761768 | 3.515272 | 0.284473 | 2.486090 | 124.332869 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 629,479-665,149 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 80,811-92,960 calls.

Fisher and Mann-Whitney below compare recommended with each ASE arm.

| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recommended against basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.293972 | 3.401683 |
| recommended against minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.284543 | 3.514402 |

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
