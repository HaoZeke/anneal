Copyright (c) anneal developers

Au55, seeds 2000-2011, maximum budget 1,000,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -191.03085280; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 0a08925987e7463fa8f02067ad2881a9.
The recorded wall caps are recommended 107.929842 s; basin hopping 107.929842 s; minima hopping 107.929842 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -191.03085280 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | -190.81083051 | 0.22002228 | 33.279924 | 1.000000 | 1.000000 | 27.704543 | 5.575382 |
| recommended | 0/12 | 1 | -190.98001732 | 0.05083548 | 36.000117 | 1.081737 | 0.924439 | 28.228509 | 7.807164 |
| basin hopping | 0/12 | — | -189.79900994 | 1.23184286 | 107.930215 | — | — | 15.963161 | 91.967020 |
| minima hopping | 0/12 | — | -190.29395447 | 0.73689833 | 110.328913 | — | — | 2.350913 | 107.935484 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 522,329-590,386 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 72,762-86,949 calls.

Ratios and tests are printed only for same-stop pairs.

| comparison | hits | Fisher p |
| --- | --- | --- |
| recommended against basin hopping | 0/12 vs 0/12 | — |
| recommended against minima hopping | 0/12 vs 0/12 | — |

The Fisher cells above are blank where the arms do not share a stop.
Medians are type 7. Each wall is that arm's own median. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.

## Historical rows

The earlier `rows.csv` and `rows-for-cluster.csv` tables are retained;
the anneal, basin-hopping and minima-hopping rows of `rows.csv` at 33bed75f are kept
and superseded by `rows-for-cluster.csv`; `superseded.csv` lists each of them.
