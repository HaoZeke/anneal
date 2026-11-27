Copyright (c) anneal developers

LJ38, seeds 2000-2011, maximum budget 400,000, host 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873, one core.
The reference minimum is -173.92842700; a hit is at most 0.001 above it.
Anneal uses the cluster preset. Recommended uses Config::recommended.
ASE basin hopping and minima hopping use
the ASE 3.29.0 constructor defaults.
Rows are in `rows-61fa67cf.csv` and carry code_revision 61fa67cf380e46876b569da45e7cc091647a775b.
Rows carry harness_blob 7d084438b8bf626895204067126d06e96a994db9.
Rows carry ase_version 3.29.0.
Rows carry session 0a08925987e7463fa8f02067ad2881a9.
The recorded wall caps are recommended 4.644507 s; basin hopping 4.644507 s; minima hopping 4.644507 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p vs anneal | median best | gap to -173.928427 | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10/12 | 1 | -173.92842659 | 0.00000041 | 0.982206 | 1.000000 | 1.000000 | 0.582390 | 0.399817 |
| recommended | 12/12 | 0.478261 | -173.92842659 | 0.00000041 | 0.605387 | 0.616354 | 1.622444 | 0.283605 | 0.321782 |
| basin hopping | 0/12 | — | -171.10081694 | 2.82761006 | 4.644532 | 4.728672 | 0.211476 | 1.313719 | 3.330812 |
| minima hopping | 0/12 | — | -170.08494879 | 3.84347821 | 4.858002 | 4.946008 | 0.202183 | 0.136979 | 4.727197 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 315,352-351,628 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 21,363-31,032 calls.

Basin hopping and minima hopping stopped at the wall on every seed.
Recommended did not. The stops differ, and there is no paired test.

| comparison | hits | Fisher p | median wall ratio | inverse ratio |
| --- | --- | --- | --- | --- |
| recommended against basin hopping | 12/12 vs 0/12 | — | 0.130344 | 7.672005 |
| recommended against minima hopping | 12/12 vs 0/12 | — | 0.124616 | 8.024621 |

The Fisher cells above are blank where one arm stopped at the wall on every seed.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
