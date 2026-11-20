# SOAP-prior knockout, shard A

These rows record the SOAP-prior knockout at tree
`b12293b9d637dd32417be3aa482302e0f30dd594`. The revision column is a
tree identifier.

Arm A is `stock`; arm B is `H2H_KNOCKOUT=prior_soap`. Both arms use the
same release binary. Seeds 2000--2011 share one start per system and run
in ABBA order: stock first on even seeds and arm B first on odd seeds.
Every process is pinned to core 0 with all supported thread counts set to
one. Runs are serial and use the paper budgets.

Before measurement, the three knockout tests and the driver test passed.
For seed 2000 on every system, the develop and test-tree stock binaries
matched in recorded best-energy bits and charged calls. All 192 measured
runs completed; there were no crashes or retries. Binary and start-file
hashes and the step-0 records are in `session.csv`. Host information is
identified only by its machine identifier hash in `hosts.csv`.

`stats.csv` is computed with the paired routines in
`scripts/paired_stats.py`. The paired table below is ordered as both,
stock only, arm B only, neither. The interval is Newcombe method 10 for
the stock hit rate less the arm B hit rate.

| System | Stock hits | Arm B hits | Paired table | McNemar p | 95% interval |
| --- | ---: | ---: | --- | ---: | --- |
| lj38 | 12/12 | 11/12 | 11, 1, 0, 0 | 1 | [-0.168641353899, 0.353879911141] |
| lj55 | 12/12 | 12/12 | 12, 0, 0, 0 | 1 | [-0.242494006655, 0.242494006655] |
| lj75 | 5/12 | 5/12 | 1, 4, 4, 3 | 1 | [-0.404099118169, 0.404099118169] |
| au55 | 4/12 | 4/12 | 0, 4, 4, 4 | 1 | [-0.410117514757, 0.410117514757] |
| m55r6 | 12/12 | 12/12 | 12, 0, 0, 0 | 1 | [-0.242494006655, 0.242494006655] |
| m55r14 | 9/12 | 7/12 | 6, 3, 1, 2 | 0.625 | [-0.175819727742, 0.462045241802] |
| m80r6 | 12/12 | 12/12 | 12, 0, 0, 0 | 1 | [-0.242494006655, 0.242494006655] |
| m80r14 | 7/12 | 6/12 | 3, 4, 3, 2 | 1 | [-0.30677603218, 0.442679374839] |
