# Anneal against GMIN, n=24

The committed n=12 rows in bench/gmin-panel use seeds 2000-2011 on every
cell. The declared seeds are 2000-2023, that original twelve plus
2012-2023.

h2h_gmin_proof at BASE contains the lj103 entry (103 atoms, reference
-575.766131, budget 4000000). The lj103 table patch was not applied.
Both arms have stored rows.

Cells are the GMIN main panel. A hit is a quenched energy at or below
the reference plus 0.001. A miss is stored at the budget. b counts seeds
where anneal hits and GMIN misses. c counts the opposite. The difference
is the anneal hit proportion minus the GMIN hit proportion. McNemar is
the exact two-sided binomial test on the discordant pairs. The interval
is Newcombe's paired method 10 at 95 percent. Wilcoxon is the two-sided
signed-rank test of anneal wall minus GMIN wall: exact when every
nonzero absolute difference is unique, and the normal approximation with
the tie and continuity corrections otherwise. Zero wall differences are
dropped before ranking. Median calls use Hyndman-Fan type 7 with a miss
ranked after every hit. Median wall diff is the type 7 median of the
paired wall differences.

Declared cells and paper budgets:

- lj38: 400000
- lj55: 1000000
- lj75: 4000000
- lj98: 4000000
- lj102: 4000000
- lj103: 4000000
- lj104: 4000000
- lj150: 4000000
- au55: 1000000
- m55r6: 1000000
- m55r14: 1000000
- m80r6: 4000000
- m80r14: 4000000

Declared seeds: 2000-2023 (24).

Stored successful (slot, arm) counts: slot 0 anneal 156, slot 0 gmin
156, slot 1 anneal 156, slot 1 gmin 156.

Rows with an error: 0.

| Cell | Budget | Anneal hits | GMIN hits | b | c | McNemar p | Diff | Newcombe low | Newcombe high | Wilcoxon p | Wilcoxon | Zeros | Median wall diff | Anneal median calls | GMIN median calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 400000 | 23/24 | 15/24 | 9 | 1 | 0.021484375 | 0.333333 | 0.086295 | 0.539562 | 1.04904174805e-05 | exact | 0 | -2.018482 | 117832 | 340051 |
| lj55 | 1000000 | 24/24 | 24/24 | 0 | 0 | 1 | 0.000000 | -0.137976 | 0.137976 | 0.000175476074219 | exact | 0 | -0.105031 | 6376 | 15764 |
| lj75 | 4000000 | 7/24 | 0/24 | 7 | 0 | 0.015625 | 0.291667 | 0.093300 | 0.491677 | 1.19209289551e-07 | exact | 0 | -53.833464 | 4000000 | 4000000 |
| lj98 | 4000000 | 16/24 | 1/24 | 15 | 0 | 6.103515625e-05 | 0.625000 | 0.368714 | 0.782388 | 1.19209289551e-07 | exact | 0 | -107.300948 | 2595033 | 4000000 |
| lj102 | 4000000 | 11/24 | 10/24 | 7 | 6 | 1 | 0.041667 | -0.236064 | 0.310978 | 0.00433504581451 | exact | 0 | -78.483502 | 4000000 | 4000000 |
| lj103 | 4000000 | 13/24 | 7/24 | 9 | 3 | 0.14599609375 | 0.250000 | -0.026503 | 0.479137 | 3.65972518921e-05 | exact | 0 | -82.505483 | 3685736.5 | 4000000 |
| lj104 | 4000000 | 11/24 | 13/24 | 3 | 5 | 0.7265625 | -0.083333 | -0.301793 | 0.149126 | 0.0248636007309 | exact | 0 | -80.380799 | 4000000 | 3550516 |
| lj150 | 4000000 | 18/24 | 15/24 | 5 | 2 | 0.453125 | 0.125000 | -0.098925 | 0.333043 | 0.000372767448425 | exact | 0 | -167.213254 | 286952.5 | 3129518.5 |
| au55 | 1000000 | 8/24 | 1/24 | 8 | 1 | 0.0390625 | 0.291667 | 0.053509 | 0.499112 | 2.4676322937e-05 | exact | 0 | -23.900193 | 1000000 | 1000000 |
| m55r6 | 1000000 | 24/24 | 24/24 | 0 | 0 | 1 | 0.000000 | -0.137976 | 0.137976 | 0.944107174873 | exact | 0 | -0.050977 | 21373.5 | 25038.5 |
| m55r14 | 1000000 | 18/24 | 0/24 | 18 | 0 | 7.62939453125e-06 | 0.750000 | 0.507851 | 0.880006 | 1.19209289551e-07 | exact | 0 | -9.423762 | 498185 | 1000000 |
| m80r6 | 4000000 | 24/24 | 24/24 | 0 | 0 | 1 | 0.000000 | -0.137976 | 0.137976 | 0.855335712433 | exact | 0 | -2.742198 | 72907 | 232300 |
| m80r14 | 4000000 | 15/24 | 0/24 | 15 | 0 | 6.103515625e-05 | 0.625000 | 0.383749 | 0.788406 | 3.65972518921e-05 | exact | 0 | -17.223463 | 3517990.5 | 4000000 |

Code revision 61fa67cf380e46876b569da45e7cc091647a775b. GMIN svn
revision 9cbec1a446352bc9ef064bd2b1a5767194655919. Every code revision
and harness blob id resolves as a git object.
