# h2o20-fix5 gate

Written by `verdict.py` from the rows; do not edit by hand.

- Part 1: anneal 1/12 against CREST 0/12 hits, Fisher p 1, calls Mann-Whitney p 0.3593.
- Part 2: bit-identical in all 33 cells on every seed.
- Hashed machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

Protocol: release builds, `taskset -c 0`, every thread variable at 1, one run at a time, ABBA arm order by seed, no warm-up and no shortened run. Medians are Hyndman-Fan type 7. Fisher is exact two-sided. Mann-Whitney is two-sided with tie and continuity correction.

## Part 1: GFN2 (H2O)20 at TIP, anneal against CREST

Rows: `bench/h2o20-crest/rows-b04621fd81776cfb78626162c5b3e746acb9859f.csv`, 24 of 24. Seeds 2000-2011, budget 28000 charged calls, the harness's `run_anneal` (default config: TIP4P surrogate stage, then the flexible GFN2 search, no capped Lindh) and `run_crest`, its vtight rescore. Even seeds ran anneal then CREST, odd seeds CREST then anneal, in one session.

A hit is a finite rescore at or below -101.749249454294 Eh plus 1e-06 Eh, the reference and rule of the 131b411a panel. The looser 0.001 Eh rule is shown beside it. Calls to a hit rank misses worst. There is no A/A arm in Part 1, so the A/A ratio is n/a.

| Arm | n | Hits 1e-6 | Hits 1e-3 | Fisher p | Calls MW p | Wall median s | Inside median s | Outside median s | Outside/charged median s | anneal/CREST | A/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12 | 1/12 | 1/12 | 1 | 0.3593 | 1213.438 | 1210.522 | 2.756 | 9.842920e-05 | 0.04346 | n/a |
| crest | 12 | 0/12 | 0/12 |  |  | 848.070 | 787.223 | 63.411 | 2.264671e-03 |  |  |

Fisher p under the 1e-3 rule: 1.

On the rescored energies, the harness's statistics give Mann-Whitney p 0.0001233457544, paired Wilcoxon W+ 0.0, W- 78.0, exact p 0.00048828125, rank-biserial -1, and median paired gap, anneal minus CREST, -0.08005476114 Eh.

Per row:

| Seed | Slot | Arm | Rescored Eh | Hit | Calls | Charged | Wall s | Inside s | Outside s | Outside/charged s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 0 | anneal | -101.739726988707 | 0 | 28000 | 28000 | 1279.284623 | 1277.677183 | 1.607440 | 5.740857e-05 | ok |
| 2000 | 1 | crest | -101.442399225283 | 0 | 28000 | 28000 | 539.763938 | 475.329270 | 64.434668 | 2.301238e-03 | stopped |
| 2001 | 0 | crest | -101.708943681209 | 0 | 28000 | 28000 | 1079.731648 | 1013.215416 | 66.516232 | 2.375580e-03 | stopped |
| 2001 | 1 | anneal | -101.742910832600 | 0 | 28000 | 28000 | 1242.359836 | 1239.154616 | 3.205220 | 1.144721e-04 | ok |
| 2002 | 0 | anneal | -101.739442912257 | 0 | 28000 | 28000 | 1235.855312 | 1233.103655 | 2.751657 | 9.827346e-05 | ok |
| 2002 | 1 | crest | -101.609073698204 | 0 | 28000 | 28000 | 748.936311 | 689.226268 | 59.710043 | 2.132502e-03 | stopped |
| 2003 | 0 | crest | -101.731882672425 | 0 | 28000 | 28000 | 1324.937082 | 1244.455172 | 80.481910 | 2.874354e-03 | stopped |
| 2003 | 1 | anneal | -101.743953867921 | 0 | 28000 | 28000 | 1223.184860 | 1219.902844 | 3.282017 | 1.172149e-04 | ok |
| 2004 | 0 | anneal | -101.740723547092 | 0 | 28000 | 28000 | 1211.988420 | 1208.908897 | 3.079522 | 1.099829e-04 | ok |
| 2004 | 1 | crest | -101.685696713345 | 0 | 28000 | 28000 | 905.185447 | 842.798533 | 62.386914 | 2.228104e-03 | stopped |
| 2005 | 0 | crest | -101.636627906562 | 0 | 28000 | 28000 | 768.593340 | 709.254827 | 59.338513 | 2.119233e-03 | stopped |
| 2005 | 1 | anneal | -101.749296717283 | 1 | 23965 | 28000 | 1180.121796 | 1177.651431 | 2.470365 | 8.822732e-05 | ok |
| 2006 | 0 | anneal | -101.743229701827 | 0 | 28000 | 28000 | 1214.887059 | 1212.135019 | 2.752040 | 9.828714e-05 | ok |
| 2006 | 1 | crest | -101.587477873067 | 0 | 28000 | 28000 | 743.286779 | 683.790067 | 59.496712 | 2.124883e-03 | stopped |
| 2007 | 0 | crest | -101.602974783781 | 0 | 28000 | 28000 | 775.779426 | 714.013668 | 61.765758 | 2.205920e-03 | stopped |
| 2007 | 1 | anneal | -101.740161486992 | 0 | 28000 | 28000 | 1239.175307 | 1236.376184 | 2.799123 | 9.996868e-05 | ok |
| 2008 | 0 | anneal | -101.745193952705 | 0 | 28000 | 28000 | 1197.021711 | 1194.261715 | 2.759995 | 9.857125e-05 | ok |
| 2008 | 1 | crest | -101.741849730260 | 0 | 28000 | 28000 | 1314.413072 | 1235.335841 | 79.077231 | 2.824187e-03 | stopped |
| 2009 | 0 | crest | -101.631464980667 | 0 | 28000 | 28000 | 790.955268 | 731.647730 | 59.307538 | 2.118126e-03 | stopped |
| 2009 | 1 | anneal | -101.736547669198 | 0 | 28000 | 28000 | 1160.271404 | 1157.691978 | 2.579427 | 9.212239e-05 | ok |
| 2010 | 0 | anneal | -101.741983271533 | 0 | 28000 | 28000 | 1174.440238 | 1171.528035 | 2.912203 | 1.040072e-04 | ok |
| 2010 | 1 | crest | -101.712634861168 | 0 | 28000 | 28000 | 1005.235667 | 940.026202 | 65.209465 | 2.328909e-03 | stopped |
| 2011 | 0 | crest | -101.736269335117 | 0 | 28000 | 28000 | 1288.217422 | 1211.315438 | 76.901984 | 2.746499e-03 | stopped |
| 2011 | 1 | anneal | -101.742250684333 | 0 | 28000 | 28000 | 1175.639378 | 1173.557483 | 2.081896 | 7.435343e-05 | ok |

### Against the 131b411a rows

The 131b411a rows in `bench/h2o20-crest/seeds.csv` are unchanged. Their `anneal` arm was the example with the capped Lindh quench always on and auto TRIC chosen by default; TIP's default arm has both off. Those rows record no charged count, so charged identity is not available against them. Best identity is the rescored string.

| Seed | anneal 131b411a | anneal TIP | Delta Eh | Same | CREST 131b411a | CREST now | Delta Eh | Same | capped Lindh 131b411a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | -101.744993318865 | -101.739726988707 | +0.005266330158 | no | -101.735805296063 | -101.442399225283 | +0.293406070780 | no | -101.745068656209 |
| 2001 | -101.748641921094 | -101.742910832600 | +0.005731088494 | no | -101.731662287000 | -101.708943681209 | +0.022718605791 | no | -101.747605472117 |
| 2002 | -101.742146268260 | -101.739442912257 | +0.002703356003 | no | -101.731701032688 | -101.609073698204 | +0.122627334484 | no | -101.748912747494 |
| 2003 | -101.746457867233 | -101.743953867921 | +0.002503999312 | no | -101.736906986673 | -101.731882672425 | +0.005024314248 | no | -101.742435962771 |
| 2004 | -101.747493459097 | -101.740723547092 | +0.006769912005 | no | -101.734191463941 | -101.685696713345 | +0.048494750596 | no | -101.746472072898 |
| 2005 | -101.742424249406 | -101.749296717283 | -0.006872467877 | no | -101.734263453673 | -101.636627906562 | +0.097635547111 | no | -101.744091233981 |
| 2006 | -101.747779192545 | -101.743229701827 | +0.004549490718 | no | -101.738019426743 | -101.587477873067 | +0.150541553676 | no | -101.747526747916 |
| 2007 | -101.747349140102 | -101.740161486992 | +0.007187653110 | no | -101.735556263601 | -101.602974783781 | +0.132581479820 | no | -101.743331883573 |
| 2008 | -101.743363851546 | -101.745193952705 | -0.001830101159 | no | -101.736010145596 | -101.741849730260 | -0.005839584664 | no | -101.749249454294 |
| 2009 | -101.747137486582 | -101.736547669198 | +0.010589817384 | no | -101.731784956688 | -101.631464980667 | +0.100319976021 | no | -101.746581772118 |
| 2010 | -101.746482893227 | -101.741983271533 | +0.004499621694 | no | -101.733608309927 | -101.712634861168 | +0.020973448759 | no | -101.744240010727 |
| 2011 | -101.747959455526 | -101.742250684333 | +0.005708771193 | no | -101.735497358585 | -101.736269335117 | -0.000771976532 | no | -101.743643341097 |

| Arm | Hits 131b411a | Hits now | Median Eh 131b411a | Median Eh now | Wall 131b411a | Wall now | Inside 131b411a | Inside now | Outside 131b411a | Outside now |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1/12 | -101.74681019 | -101.742116978 | 1743.167 | 1213.438 | 1251.828 | 1210.522 | 489.956 | 2.756 |
| crest | 0/12 | 0/12 | -101.734880406 | -101.66116231 | 1341.468 | 848.070 | 1263.009 | 787.223 | 78.591 | 63.411 |

## Part 2: non-water bit identity, DEVELOP against TIP

Rows: `bench/h2o20-fix5-gate/panel-rows.csv`, 297 of 297. A is DEVELOP `61fa67cf380e46876b569da45e7cc091647a775b`, B is TIP `bace7cfc229a1f9297c74b738af2b7ebfe059295`, A2 a second fresh release build of DEVELOP. Seeds 2000-2002, the drivers' paper budgets, no H2H_* variable. Even seeds ran A, B, A2 and odd seeds A2, B, A. A hit is a finite best at or below the reference plus 1e-3. Fisher and Mann-Whitney compare A with B. The cost columns are medians of outside_s/charged; B/A sits next to A2/A. Identity compares best_bits, calls and charged per seed.

MW is read as the Gupta metal cells of `h2h_metal`; the panel is non-water and no driver in either tree has an mW water cell.

| Family | Cell | Budget | Hits A | Hits B | Hits A2 | Fisher p | Calls MW p | Outside/charged A / B s | B/A | A2/A | Identical per seed (A vs B; A vs A2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj | lj38 | 400000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 2.4207e-06 / 2.4554e-06 | 1.014 | 0.9975 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| lj | lj55 | 1000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 5.5832e-06 / 5.5749e-06 | 0.9985 | 0.989 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| lj | lj75 | 4000000 | 1/3 | 1/3 | 1/3 | 1 | 1 | 7.1063e-06 / 7.3254e-06 | 1.031 | 0.9986 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| lj | lj98 | 4000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 9.9736e-06 / 1.0038e-05 | 1.006 | 1.013 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj105n38 | 400000 | 1/3 | 1/3 | 1/3 | 1 | 1 | 9.3978e-06 / 9.3403e-06 | 0.9939 | 0.9977 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj115n38 | 400000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 7.9443e-06 / 7.8850e-06 | 0.9925 | 1.002 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj130n38 | 400000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 1.8823e-05 / 1.8797e-05 | 0.9986 | 1.011 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj105n55 | 1000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 3.3166e-06 / 3.2823e-06 | 0.9897 | 0.9866 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj115n55 | 1000000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 3.1141e-05 / 3.2484e-05 | 1.043 | 1.017 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj130n55 | 1000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 2.3512e-05 / 2.3316e-05 | 0.9917 | 1.034 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| blj | blj130n98 | 4000000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 1.3643e-05 / 1.3631e-05 | 0.9992 | 1.064 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| morse | m55r6 | 1000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 4.9425e-06 / 4.9229e-06 | 0.996 | 0.9992 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| morse | m55r14 | 1000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 3.8999e-06 / 3.9118e-06 | 1.003 | 1.003 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| morse | m80r6 | 4000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 1.1381e-05 / 1.1362e-05 | 0.9983 | 1.011 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| morse | m80r14 | 4000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 9.3178e-06 / 8.7563e-06 | 0.9397 | 0.9402 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc126-38 | 400000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 2.2407e-06 / 2.2158e-06 | 0.9889 | 1.001 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc108-38 | 400000 | 1/3 | 1/3 | 1/3 | 1 | 1 | 2.7583e-06 / 2.7173e-06 | 0.9851 | 0.9594 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc96-38 | 400000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 2.4651e-06 / 2.4697e-06 | 1.002 | 1.022 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc126-55 | 1000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 4.3087e-06 / 4.2516e-06 | 0.9867 | 0.995 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc108-55 | 1000000 | 1/3 | 1/3 | 1/3 | 1 | 1 | 4.8137e-06 / 4.7877e-06 | 0.9946 | 1.012 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc96-55 | 1000000 | 3/3 | 3/3 | 3/3 | 1 | 1 | 3.6243e-06 / 3.5805e-06 | 0.9879 | 1.004 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc126-80 | 1000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 8.1223e-06 / 7.9881e-06 | 0.9835 | 0.9959 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc108-80 | 1000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 9.1922e-06 / 9.0965e-06 | 0.9896 | 0.9964 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| sc | sc96-80 | 1000000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 1.0976e-05 / 1.0854e-05 | 0.9889 | 0.9993 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | zn38 | 50000 | 1/3 | 1/3 | 1/3 | 1 | 1 | 2.8283e-06 / 2.8358e-06 | 1.003 | 1.007 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | zn55 | 80000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 4.2034e-06 / 4.2146e-06 | 1.003 | 0.9993 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | zn75 | 100000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 6.2320e-06 / 6.2479e-06 | 1.003 | 1 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | zn80 | 100000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 6.7748e-06 / 6.8131e-06 | 1.006 | 0.9998 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | cu19au19 | 80000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 2.4700e-06 / 2.4905e-06 | 1.008 | 1.008 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | au38 | 400000 | 2/3 | 2/3 | 2/3 | 1 | 1 | 4.2443e-06 / 4.2303e-06 | 0.9967 | 0.9944 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | cu6au32 | 400000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 4.6132e-06 / 4.6681e-06 | 1.012 | 0.9988 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | au55 | 1000000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 9.1434e-06 / 9.2076e-06 | 1.007 | 0.9989 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |
| mw | cu27au28 | 1000000 | 0/3 | 0/3 | 0/3 | 1 | 1 | 9.2356e-06 / 9.2733e-06 | 1.004 | 1.008 | 2000 yes, 2001 yes, 2002 yes; A2: 2000 yes, 2001 yes, 2002 yes |

Every complete cell is identical on every seed in best_bits, calls and charged.

## Provenance

| Component | Version | Source | sha256 |
| --- | --- | --- | --- |
| xtb | 6.7.1 (edcfbbe) | conda-forge xtb-6.7.1-h8876d29_4 from the rgpot 8efdd246 xtbbld lock | `481a1e7d8151fc08e0341b8e06ab60ac09cf1c2628738eeb83346271274bf8f4` |
| crest | 3.0.2 (6914a25) | conda-forge crest-3.0.2-h8876d29_1 | `16c5de4578000dfd8ea5758da9f5c497778be0dd7a6a1c1e8933030479d0e56e` |
| libxtb_engine.so | rgpot 8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e | meson release with_xtb with_fortran_pots in the locked xtbbld env | `f7e5a40140cac0c67863f4dce998408f288b098aad0c7efd0c3fc1fca1ac4de4` |
| librgpot.so | rgpot 8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e | same build; Sutton-Chen for h2h_sc | `53eba6d99d28d3f7f9704ea77922897db7760fb6e4108416bd50343938aac5ed` |
| gfn2_water_h2h | tree b04621fd81776cfb78626162c5b3e746acb9859f | cargo 1.88.0 release --locked --features rgpot-ex | `edf5215f6da97f70370b4207dbac15b72fd3ff6640a4d9792b75b0b5bb66391b` |
| gfn2_water_crest_driver | tree b04621fd81776cfb78626162c5b3e746acb9859f | cargo 1.88.0 release --locked --features rgpot-ex | `261cdfaff865b00232b0844eb0958b568b3a2642bfed4772b95843cb79721adf` |

Starts: 12 regenerated by TIP's `gfn2_water_h2h` with `H2H_START_ONLY=1`, the switch its header documents. 12 of 12 are byte-identical to the same switch at c2de652d, the original panel's revision. Hashes are in `starts.csv`.

The generator moves every water 0.25 angstrom away from the centroid per step until no intermolecular contact is below 1.5 angstrom. Spread steps per seed: 2000 177, 2001 19, 2002 37, 2003 13, 2004 22, 2005 37, 2006 36, 2007 32, 2008 1, 2009 29, 2010 17, 2011 9.

Arms were checked by tree only: DEVELOP `cff796c507a8fa1d0bfb28888c3570b61988055e`, TIP `b04621fd81776cfb78626162c5b3e746acb9859f`. Every row carries its library tree, the runner blob, the driver blob and the hashed machine id.
