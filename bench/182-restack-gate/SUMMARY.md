# Restack gate

Overall: FAIL.

Performance uses hit loss, strict total outside time, and the
one-sided RssAnon bar. Parity is B against P on cluster cells.
TIP4P identity is B against A. A strict time failure stays a
failure when the adjusted limit would pass. Per-call time is not
the time bar.

| Cell | Hits A/B/P/A2 | Outside B/A | Rss B/A | Perf | Parity | Id |
| --- | ---: | --- | --- | --- | --- | --- |
| m38r14 | 12/12/12/12 | 0.023101279 [0.016642434, 0.032340228] | 0.17949362 [0.15789149, 0.20379412] | PASS | 1 | - |
| sc126-38 | 12/12/12/12 | 0.018369814 [0.010069816, 0.035490034] | 0.2749738 [0.21947464, 0.33933938] | PASS | 1 | - |
| sc126-55 | 12/12/12/12 | 0.059917303 [0.033717084, 0.10053617] | 0.26881341 [0.21266157, 0.31790211] | PASS | 1 | - |
| sc108-38 | 7/12/12/7 | 0.0064785656 [0.0039664144, 0.011169158] | 0.11044424 [0.077628336, 0.1615688] | PASS | 1 | - |
| sc96-38 | 12/12/12/12 | 0.024458713 [0.01901078, 0.030753828] | 0.32443613 [0.29889943, 0.34938692] | PASS | 1 | - |
| sc96-55 | 12/12/12/12 | 0.080797217 [0.057232072, 0.1155732] | 0.28901718 [0.26745201, 0.31028713] | PASS | 1 | - |
| lj38 | 12/12/12/12 | 0.014075071 [0.0084579245, 0.024106853] | 0.16951362 [0.13783535, 0.20866788] | PASS | 1 | - |
| lj55 | 12/12/12/12 | 0.1324086 [0.082988573, 0.2162553] | 0.22941726 [0.19289667, 0.27466568] | PASS | 1 | - |
| m38r6 | 12/12/12/12 | 0.020350013 [0.011723263, 0.033191997] | 0.1415976 [0.10422322, 0.18340822] | PASS | 1 | - |
| m55r6 | 12/12/12/12 | 0.027903422 [0.018119197, 0.043442122] | 0.1337836 [0.1089652, 0.16071774] | PASS | 1 | - |
| lj98 | 7/7/7/7 | 0.99698407 [0.97436533, 1.0177842] | 0.99890031 [0.99609004, 1.0007603] | PASS | 0 | - |
| lj75 | 5/5/5/5 | 1.0125857 [0.99565544, 1.0323765] | 0.99996611 [0.99911039, 1.0009221] | PASS | 0 | - |
| tip4p9 | 12/12/-/12 | 0.98453218 [0.97227375, 0.99446574] | 0.99984448 [0.99852419, 1.0011581] | PASS | - | 1 |
| tip4p10 | 12/12/-/12 | 0.98975416 [0.97863682, 0.99897697] | 1.0002199 [0.9991912, 1.0015935] | PASS | - | 1 |
| tip4p14 | 12/12/-/12 | 1.0040024 [0.97812804, 1.0323975] | 1.0000531 [0.99986569, 1.0002267] | PASS | - | 1 |
| tip4p16 | 2/2/-/2 | 0.98075238 [0.97021459, 0.99076178] | 1.0000057 [0.99997922, 1.0000313] | PASS | - | 1 |
| tip4p17 | 5/5/-/5 | 0.99496103 [0.98344032, 1.0072678] | 0.99999877 [0.99996221, 1.0000296] | PASS | - | 1 |

## Gates

- m38r14: hits 1 Fisher B/A 1 MWU B/A 3.6442027e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.72788257 [0.63200124, 0.83960274] artifact 0; rss pass 1 real 0 limit 1.1
- sc126-38: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.93761845 [0.80776063, 1.0798218] artifact 0; rss pass 1 real 0 limit 1.1
- sc126-55: hits 1 Fisher B/A 1 MWU B/A 3.6442027e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.48749688 [0.44634313, 0.52978203] artifact 0; rss pass 1 real 0 limit 1.1
- sc108-38: hits 1 Fisher B/A 0.037267081 MWU B/A 3.3282095e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.81142815 [0.70820451, 0.92777696] artifact 0; rss pass 1 real 0 limit 1.1
- sc96-38: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.89743295 [0.76416321, 1.0495441] artifact 0; rss pass 1 real 0 limit 1.1
- sc96-55: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.55701586 [0.5013146, 0.61966422] artifact 0; rss pass 1 real 0 limit 1.1
- lj38: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.8046116 [0.67393006, 1.005883] artifact 0; rss pass 1 real 0 limit 1.1
- lj55: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.61732148 [0.49786939, 0.77741418] artifact 0; rss pass 1 real 0 limit 1.1
- m38r6: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.55071437 [0.48570423, 0.62823823] artifact 0; rss pass 1 real 0 limit 1.1
- m55r6: hits 1 Fisher B/A 1 MWU B/A 3.6584554e-05; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.58400667 [0.53710645, 0.63766849] artifact 0; rss pass 1 real 0 limit 1.1
- lj98: hits 1 Fisher B/A 1 MWU B/A 0.85733041; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.99359334 [0.97193466, 1.0123682] artifact 0; rss pass 1 real 0 limit 1.1
- lj75: hits 1 Fisher B/A 1 MWU B/A 0.89741747; time strict 1 adjusted 1 real 0 limit 1.1; per-call 1.0116371 [0.99490535, 1.031204] artifact 0; rss pass 1 real 0 limit 1.1
- tip4p9: hits 1 Fisher B/A 1 MWU B/A 1; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.98453218 [0.97227376, 0.99446574] artifact 0; rss pass 1 real 0 limit 1.1
- tip4p10: hits 1 Fisher B/A 1 MWU B/A 1; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.98975417 [0.97863681, 0.99897699] artifact 0; rss pass 1 real 0 limit 1.1
- tip4p14: hits 1 Fisher B/A 1 MWU B/A 1; time strict 1 adjusted 1 real 0 limit 1.1; per-call 1.0040024 [0.97812804, 1.0323975] artifact 0; rss pass 1 real 0 limit 1.1
- tip4p16: hits 1 Fisher B/A 1 MWU B/A 1; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.98075237 [0.9702146, 0.99076177] artifact 0; rss pass 1 real 0 limit 1.1
- tip4p17: hits 1 Fisher B/A 1 MWU B/A 1; time strict 1 adjusted 1 real 0 limit 1.1; per-call 0.99496103 [0.98344032, 1.0072678] artifact 0; rss pass 1 real 0 limit 1.1

## Parity and identity

- m38r14 B/A: seed=2000 field=evals B=1499 A=20225
- m38r14 anchor seed 2000 evals 1500 match 0
- sc126-38 B/A: seed=2000 field=evals B=1163 A=256599
- sc126-38 anchor seed 2000 evals 1164 match 0
- sc126-55 B/A: seed=2000 field=evals B=2910 A=8993
- sc126-55 anchor seed 2000 evals 2911 match 0
- sc108-38 B/A: seed=2000 field=evals B=1677 A=244800
- sc108-38 anchor seed 2000 evals 1678 match 0
- sc96-38 B/A: seed=2000 field=evals B=1288 A=20721
- sc96-38 anchor seed 2000 evals 1289 match 0
- sc96-55 B/A: seed=2000 field=evals B=2684 A=23787
- sc96-55 anchor seed 2000 evals 2685 match 0
- lj38 B/A: seed=2000 field=evals B=602 A=5527
- lj38 anchor seed 2000 evals 603 match 0
- lj55 B/A: seed=2000 field=evals B=1486 A=5067
- lj55 anchor seed 2000 evals 1487 match 0
- m38r6 B/A: seed=2000 field=evals B=1614 A=67722
- m38r6 anchor seed 2000 evals 1615 match 0
- m55r6 B/A: seed=2000 field=evals B=1155 A=24224
- m55r6 anchor seed 2000 evals 1156 match 0
- lj98 B/P: seed=2000 field=charged B=370989 P=365919
- lj98 B/A: seed=2000 field=evals B=365919 A=363251
- lj98 anchor seed 2000 evals 365919 match 1
- lj75 B/P: seed=2001 field=charged B=3330980 P=3329914
- lj75 B/A: seed=2001 field=evals B=3329914 A=3327845
- lj75 anchor seed 2001 evals 3329914 match 1

## Heated lattice exchange

lattice_exchange_follows_the_plateau passed on B and on P.
Cold fingerprint 0x100885ac118293e1 on both trees.
Heated fingerprint 0x44038697bde16fb7 on P.
Heated fingerprint 0x1a623a8e37ad1d43 on B.
