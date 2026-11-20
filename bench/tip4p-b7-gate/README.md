# tip4p b7 no-loss gate

FIX if B has fewer hits than A on any system, or a calls loss at p < 0.05, or a B/A outside-time-per-call ratio above 1.10 on any system. Otherwise PASS.

Verdict: FIX

Failing cells: tip4p18 (hits); tip4p19 (hits); tip4p20 (hits).

Seeds 2000-2011. One serial session, core 0, one thread in each pool. Arms alternate ABBA. A hit is a finite best at or below the reference plus 0.001. Misses rank at budget + 1 for the calls test. Outside time per call is outside seconds divided by charged calls on each row. The ratio is the type-7 median on B over the type-7 median on A. Fisher p is the exact two-sided test. Mann-Whitney p is the two-sided normal approximation with the tie and continuity corrections. A calls loss is that p below 0.05 with B ranked slower. best is the driver token. best_bits is the IEEE-754 encoding of that token. Per-job timeout 14400 seconds.

## Verdict

| system | hits A | hits B | Fisher p | calls A | calls B | MW U | MW p | wall A | wall B | outside/call A | outside/call B | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | 12/12 | 12/12 | 1 | 6339 | 5069.5 | 80 | 0.665006 | 0.020408 | 0.016825 | 1.2972e-06 | 1.28536e-06 | 0.990872 |
| tip4p9 | 12/12 | 12/12 | 1 | 5839.5 | 5183 | 61 | 0.54437 | 0.021305 | 0.018734 | 1.35037e-06 | 1.33811e-06 | 0.990925 |
| tip4p10 | 12/12 | 12/12 | 1 | 13138.5 | 13178.5 | 81 | 0.623605 | 0.054570 | 0.055758 | 1.48538e-06 | 1.50106e-06 | 1.01055 |
| tip4p11 | 12/12 | 12/12 | 1 | 109288 | 143767 | 72 | 1 | 0.524868 | 0.694843 | 1.60766e-06 | 1.65124e-06 | 1.0271 |
| tip4p12 | 12/12 | 12/12 | 1 | 41418.5 | 51587 | 50 | 0.214494 | 0.224745 | 0.283520 | 1.80149e-06 | 1.86779e-06 | 1.0368 |
| tip4p13 | 12/12 | 12/12 | 1 | 256396 | 126344 | 83 | 0.54437 | 1.861903 | 0.900033 | 2.00445e-06 | 2.08137e-06 | 1.03837 |
| tip4p14 | 12/12 | 12/12 | 1 | 135853 | 91917 | 103 | 0.0782522 | 1.092801 | 0.688161 | 2.32014e-06 | 2.33469e-06 | 1.00627 |
| tip4p15 | 5/12 | 6/12 | 1 | 2e+06 | 1.67992e+06 | 82 | 0.549957 | 16.691188 | 13.705671 | 2.60219e-06 | 2.55538e-06 | 0.982013 |
| tip4p16 | 1/12 | 3/12 | 0.590062 | 2e+06 | 2e+06 | 83.5 | 0.328106 | 18.250267 | 18.503283 | 2.77845e-06 | 2.96761e-06 | 1.06808 |
| tip4p17 | 3/12 | 4/12 | 1 | 2e+06 | 2e+06 | 72 | 1 | 21.687041 | 21.603082 | 3.35106e-06 | 3.41402e-06 | 1.01879 |
| tip4p18 | 3/12 | 2/12 | 1 | 2e+06 | 2e+06 | 68 | 0.775997 | 23.555778 | 23.831355 | 3.66558e-06 | 3.67121e-06 | 1.00154 |
| tip4p19 | 7/12 | 2/12 | 0.0893795 | 1.08178e+06 | 2e+06 | 44 | 0.0679386 | 13.053231 | 24.554539 | 3.39958e-06 | 3.5422e-06 | 1.04195 |
| tip4p20 | 4/12 | 2/12 | 0.640405 | 2e+06 | 2e+06 | 59 | 0.342778 | 26.876029 | 27.221206 | 3.76201e-06 | 3.82552e-06 | 1.01688 |
| tip4p21 | 0/12 | 0/12 | 1 | 2e+06 | 2e+06 | 72 | 1 | 28.553828 | 29.593145 | 4.09214e-06 | 4.26501e-06 | 1.04225 |
| lj38 | 12/12 | 12/12 | 1 | 127840 | 127840 | 72 | 1 | 0.644035 | 0.632973 | 2.66499e-06 | 2.59331e-06 | 0.973104 |
| lj75 | 5/12 | 5/12 | 1 | 4e+06 | 4e+06 | 72 | 1 | 58.026369 | 57.495261 | 6.23353e-06 | 6.17672e-06 | 0.990886 |
| lj98 | 7/12 | 7/12 | 1 | 3.16681e+06 | 3.16681e+06 | 72 | 1 | 76.452309 | 75.335931 | 9.40841e-06 | 9.50233e-06 | 1.00998 |
| m38r14 | 4/12 | 4/12 | 1 | 400001 | 400001 | 72 | 1 | 3.369713 | 3.379344 | 2.46874e-06 | 2.49152e-06 | 1.00923 |
| sc126-55 | 12/12 | 12/12 | 1 | 18379.5 | 18379.5 | 72 | 1 | 8.035564 | 8.035158 | 4.48447e-06 | 4.5148e-06 | 1.00676 |

## Hits by seed

| system | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p9 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p10 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p11 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p12 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p13 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p14 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| tip4p15 | 1/0 | 1/1 | 0/0 | 1/1 | 0/1 | 1/1 | 0/0 | 0/0 | 1/0 | 0/1 | 0/0 | 0/1 |
| tip4p16 | 0/1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 | 0/1 | 0/0 | 0/0 | 0/0 | 0/0 |
| tip4p17 | 0/0 | 0/0 | 1/1 | 0/0 | 0/1 | 0/1 | 0/1 | 0/0 | 1/0 | 0/0 | 0/0 | 1/0 |
| tip4p18 | 1/0 | 0/0 | 0/1 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 |
| tip4p19 | 1/0 | 0/1 | 0/0 | 0/0 | 0/0 | 1/0 | 1/0 | 1/1 | 0/0 | 1/0 | 1/0 | 1/0 |
| tip4p20 | 0/0 | 1/1 | 1/0 | 0/0 | 1/0 | 0/1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 |
| tip4p21 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| lj38 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| lj75 | 0/0 | 1/1 | 0/0 | 0/0 | 0/0 | 1/1 | 0/0 | 0/0 | 1/1 | 0/0 | 1/1 | 1/1 |
| lj98 | 1/1 | 1/1 | 1/1 | 1/1 | 0/0 | 1/1 | 1/1 | 0/0 | 1/1 | 0/0 | 0/0 | 0/0 |
| m38r14 | 0/0 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 | 1/1 | 0/0 | 1/1 |
| sc126-55 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

Each cell is hit A / hit B.

## Non-water bit identity

| system | seed | best_bits | charged |
| --- | --- | --- | --- |
| lj38 | 2000 | identical | identical |
| lj38 | 2001 | identical | identical |
| lj38 | 2002 | identical | identical |
| lj38 | 2003 | identical | identical |
| lj38 | 2004 | identical | identical |
| lj38 | 2005 | identical | identical |
| lj38 | 2006 | identical | identical |
| lj38 | 2007 | identical | identical |
| lj38 | 2008 | identical | identical |
| lj38 | 2009 | identical | identical |
| lj38 | 2010 | identical | identical |
| lj38 | 2011 | identical | identical |
| lj75 | 2000 | identical | identical |
| lj75 | 2001 | identical | identical |
| lj75 | 2002 | identical | identical |
| lj75 | 2003 | identical | identical |
| lj75 | 2004 | identical | identical |
| lj75 | 2005 | identical | identical |
| lj75 | 2006 | identical | identical |
| lj75 | 2007 | identical | identical |
| lj75 | 2008 | identical | identical |
| lj75 | 2009 | identical | identical |
| lj75 | 2010 | identical | identical |
| lj75 | 2011 | identical | identical |
| lj98 | 2000 | identical | identical |
| lj98 | 2001 | identical | identical |
| lj98 | 2002 | identical | identical |
| lj98 | 2003 | identical | identical |
| lj98 | 2004 | identical | identical |
| lj98 | 2005 | identical | identical |
| lj98 | 2006 | identical | identical |
| lj98 | 2007 | identical | identical |
| lj98 | 2008 | identical | identical |
| lj98 | 2009 | identical | identical |
| lj98 | 2010 | identical | identical |
| lj98 | 2011 | identical | identical |
| m38r14 | 2000 | identical | identical |
| m38r14 | 2001 | identical | identical |
| m38r14 | 2002 | identical | identical |
| m38r14 | 2003 | identical | identical |
| m38r14 | 2004 | identical | identical |
| m38r14 | 2005 | identical | identical |
| m38r14 | 2006 | identical | identical |
| m38r14 | 2007 | identical | identical |
| m38r14 | 2008 | identical | identical |
| m38r14 | 2009 | identical | identical |
| m38r14 | 2010 | identical | identical |
| m38r14 | 2011 | identical | identical |
| sc126-55 | 2000 | identical | identical |
| sc126-55 | 2001 | identical | identical |
| sc126-55 | 2002 | identical | identical |
| sc126-55 | 2003 | identical | identical |
| sc126-55 | 2004 | identical | identical |
| sc126-55 | 2005 | identical | identical |
| sc126-55 | 2006 | identical | identical |
| sc126-55 | 2007 | identical | identical |
| sc126-55 | 2008 | identical | identical |
| sc126-55 | 2009 | identical | identical |
| sc126-55 | 2010 | identical | identical |
| sc126-55 | 2011 | identical | identical |

Every non-water seed is bit-identical.

## Schedule and provenance

Counted (slot, arm) pairs: (0, A) = 114, (0, B) = 114, (1, A) = 114, (1, B) = 114.

Library tree A: `14b537fd9b73b8fe21e01721954b41a278c5d223`.
Library tree B: `c75edc53ed1ec9b91b1bd76372661ddec9e33a90`.
Harness blob: `94350c0821c0721bf2a77d6d48bab3351c05c799`.
Hashed machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.
