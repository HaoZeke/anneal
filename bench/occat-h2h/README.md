# OCCat head-to-head

Wang, Xu, and Cheng, Journal of Computational Chemistry 47, e70583 (2026), doi:10.1002/jcc.70583.
The release is https://github.com/LEDlamar/OCCat_release revision 64330de6a71259471d3a99a5d2afabb0a46a9ea0.
Anneal and OCCat ran in one serial session under `sudo unshare --uts` with hostname bench, core 0.
GMIN rows already stored at the same seed and budget were copied.
A hit is a quenched energy within 0.001 of the Cambridge value.
A miss is the full paper budget.
Medians are Hyndman-Fan type 7.
Inside is time in the potential. Outside is the rest of the wall.
Each wall is that code's own median. Anneal counts one energy-and-gradient pass. OCCat and GMIN count their own evaluations. This note prints no wall ratio and no cross-arm test. `tests.csv` keeps the stored columns.

| System | Code | Hits | Median evals | Median wall (s) | Median inside (s) | Median outside (s) |
| --- | --- | --- | --- | --- | --- | --- |
| lj38 | anneal | 12/12 | 13469.500000 | 0.196437 | 0.144366 | 0.053257 |
| lj38 | occat | 0/12 | 400000.000000 | 345.737106 | 336.855606 | 8.634596 |
| lj38 | gmin | 7/12 | 389399.000000 | 3.356670 | 2.457148 | 0.892349 |
| lj55 | anneal | 12/12 | 6080.000000 | 0.189617 | 0.133079 | 0.056539 |
| lj55 | occat | 0/12 | 1000000.000000 | 1797.669867 | 1769.067673 | 28.602195 |
| lj55 | gmin | 12/12 | 12764.500000 | 0.205454 | 0.167391 | 0.038122 |
| lj75 | anneal | 5/6 | 1485262.000000 | 79.748498 | 59.727572 | 20.020927 |
| lj75 | occat | 0/6 | 4000000.000000 | 13369.593296 | 13214.802148 | 156.346366 |
| lj75 | gmin | 0/6 | 4000000.000000 | 101.313500 | 86.351677 | 14.949635 |
| cu38 | anneal | 12/12 | 7426.500000 | 0.155761 | 0.124461 | 0.033680 |
| cu38 | occat | 0/12 | 400000.000000 | 370.751968 | 179.488185 | 191.263784 |
| cu38 | gmin | 0/12 | 400000.000000 | 11.972000 | 11.006452 | 0.952224 |
| cu55 | anneal | 1/1 | 2341.000000 | 0.108601 | 0.076119 | 0.032482 |
| au55 | anneal | 4/12 | 1000000.000000 | 35.722883 | 26.395429 | 9.241101 |
| cu6au32 | anneal | 12/12 | 400000.000000 | 6.507066 | 6.036478 | 0.473206 |
| cu19au19 | anneal | 12/12 | 400000.000000 | 6.504130 | 6.038799 | 0.463610 |
| cu27au28 | anneal | 12/12 | 1000000.000000 | 33.265678 | 31.529999 | 1.735680 |

A system runs seeds 2000-2005 and continues through 2011 when six more seeds of the slower code fit in six hours. A miss is the full paper budget.
