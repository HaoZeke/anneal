# OCCat head-to-head

Wang, Xu, and Cheng, Journal of Computational Chemistry 47, e70583 (2026), doi:10.1002/jcc.70583.
The release is https://github.com/LEDlamar/OCCat_release revision 64330de6a71259471d3a99a5d2afabb0a46a9ea0.
Anneal and OCCat ran in one serial session under `sudo unshare --uts` with hostname bench, core 0.
GMIN rows already stored at the same seed and budget were copied.
A hit is a quenched energy within 0.001 of the Cambridge value.
A miss is the full paper budget.
Medians are Hyndman-Fan type 7.
Inside is time in the potential. Outside is the rest of the wall.
The wall ratio is that code's median wall divided by anneal's.

| System | Code | Hits | Median evals | Median wall (s) | Median inside (s) | Median outside (s) | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | anneal | 12/12 | 13469.500000 | 0.196437 | 0.144366 | 0.053257 | 1.000000 |
| lj38 | occat | 0/12 | 400000.000000 | 345.737106 | 336.855606 | 8.634596 | 1760.045137 |
| lj38 | gmin | 7/12 | 389399.000000 | 3.356670 | 2.457148 | 0.892349 | 17.087812 |
| lj55 | anneal | 12/12 | 6080.000000 | 0.189617 | 0.133079 | 0.056539 | 1.000000 |
| lj55 | occat | 0/12 | 1000000.000000 | 1797.669867 | 1769.067673 | 28.602195 | 9480.506108 |
| lj55 | gmin | 12/12 | 12764.500000 | 0.205454 | 0.167391 | 0.038122 | 1.083521 |
| lj75 | anneal | 5/6 | 1485262.000000 | 79.748498 | 59.727572 | 20.020927 | 1.000000 |
| lj75 | occat | 0/6 | 4000000.000000 | 13369.593296 | 13214.802148 | 156.346366 | 167.646959 |
| lj75 | gmin | 0/6 | 4000000.000000 | 101.313500 | 86.351677 | 14.949635 | 1.270413 |
| cu38 | anneal | 12/12 | 7426.500000 | 0.155761 | 0.124461 | 0.033680 | 1.000000 |
| cu38 | occat | 0/12 | 400000.000000 | 370.751968 | 179.488185 | 191.263784 | 2380.261866 |
| cu38 | gmin | 0/12 | 400000.000000 | 11.972000 | 11.006452 | 0.952224 | 76.861345 |
| cu55 | anneal | 1/1 | 2341.000000 | 0.108601 | 0.076119 | 0.032482 | 1.000000 |
| au55 | anneal | 8/12 | 287363.500000 | 11.382569 | 8.973861 | 2.014004 | 1.000000 |
| cu6au32 | anneal | 12/12 | 400000.000000 | 6.507066 | 6.036478 | 0.473206 | 1.000000 |
| cu19au19 | anneal | 12/12 | 400000.000000 | 6.504130 | 6.038799 | 0.463610 | 1.000000 |
| cu27au28 | anneal | 12/12 | 1000000.000000 | 33.265678 | 31.529999 | 1.735680 | 1.000000 |

Fisher is the two-sided exact test on hits against anneal. Mann-Whitney is two-sided on evaluations, wall, time inside the potential, and time outside it. A miss is stored as the budget. A system runs seeds 2000-2005 and continues through 2011 when six more seeds of the slower code fit in six hours.

| System | n | Anneal | OCCat | GMIN | Fisher OCCat | Fisher GMIN | MW evals OCCat | MW wall OCCat | MW inside OCCat | MW outside OCCat | MW evals GMIN | MW wall GMIN | MW inside GMIN | MW outside GMIN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 12 | 12 | 0 | 7 | 7.39602e-07 | 0.0372671 | 8.90101e-06 | 3.22564e-05 | 3.22564e-05 | 3.22564e-05 | 2.97837e-05 | 4.14604e-05 | 4.14604e-05 | 4.14604e-05 |
| lj55 | 12 | 12 | 0 | 12 | 7.39602e-07 | 1 | 8.90101e-06 | 3.22564e-05 | 3.22564e-05 | 3.22564e-05 | 0.0567468 | 0.563703 | 0.386476 | 0.326348 |
| lj75 | 6 | 5 | 0 | 0 | 0.0151515 | 0.0151515 | 0.00739713 | 0.00394775 | 0.00394775 | 0.00394775 | 0.00739713 | 0.42334 | 0.336668 | 1 |
| cu38 | 12 | 12 | 0 | 0 | 7.39602e-07 | 7.39602e-07 | 8.90101e-06 | 3.22564e-05 | 3.22564e-05 | 3.22564e-05 | 8.90101e-06 | 3.22564e-05 | 3.22564e-05 | 3.22564e-05 |
