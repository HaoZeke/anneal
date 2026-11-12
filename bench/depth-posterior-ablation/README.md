# Depth-posterior ablation

LJ38, LJ55, LJ75, LJ98, and LJ104. Seeds 2000 through 2005, both arms, one serial session. `posterior` is the paper preset. `uniform` draws one move arm uniformly and leaves the reward, the screen, and the library alone.

A hit is a quenched energy within 0.001 of the Cambridge minimum. The budgets are 400000, 1000000, 4000000, 4000000, and 4000000 force calls. The median and IQR are the type-7 quartiles of the force calls on the hits. Per-seed calls and wall times are in `results.csv`. The raw logs are listed in `data/hf-files.csv` and stay out of git.

| Size | Arm | Hits | Median calls | IQR | Wall sum (s) |
| --- | --- | --- | --- | --- | --- |
| LJ38 | posterior | 6/6 | 13554.5 | 3033.5 | 0.964 |
| LJ38 | uniform | 6/6 | 12353.5 | 1120.5 | 0.814 |
| LJ55 | posterior | 6/6 | 17926.0 | 13316.8 | 2.273 |
| LJ55 | uniform | 6/6 | 12777.5 | 13479.8 | 2.517 |
| LJ75 | posterior | 5/6 | 443291.0 | 72338.0 | 643.468 |
| LJ75 | uniform | 6/6 | 921422.0 | 919620.0 | 670.476 |
| LJ98 | posterior | 4/6 | 1556640.0 | 771732.8 | 991.301 |
| LJ98 | uniform | 6/6 | 1554400.0 | 1246638.0 | 733.009 |
| LJ104 | posterior | 3/6 | 2180278.0 | 1247471.5 | 1353.324 |
| LJ104 | uniform | 2/6 | 3081434.0 | 466361.0 | 1753.213 |
