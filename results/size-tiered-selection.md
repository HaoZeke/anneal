# Held-out selector panel

Seeds 7000-7005 at the paper budgets (4e6 calls on LJ75, LJ98, and LJ104).
A hit is a recorded minimum within 0.001 of the Cambridge energy.
A miss counts as the full budget.
The median and the interquartile range are Hyndman-Fan type 7.
The three arms ran serially in one session, hostname `bench`, with the
job order shuffled at seed 712801 and `RAYON_NUM_THREADS=1`.

The candidate rule kept a uniform move draw until the climb's
unprecedented-stall test, then handed later draws to the depth posterior.
The posterior was updated during the uniform phase.
On this panel the stall test did not fire inside the budget, and every
tiered seed repeated the uniform trajectory.
The arm ties the uniform draw.
It does not beat both fixed settings, so the selector is not part of
`Config::recommended`.

| Cluster | Arm | Hits | Median calls | IQR | Wall sum (s) |
| --- | --- | --- | --- | --- | --- |
| LJ75 | tiered | 6/6 | 1516553.5 | 994249 | 771.092 |
| LJ75 | posterior | 6/6 | 2457196.5 | 2153864.5 | 1054.282 |
| LJ75 | uniform | 6/6 | 1516553.5 | 994249 | 771.418 |
| LJ98 | tiered | 3/6 | 2192249.5 | 3712256.25 | 905.065 |
| LJ98 | posterior | 5/6 | 1628561.5 | 1553935.75 | 703.868 |
| LJ98 | uniform | 3/6 | 2192249.5 | 3712256.25 | 903.212 |
| LJ104 | tiered | 4/6 | 1499422 | 2419232.75 | 980.673 |
| LJ104 | posterior | 2/6 | 4000000 | 1038963.75 | 1371.079 |
| LJ104 | uniform | 4/6 | 1499422 | 2419232.75 | 977.345 |

Rows are in `size-tiered-selection.csv`.
