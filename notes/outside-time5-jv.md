# Outside-time 5: call counts and a Jonker-Volgenant attempt

Base `22110d77784e6a9e8282c2f0758ec980f55ea9eb`. The column scan and the
reduced-cost adjust are about 37% of Au98 time outside the potential.
The question was whether those calls repeat a matrix, or a matrix that
barely moved. They do not. Replacing the scan with Jonker-Volgenant is
the same permutation and a slower one, so it is not the default.

## What was counted

A 40,000-evaluation seed, serial. The residual memo is keyed on the
coordinate bits and the axis bits. A miss builds the squared-distance
matrix of the centred cluster against its fivefold rotation and solves
the assignment. A revisit is a later miss on an axis whose bits matched
a matrix already solved.

Au98, seed 2000:

| | |
| --- | ---: |
| memo hits | 90 |
| memo misses | 1,110 |
| strict row-minimum exits | 0 |
| solver calls | 1,110 |
| unique optima | 1,110 |
| distinct axis bit-patterns | 668 |
| revisits | 442 |
| revisits with identical cost bits | 0 |
| revisits whose largest entry moved by less than 1e-3 | 0 |
| mean rows changed on a revisit | 98 / 98 |
| mean largest absolute entry change | 67.5 |
| revisits that kept the permutation | 142 |

LJ150, seed 2000, same budget:

| | |
| --- | ---: |
| memo hits | 52 |
| memo misses | 748 |
| solver calls | 748 |
| unique optima | 748 |
| distinct axis bit-patterns | 439 |
| revisits | 309 |
| revisits with identical cost bits | 0 |
| mean rows changed on a revisit | 150 / 150 |
| mean largest absolute entry change | 22.5 |
| revisits that kept the permutation | 63 |

Every solved optimum in both runs was unique, so any optimal permutation
would have matched the shortest-path one. The permutation still changed
on most revisits, because the costs moved by tens, not by a rounding gap.

## Jonker-Volgenant

`jonker_unique` is column reduction, one reduction transfer, two
augmenting-row passes, then shortest paths. It returns the permutation
only when the dual is feasible and the optimum is unique. On 24 rotated
clusters (six each at n = 8, 32, 98, 150) it agreed with the cold scan
every time and fell back none.

It is not the default. After the initialization, 49 to 66 of 98 rows are
still free, and the scalar paths that finish them lose to the existing
vector scan. Release Criterion for `minimum_cost_assignment` at n = 98
was about 41 µs for that scan and 80.6 µs with this initializer on the
same squared-distance matrix.
