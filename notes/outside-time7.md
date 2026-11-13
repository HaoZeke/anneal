# Outside-time 7: seed the assignment dual

Base `98dcfb3106f191baf7381fecd5624426831094fe`. Same session, serial,
hostname bench, seeds 2000–2011, paper budgets.

A sampling profile of those runs puts `scan_avx2` at 35.6% of Au98
outside time and 13.7% of LJ150 outside time. It is the only leaf above
10% on both. The scan keeps its running minimum in registers. A
column-minimum dual then matches every row whose tight column is free,
and the shortest-path search runs only for the rest. That permutation is
kept only when the dual is feasible and the optimum is unique. Any other
matrix uses the unseeded scan, so the tie break is unchanged. Pins stay
`a22093e707faa91d683813fa7cc507706c5265e6`.

The median is the mean of the two central outside µs/call values.

| system | develop µs/call | seeded µs/call | change | hits |
| --- | ---: | ---: | ---: | --- |
| Au98 | 28.160 | 26.770 | −4.94% | 0 / 0 |
| LJ150 | 47.136 | 41.938 | −11.03% | 6 / 6 |

Charged evaluations, hit flags, and the best energy match on every seed.
The 10% gate is not met on Au98.
