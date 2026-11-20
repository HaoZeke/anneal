# GFN2 floor-gate rows

The `floor-rerun-grad-fix`, `series-grad-fix` and `grad-grad-fix` rows in
`gfn2.csv` are one run per seed, not triplicates. For each of Si13 and
C20 and each seed from 2000 to 2011, the three labels store the same
`hit`, `evals_to_hit`, `charged`, `probe_calls`, `best`, `scc_rejected`,
`geometry_rejected` and `code_revision`. The wall-clock columns differ.
The file repeats that one run under the three labels.

The rows are seeds 2000-2011 at a budget of 100,000 calls, serial on core
0 with one thread, xTB 6.7.1, electronic temperature 3000 K, and the
Cooperlake OpenBLAS kernel. The rgpot revision is
`8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`. The example, engine and xTB
library SHA-256 digests are
`2e67177d880ee8663ea9f778d8614a32252e6d07a6ef81ac0a4e12344af3148e`,
`4d9dd694c1f861285d867eb019c95505364ae12dd251250de101922a1a4ed451`
and
`39232b70c4669a284e5136fc649520ab7239eb859df15c8d7af98ad5e3bfa7c9`.
A central finite difference on three geometries gave a maximum absolute
gradient error of `9.392943312037e-9` eV/angstrom.

These rows were produced with the code at `fix/gfn2-grad`
`764aa838e4acf9a4b0c2edb999ed2cf4e4dfe0a4` plus that kit's patch 1, which
is not on develop. The `code_revision` on every one of these rows is
`a37896fdf9d098a879db18f6a5d5e30b95b65da5`.

Hits are scored against the -599.345 stop target. That value is a stop
target and not a verified global minimum, so hit counts may be inflated
relative to a true-minimum reference. Eight Si13 seeds store `best`
`-599.34546694` and seed 2009 stores `-599.34536175`. Seeds 2000, 2001
and 2008 store `-641.40367700`, `-618.15035402` and `-618.20745111`.
The `hit` flag is true on all twelve Si13 seeds.

These rows are not comparable with the other GFN2 rows in the repo.

The hit counts are the `hit` column. Each label stores the same count,
and the three labels are the one run above, so the count is once per
seed. C20 is the value stored in the file.

| system | seeds | hits |
|---|---|---:|
| Si13 | 2000-2011 | 12/12 |
| C20 | 2000-2011 | 0/12 |

C20 is `false` on all 12 rows of each label, and each of those rows has
`evals_to_hit` 100000. `best` on those rows runs from `-1148.29290847`
to `-1145.94033855`. The same file also holds the earlier `base`,
`floor`, `floor-rerun`, `series` and `grad` arms. The table counts the
three labels of this run.
