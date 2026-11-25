# Size-free surface rules for lj98, against develop and 52af8e47

No rule tried here meets the bar on every cell. The bar: on lj75 (24
seeds), lj98 (12 seeds) and lj102 (12 seeds), no loss in hits and none
in calls to hit against develop, and keep the lj102 gain. None of the
rules is applied. `rules.diff` holds them all behind `ANNEAL_RULE`, a
bit field read once per run. It applies on 52af8e47.

## Arms and runs

- A: develop e51b3d2603facb856ad7b577b745495145e30297, which installs
  no surface at 98 points or above.
- H: 52af8e47488deb96225cbbcb25b6d12ff93dbd8d, tree
  827c5719762369fed7278fab5a5657a3524b8e38.
- Xk: H plus `rules.diff`, run with `ANNEAL_RULE=k`.

Each arm ran its own `examples/h2h_seed lj <n> 0 4000000 <seed>
recommended <stop>` (cargo --release --locked). The stops were
-397.491331 (lj75), -543.664361 (lj98) and -569.362652 (lj102), each
the reference plus 1e-3. Seeds were lj75 2000-2023 and lj98 and lj102
2000-2011. Rows marked `serial` ran one at a time under `taskset -c k`,
with up to seven such streams on separate cores. Rows marked
`parallel8` are the A and H rows for lj98 and for lj102 2001-2011. They
come from the earlier round, with eight runs at a time. Call counts do
not depend on scheduling, so `rows.csv` has no times. The rows come
from four successive builds of the same diff. The final build matched
the earlier rows call for call on rule 0 (lj75 2000, 2004), rule 32
(2000, 2001), rule 1 (2004, 2007), rule 4 (2001) and rule 128 (2004).

`python3 stats.py rows.csv` prints the table below. It shows geometric
means of per-seed ratios to A, with a miss counted at the budget, and
95 per cent bootstrap intervals over seeds.

## Rules

- 1: a restraint block ends when its charged calls reach `block` hops
  at the run's plain cost per hop, not after `block` hops. Credit stays
  the block's signed gap, so the arms are compared on equal calls.
- 8: the mirror image of 1. Plain blocks run until they reach `block`
  hops at the restraint's cost per hop.
- 4: a block draws the restraint only while the run is stalled, which
  means it has spent at least twice the calls it took to reach its
  best. Otherwise it walks the plain surface.
- 2: as 4, but once the run has stalled the restraint stays on offer.
  It was run on the first round only and is not in the table.
- 32: no unrestrained quench beside a restrained relaxation.
- 128: the unrestrained quench runs while its charged calls are at
  most one hop's worth (the run's calls per hop) for each minimum it
  supplied, plus one.
- 256: the unrestrained quench runs only after a restraint block in
  this run has failed and put the chain on a plain hold. This is the
  step #168 describes, the one that leaves a basin the restraint has
  stopped leaving.

Combinations: 33 = 32 + 1, 40 = 32 + 8, 129 = 128 + 1.

## Results against A

| cell | arm | hits (A) | calls to hit [95%] | calls per hop [95%] |
|---|---|---|---|---|
| lj75 | H | 24 (24) | 0.907 [0.605, 1.370] | 1.153 [1.001, 1.379] |
| lj75 | X1 | 24 (24) | 1.359 [0.773, 2.497] | 1.067 [0.887, 1.314] |
| lj75 | X4 | 24 (24) | 5.068 [3.067, 8.176] | 0.449 [0.365, 0.551] |
| lj75 | X8 | 24 (24) | 0.907 [0.605, 1.370] | 1.153 [1.001, 1.379] |
| lj75 | X32 | 24 (24) | 1.043 [1.000, 1.131] | 0.980 [0.947, 1.000] |
| lj75 | X33 | 24 (24) | 1.485 [1.102, 2.166] | 0.899 [0.821, 0.967] |
| lj75 | X40 | 24 (24) | 1.287 [1.000, 1.863] | 0.913 [0.808, 1.000] |
| lj75 | X128 | 24 (24) | 0.919 [0.611, 1.381] | 1.117 [0.972, 1.338] |
| lj75 | X129 | 24 (24) | 1.152 [0.716, 1.854] | 1.053 [0.898, 1.281] |
| lj75 | X256 | 24 (24) | 1.042 [1.000, 1.127] | 0.980 [0.947, 1.000] |
| lj98 | H | 11 (10) | 1.285 [0.418, 3.644] | 1.200 [1.019, 1.564] |
| lj98 | X1 | 10 (10) | 0.985 [0.342, 2.383] | 1.216 [1.043, 1.564] |
| lj98 | X4 | 12 (10) | 0.958 [0.590, 1.598] | 1.047 [1.005, 1.090] |
| lj98 | X8 | 11 (10) | 1.088 [0.361, 2.998] | 1.181 [1.011, 1.530] |
| lj98 | X32 | 11 (10) | 1.192 [0.746, 1.967] | 1.146 [1.067, 1.256] |
| lj98 | X33 | 10 (10) | 1.632 [1.033, 2.631] | 1.049 [1.008, 1.094] |
| lj98 | X40 | 9 (10) | 0.796 [0.328, 1.701] | 1.088 [1.010, 1.190] |
| lj98 | X128 | 12 (10) | 1.570 [0.631, 3.713] | 1.033 [0.978, 1.089] |
| lj98 | X129 | 11 (10) | 1.361 [0.836, 2.170] | 1.037 [1.001, 1.080] |
| lj98 | X256 | 11 (10) | 1.109 [0.570, 2.120] | 1.113 [1.050, 1.175] |
| lj102 | H | 12 (8) | 0.021 [0.009, 0.054] | 4.031 [2.819, 5.813] |
| lj102 | X1 | 12 (8) | 0.023 [0.010, 0.058] | 3.753 [2.511, 5.638] |
| lj102 | X4 | 12 (8) | 0.048 [0.022, 0.108] | 1.808 [1.439, 2.311] |
| lj102 | X8 | 12 (8) | 0.021 [0.009, 0.054] | 4.031 [2.819, 5.813] |
| lj102 | X32 | 12 (8) | 0.021 [0.009, 0.050] | 3.307 [2.555, 4.299] |
| lj102 | X33 | 12 (8) | 0.024 [0.010, 0.056] | 3.072 [2.269, 4.163] |
| lj102 | X40 | 12 (8) | 0.021 [0.009, 0.050] | 3.307 [2.555, 4.299] |
| lj102 | X128 | 12 (8) | 0.020 [0.009, 0.051] | 3.928 [2.698, 5.733] |
| lj102 | X129 | 12 (8) | 0.023 [0.010, 0.057] | 3.714 [2.466, 5.610] |
| lj102 | X256 | 12 (8) | 0.021 [0.009, 0.050] | 3.307 [2.555, 4.299] |

## Reading

Every rule keeps the lj102 gain. X4 keeps the least of it (0.048).

On lj75, the excess calls per hop of H are the unrestrained quench.
Without it (X32), or with it only after a failed block (X256), lj75
runs call for call as develop on 22 of 24 seeds. Calls per hop come to
0.980. The two seeds that differ, 2000 and 2008, are the ones where a
restraint block failed and the portfolio held the plain arm. Seed 2000
takes 354,459 calls under X256, against 143,083 under A and 8,739
under H. That one seed puts calls to hit at 1.042. Part of H's lj75 lead
over A comes from the same quench: 8,739 against 143,083 calls on seed 2000.

Charged-call blocks (1) put lj98 at 0.985, but they lose on lj75
(1.359). On 20 of 24 lj75 seeds X1 runs as H does. On the other four,
the shortened restraint blocks fail more often, and the holds that
follow cost up to 66 times H's calls (seed 2007: 2,924,560 against
44,025). Drawing the restraint only on a stall (4) is the best lj98
arm here (0.958, 12 hits). It loses five-fold on lj75, where develop
draws the restraint from the first block and the Marks decahedron
needs it early.

On lj98, X256 and X32 remove H's two worst seeds. Seed 2010 takes
100,572 calls (H 4,000,000, A 79,374). Seed 2011 takes 1,439,267
under X256 and 184,255 under X32 (H 2,787,070, A 217,593). Seed 2008
gets slower (X256 2,496,383, X32 a miss, A 645,383). The per-seed log
ratios on lj98 have a spread that 12 seeds cannot resolve. Every lj98
interval above crosses 1 except X33's, which lies above it. The lj98
part of the bar cannot be met or failed on these samples. A rule that
draws the restraint at 98 points walks a different chain from develop,
which never draws it. The ratio is then a draw from the spread of
develop's own seeds, whose calls to hit run from 79,374 to a miss.
