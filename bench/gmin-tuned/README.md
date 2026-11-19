# GMIN inputs tuned per cell

The same procedure is used for every cell. Nothing in the scoring, the
grid, the budget fraction, the seeds or the hit rule depends on the size.
The size supplies the atom or molecule count, the Cambridge reference,
the paper budget, the Morse rho when the cell is Morse, and the Gupta
container radius when the cell is gold. Anneal is not tuned. It runs
`Config::recommended` through the example the existing panel uses, or
`h2h_seed` on rigid TIP4P.

## Cells and budgets

Cells are LJ38, LJ75, LJ98, LJ104, M80 at rho 6 and 14, Au38 on the
Darby Gupta gold row, and rigid TIP4P (H2O)n for n = 8..21. The call
budget is the paper budget already recorded for that cell: 400000 at 38
atoms, 4000000 at 75, 98, 104 and 80 atoms, and 2000000 for every TIP4P
size. A hit is a quenched energy at or below the Cambridge minimum plus
0.001.

## Documented defaults

The grid is relative to the defaults of the potential, which are the
same at every size of that potential.

| Potential | Translational step | Angular-move factor | Rigid angular step | Temperature | Step adjustment |
| --- | --- | --- | --- | --- | --- |
| Lennard-Jones | 0.40 | 0.51 | unused | 1.0 | on, target 0.5 |
| Morse | 0.40 | 0.51 | unused | 1.0 | on, target 0.5 |
| Gupta gold | 0.40 times r0/2^(1/6) | 0.51 | unused | 0.1 | on, target 0.5 |
| Rigid TIP4P | 0.40 | 0.51 | 0.30 | 2.0 | off (`FIXSTEP`) |

Lennard-Jones and Morse defaults are the Wales LJ38 introduction file.
Gupta uses that step converted by the gold nearest-neighbour distance
and the 0.1 temperature of the Wales Gupta example. TIP4P defaults are
the rigid TIP4P file: temperature fixed, step fixed. GMIN's `STEP` line
is the translational step, then the factor that selects an angular move,
then the rigid-body angular step. The TIP4P file leaves the third entry
unset, so the angular step is the source default 0.30. Gold's container
radius stays on GMIN's nearest-neighbour formula and is not a tuned knob.

## Binary

The existing panel names Wales revision
aac1548a8aec13b53ad12a8d6904625081685d2e. The public archive that
revision was taken from now serves
9cbec1a446352bc9ef064bd2b1a5767194655919 (tarball sha256
ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776).
This campaign builds that image. The panel patches are applied, and
the rigid TIP4P potential is timed with the same clock as the other
potentials. Alignment, MPI, OpenMP, sparse storage, Python and the
bundled BLAS and LAPACK are off. The link is the system reference
BLAS and LAPACK. `build.txt` records the compiler, the cmake flags
and the sha256 of the binary. The anneal examples are release builds
of 5eb79cdcb0a986984dc7b487e5a3cc93224246f9, locked.

## Grid

Every cell is scored on the same relative scales, 0.5, 1 and 2, for the
translational step and for the temperature. Where the documented input
adjusts the step, the target acceptance is 0.25, 0.5 and 0.75. Where the
documented input fixes the step, the rigid angular step takes the same
scales 0.5, 1 and 2 and the acceptance target is not a knob. That is 27
points at every cell. The point order is step scale, then the third
axis, then temperature scale, each axis ascending. A remaining tie takes
the earlier point.

## Tuning seeds and budget

Tuning seeds are 2000-2011's complement 3000-3007, eight seeds, the same
seeds and the same starting geometry at every point. The tuning budget
is 1/20 of the paper budget, integer division. Each point is scored by
hit count, then the median calls-to-hit among the hits, then the median
best energy. Medians are Hyndman-Fan type 7. A point with no hit has no
calls-to-hit median, so those points are separated by the median best
energy. The calls spent are the sum of charged evaluations over the 27
points and eight seeds.

## Evaluation

Evaluation seeds are 2000-2011, twelve seeds, the same seeds and the
same starting geometry on both arms. Both arms are release binaries.
The whole process tree is pinned to core 0, one thread, with every
thread count set to 1. Seed pairs run in ABBA then BAAB order, so each
slot carries each arm equally often. A run stops at a hit, at the paper
budget, or at the 7200 second wall cap, and the cap is the same on both
arms. There is no quick mode and no discarded warm-up seed.

Each stored row records calls, calls-to-hit, best energy, wall, time
inside the potential, time outside it, outside time per call, core,
affinity, the hashed machine id, and one command that repeats the run.
`--verify` repeats that command for one stored row and checks the hit,
the calls and the best energy.

The same-binary A/A block runs GMIN against GMIN on LJ38, ABBA, twelve
seeds, the tuned LJ38 input, and is the noise band.

Hits use Fisher's exact test. Calls to the minimum and wall to the
minimum use the two-sided Mann-Whitney test, normal approximation with
the tie and continuity corrections; a miss ranks above every hit, and a
wall-capped miss censors the test. Bonferroni multiplies each family
by the number of cells in the table and caps the result at 1. Fisher,
calls and wall are three families.

<!-- gmin-tuned results: start -->
## Tuned values

| Cell | Step | Angular | Temperature | Accept | Tuning hits | Median calls-to-hit | Median best | Tuning calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 0.400000 | - | 1.0 | 0.75 | 2/8 | 2462.50 | -173.1343047650 | 4113792 |
| lj75 | 0.200000 | - | 1.0 | 0.25 | 1/8 | 103721.00 | -396.2822287785 | 43103721 |
| lj98 | 0.200000 | - | 1.0 | 0.5 | 1/8 | 196626.00 | -542.9495961815 | 43196626 |
| lj104 | 0.400000 | - | 2.0 | 0.5 | 3/8 | 119863.00 | -581.4731419155 | 42574304 |
| m80r6 | 0.400000 | - | 1.0 | 0.5 | 6/8 | 70700.50 | -378.3334452905 | 39509691 |
| m80r14 | 0.200000 | - | 1.0 | 0.5 | 0/8 | - | -330.9385094590 | 43200000 |
| au38 | 1.027741 | - | 0.05 | 0.5 | 0/8 | - | -130.4725746295 | 4320000 |
| tip4p8 | 0.800000 | 0.600000 | 4.0 | - | 5/8 | 26006.00 | -305.5189259355 | 17835768 |
| tip4p9 | 0.800000 | 0.600000 | 4.0 | - | 4/8 | 20434.50 | -343.9859094935 | 17968591 |
| tip4p10 | 0.800000 | 0.600000 | 4.0 | - | 4/8 | 84670.00 | -390.9904163695 | 20609914 |
| tip4p11 | 0.800000 | 0.600000 | 4.0 | - | 3/8 | 38223.00 | -431.3198293210 | 21049410 |
| tip4p12 | 0.800000 | 0.150000 | 4.0 | - | 2/8 | 42963.00 | -487.0735896935 | 21186950 |
| tip4p13 | 0.800000 | 0.600000 | 2.0 | - | 1/8 | 11659.00 | -529.7113295105 | 21511659 |
| tip4p14 | 0.800000 | 0.300000 | 2.0 | - | 0/8 | - | -579.0652378415 | 21600000 |
| tip4p15 | 0.800000 | 0.600000 | 4.0 | - | 1/8 | 25504.00 | -616.4690151405 | 21488597 |
| tip4p16 | 0.800000 | 0.150000 | 4.0 | - | 0/8 | - | -664.1749523155 | 21600000 |
| tip4p17 | 0.800000 | 0.150000 | 2.0 | - | 0/8 | - | -715.7365628065 | 21600000 |
| tip4p18 | 0.400000 | 0.300000 | 4.0 | - | 0/8 | - | -757.1351445360 | 21600000 |
| tip4p19 | 0.800000 | 0.600000 | 4.0 | - | 0/8 | - | -805.5228672455 | 21600000 |
| tip4p20 | 0.800000 | 0.150000 | 1.0 | - | 0/8 | - | -847.3777031115 | 21600000 |
| tip4p21 | 0.400000 | 0.300000 | 4.0 | - | 0/8 | - | -893.1301827330 | 21600000 |

## Evaluation

| Cell | Arm | Hits | Calls | IQR | To hit | IQR | Wall | IQR | Inside | Outside | Out/call | Fisher | Bonferroni | MW calls | Bonferroni | MW wall | Bonferroni |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | anneal | 12/12 | 127839.5 | 139245.2 | 127839.5 | 139245.2 | 0.5933 | 0.6954 | 0.2803 | 0.3131 | 2.45e-06 | 6.73038e-05 | 0.000336519 | 0.00442148 | 0.0221074 | 0.00442148 | 0.0221074 |
|  | GMIN | 2/12 | 400000.0 | 0 | 400000.0 | 0 | 3.353 | 0.02135 | 2.438 | 0.9116 | 2.279e-06 |  |  |  |  |  |  |
| lj75 | anneal | 5/12 | 4000000.0 | 962776.8 | 4000000.0 | 962776.8 | 56.829 | 14.501 | 33.313 | 23.581 | 6.072e-06 | 0.0372671 | 0.186335 | 0.0164732 | 0.0823658 | 0.0164732 | 0.0823658 |
|  | GMIN | 0/12 | 4000000.0 | 0 | 4000000.0 | 0 | 106.8 | 1.335 | 91.874 | 14.790 | 3.697e-06 |  |  |  |  |  |  |
| m80r6 | anneal | 12/12 | 62677.0 | 575467.8 | 62677.0 | 575467.8 | 2.249 | 21.739 | 1.650 | 0.5992 | 1.147e-05 | 1 | 1 | 0.260236 | 1 | 0.285476 | 1 |
|  | GMIN | 12/12 | 268934.0 | 268091.5 | 268934.0 | 268091.5 | 9.739 | 9.431 | 8.725 | 1.014 | 3.733e-06 |  |  |  |  |  |  |
| m80r14 | anneal | 6/12 | 3901429.0 | 1446773.0 | 3901429.0 | 1446773.0 | 136.8 | 49.698 | 102.0 | 34.352 | 9.035e-06 | 0.0686499 | 0.343249 | 0.0339762 | 0.169881 | 0.0339762 | 0.169881 |
|  | GMIN | 1/12 | 4000000.0 | 0 | 4000000.0 | 0 | 141.1 | 0.4637 | 127.9 | 13.171 | 3.293e-06 |  |  |  |  |  |  |
| au38 | anneal | 8/12 | 186585.0 | 264407.8 | 186585.0 | 264407.8 | 3.145 | 4.666 | 2.357 | 0.7968 | 4.36e-06 | 0.00942253 | 0.0471127 | 0.00587432 | 0.0293716 | 0.00313835 | 0.0156917 |
|  | GMIN | 1/12 | 400000.0 | 0 | 400000.0 | 0 | 11.756 | 0.01565 | 10.795 | 0.9625 | 2.407e-06 |  |  |  |  |  |  |

Bonferroni uses m = 5 cells in this table.

## LJ38 same-binary noise

Absolute difference of the two GMIN arms, median, IQR and maximum.

- Calls: 0 0 0
- Calls-to-hit: 0 0 0
- Best energy: 0 0 0
- Wall seconds: 0.006245 0.01621 0.04041
- Outside seconds per call: 8.612e-09 4.4085e-09 1.7357e-08
<!-- gmin-tuned results: end -->
