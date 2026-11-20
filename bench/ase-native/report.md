Native-unit ASE head-to-head

What changed relative to the previous ASE panel

The previous runs (scripts/ase_panel and scripts/ase_nonlj) evaluated
reduced-unit potentials with ASE 3.29 constructor defaults and stopped
ASE, and the recommended anneal preset, on a wall-clock cap. Basin
hopping used FIRE, fmax 0.1, dr 0.1, and
temperature 100 k_B (about 0.008617 eV).
Minima hopping used QuasiNewton, fmax 0.05, T0 1000 K,
Ediff0 0.5 eV, timestep 1.0 fs, mdmin 2,
and a minima threshold of 0.5 angstrom. The local
step cap stayed at the optimiser default of 0.2 angstrom.
The wall cap was three times the slowest anneal seed of that system.

This run keeps the same systems, the same paper evaluation budgets,
the same hit targets, and the anneal for_cluster preset. It drops the
wall cap. Potentials are parameterised in eV and angstroms (Lennard-Jones
epsilon = 1 eV, sigma = 1 angstrom; Morse epsilon = 1 eV, r0 = 1 angstrom;
Gupta gold already in eV and angstroms). ASE's fmax is
sqrt(3) * 1e-06 = 1.732051e-06 eV/angstrom, the Euclidean per-atom
norm that meets anneal's infinity-norm quench tolerance on the vector
(a, a, a). Basin-hopping temperature and step are 0.8 * energy_scale
and 0.38 * length_scale. Minima-hopping temperature and Ediff0 use that
same energy, the timestep is the preset 0.005 time unit at mass 1 u,
and the local optimiser is L-BFGS with 8 pairs, Armijo 1e-4, curvature
0.9, at most 40 line-search probes, and a per-atom cap of
min(1, length_scale). Anneal stops a quench
after 200 WarmLbfgs iterations. Those iterations are not ASE's, so an
ASE quench stops on fmax or after 4000 ASE iterations, whichever is
first. Basin hopping restores the centre of mass after its kick;
anneal's all-points move does not.
A hit is a recorded quench at or below the reference plus
1e-03. A miss is censored at the paper budget.

Recorded stack

ASE 3.29.0, numpy 2.5.5.
Anneal base 5eb79cdcb0a986984dc7b487e5a3cc93224246f9.
Anneal tree 7a2ba49b37241a8fe42b2c45c226374de6d92274.
Core 0. Machine 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873.

Fisher is the two-sided exact test on the hit table.
Mann-Whitney is the exact two-sided permutation test on midranks.
The energy comparison uses every run. The evaluation comparison
uses evaluations consumed: evaluations to the hit when the run
hit, and the paper budget when it did not.
Medians put a non-finite best energy after every finite one and,
for an even count, average the two central values.

## lj38

Budget 400000. Reference -173.928427. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 10 | 12 | -173.928 | 255288 |  |  |  | 0.94787 |
| ase_bh | 0 | 12 | -171.031 | 400000 | 6.73038e-05 | 1.4792e-06 | 6.73038e-05 | 46.5317 |
| ase_mh | 4 | 12 | -173.252 | 400000 | 0.0360748 | 8.87523e-05 | 0.0733841 | 47.7667 |

## lj55

Budget 1000000. Reference -279.24847. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12 | 12 | -279.248 | 119726 |  |  |  | 0.78309 |
| ase_bh | 0 | 12 | -273.631 | 1e+06 | 7.39602e-07 | 7.39602e-07 | 7.39602e-07 | 122.893 |
| ase_mh | 12 | 12 | -279.248 | 100220 | 1 | 5.84286e-05 | 0.842836 | 12.4078 |

## lj98

Budget 4000000. Reference -543.665361. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 4 | 12 | -543.547 | 4e+06 |  |  |  | 81.0077 |
| ase_bh | 0 | 12 | -533.956 | 4e+06 | 0.0931677 | 7.39602e-07 | 0.0931677 | 536.653 |
| ase_mh | 0 | 12 | -539.503 | 4e+06 | 0.0931677 | 6.5085e-05 | 0.0931677 | 618.116 |

## lj104

Budget 4000000. Reference -582.086642. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0 | 12 | -580.804 | 4e+06 |  |  |  | 89.6976 |
| ase_bh | 0 | 12 | -573.212 | 4e+06 | 1 | 7.39602e-07 | 1 | 539.962 |
| ase_mh | 0 | 12 | -577.442 | 4e+06 | 1 | 0.000368322 | 1 | 622.595 |

## lj150

Budget 4000000. Reference -893.310258. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0 | 12 | -884.653 | 4e+06 |  |  |  | 186.543 |
| ase_bh | 0 | 12 | -872.27 | 4e+06 | 1 | 7.39602e-07 | 1 | 652.311 |
| ase_mh | 0 | 12 | -878.263 | 4e+06 | 1 | 1.4792e-06 | 1 | 773.344 |

## au55

Budget 1000000. Reference -191.0308528. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0 | 12 | -190.811 | 1e+06 |  |  |  | 31.8153 |
| ase_bh | 0 | 12 | -190.296 | 1e+06 | 1 | 7.39602e-07 | 1 | 156.565 |
| ase_mh | 0 | 12 | -190.977 | 1e+06 | 1 | 0.000495534 | 1 | 133.493 |

## m38r6

Budget 400000. Reference -157.477108. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 5 | 12 | -157.407 | 400000 |  |  |  | 3.06381 |
| ase_bh | 0 | 12 | -154.421 | 400000 | 0.0372671 | 7.39602e-07 | 0.0372671 | 49.238 |
| ase_mh | 2 | 12 | -157.242 | 400000 | 0.370709 | 0.209327 | 0.241255 | 47.5294 |

## m38r14

Budget 400000. Reference -144.321054. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0 | 12 | -140.312 | 400000 |  |  |  | 3.03421 |
| ase_bh | 0 | 12 | -130.164 | 400000 | 1 | 7.39602e-07 | 1 | 46.1279 |
| ase_mh | 0 | 12 | -137.66 | 400000 | 1 | 7.39602e-07 | 1 | 46.3163 |

## m55r6

Budget 1000000. Reference -250.286609. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 12 | 12 | -250.287 | 81962.5 |  |  |  | 1.18034 |
| ase_bh | 0 | 12 | -242.46 | 1e+06 | 7.39602e-07 | 7.39602e-07 | 7.39602e-07 | 132.411 |
| ase_mh | 12 | 12 | -250.287 | 111748 | 1 | 7.39602e-07 | 0.442833 | 14.3995 |

## m55r14

Budget 1000000. Reference -220.646208. Tolerance 1e-03.

| arm | hits | n | median best | median evals | Fisher p | MW p energy | MW p evals | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0 | 12 | -218.67 | 1e+06 |  |  |  | 15.3501 |
| ase_bh | 0 | 12 | -196.902 | 1e+06 | 1 | 7.39602e-07 | 1 | 126.748 |
| ase_mh | 0 | 12 | -213.632 | 1e+06 | 1 | 7.39602e-07 | 1 | 128.609 |

Comparisons are against the anneal arm of the same system. ASE arms are ase_bh, ase_mh.
