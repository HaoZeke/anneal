Four Cambridge Morse systems across two sizes and two ranges: M38 and M55 at rho = 6 and rho = 14.
Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).
code_revision 9a30f3190b3bb7040c675a904f8388eb68224151.
Potential: untruncated Morse (`epsilon = 1`, `r0 = 1`, `rho0 = rho`).
Anneal evaluates `PairPotential::morse` (`rgpot_pair::cambridge_morse`).
ASE evaluates `ase.calculators.morse.MorsePotential` with `rcut1 = 1e6`, `rcut2 = 2e6`.
Anneal `for_cluster` uses `Config::with_scales` at `PairKind::length_scale()` plus `for_cluster` Bayes screening.
Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults (`fmax` 0.1 and 0.05).
Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55).
The wall cap for `recommended`, `ase_bh` and `ase_mh` is three times the slowest `for_cluster` wall per system.

## Summary

| system | arm | hits | median best | gap | median wall s | median inside s | median outside s | stops (hit/budget/wall) | cap s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r6 (38E, -157.477108) | for_cluster | 5/12 | -157.40690196 | 0.07020604 | 3.394487 | 2.692119 | 0.702368 | 5/7/0 |  |
| m38r6 (38E, -157.477108) | recommended | 12/12 | -157.47710811 | -0.00000011 | 0.264529 | 0.187940 | 0.080779 | 12/0/0 | 10.465089 |
| m38r6 (38E, -157.477108) | basin hopping | 0/12 | -154.01443883 | 3.46266917 | 10.465763 | 8.621287 | 1.844740 | 0/0/12 | 10.465089 |
| m38r6 (38E, -157.477108) | minima hopping | 0/12 | -153.77984449 | 3.69726351 | 10.466373 | 5.687899 | 4.778787 | 0/0/12 | 10.465089 |
| m38r14 (38D, -144.321054) | for_cluster | 0/12 | -140.31165132 | 4.00940268 | 3.505552 | 2.800238 | 0.705210 | 0/12/0 |  |
| m38r14 (38D, -144.321054) | recommended | 12/12 | -144.32105392 | 0.00000008 | 0.201223 | 0.138291 | 0.062129 | 12/0/0 | 11.142690 |
| m38r14 (38D, -144.321054) | basin hopping | 0/12 | -128.24517206 | 16.07588194 | 11.143400 | 9.191566 | 1.951881 | 0/0/12 | 11.142690 |
| m38r14 (38D, -144.321054) | minima hopping | 0/12 | -126.90309396 | 17.41796004 | 11.143864 | 6.678300 | 4.465999 | 0/0/12 | 11.142690 |
| m55r6 (55B, -250.286609) | for_cluster | 12/12 | -250.28660879 | 0.00000021 | 1.295807 | 1.128095 | 0.167712 | 12/0/0 |  |
| m55r6 (55B, -250.286609) | recommended | 12/12 | -250.28660879 | 0.00000021 | 0.285501 | 0.177955 | 0.105386 | 12/0/0 | 15.803787 |
| m55r6 (55B, -250.286609) | basin hopping | 0/12 | -238.62285608 | 11.66375292 | 15.805353 | 13.925380 | 1.879936 | 0/0/12 | 15.803787 |
| m55r6 (55B, -250.286609) | minima hopping | 0/12 | -237.60803081 | 12.67857819 | 15.806368 | 9.297536 | 6.634145 | 0/0/12 | 15.803787 |
| m55r14 (55C, -220.646208) | for_cluster | 0/12 | -218.66989772 | 1.97631028 | 16.474519 | 13.595308 | 2.957568 | 0/12/0 |  |
| m55r14 (55C, -220.646208) | recommended | 5/12 | -220.51853338 | 0.12767462 | 17.183232 | 13.410117 | 3.747864 | 5/7/0 | 54.130152 |
| m55r14 (55C, -220.646208) | basin hopping | 0/12 | -196.53148796 | 24.11472004 | 54.132262 | 47.735096 | 6.396609 | 0/0/12 | 54.130152 |
| m55r14 (55C, -220.646208) | minima hopping | 0/12 | -195.70211792 | 24.94409008 | 54.135009 | 34.419608 | 20.023580 | 0/0/12 | 54.130152 |

## Fisher and Mann-Whitney against ASE arms

| system | comparison | hits | Fisher p | U | Mann-Whitney p | wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m38r6 | for_cluster vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.324342 | 3.083165 |
| m38r6 | for_cluster vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.324323 | 3.083344 |
| m38r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.025276 | 39.563762 |
| m38r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.025274 | 39.566068 |
| m38r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.314585 | 3.178786 |
| m38r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.314572 | 3.178919 |
| m38r14 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018058 | 55.378224 |
| m38r14 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018057 | 55.380532 |
| m55r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081985 | 12.197304 |
| m55r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081980 | 12.198089 |
| m55r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018064 | 55.359963 |
| m55r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018062 | 55.363522 |
| m55r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.304338 | 3.285817 |
| m55r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.304323 | 3.285984 |
| m55r14 | recommended vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.317431 | 3.150296 |
| m55r14 | recommended vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.317414 | 3.150456 |