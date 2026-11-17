Copyright (c) anneal developers

# Non-LJ ASE head-to-head (Cambridge Morse clusters)

Four Cambridge Morse systems across two sizes and two ranges: M38 and M55 at rho = 6 and rho = 14.
Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).
code_revision 90b88235fe54abace3bde32789582eb6573ed40a.
Potential: untruncated Morse (`epsilon = 1`, `r0 = 1`, `rho0 = rho`).
Anneal evaluates `PairPotential::morse` (`rgpot_pair::cambridge_morse`).
ASE evaluates `ase.calculators.morse.MorsePotential` with `rcut1 = 1e6`, `rcut2 = 2e6`.
Anneal `for_cluster` uses `Config::with_scales` at `PairKind::length_scale()` plus `for_cluster` Bayes screening.
Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults (`fmax` 0.1 and 0.05).
Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55).
The wall cap for `recommended`, `ase_bh` and `ase_mh` is three times the slowest `for_cluster` wall per system.

## Cross-code potential agreement (seed 2000 start)

| system | N | rho | E anneal | E ASE | |dE| | max |dF| |
| --- | --- | --- | --- | --- | --- | --- |
| m38r6 | 38 | 6 | -23.2866168438 | -23.2866168438 | 1.421e-14 | 1.421e-14 |
| m38r14 | 38 | 14 | 647.2558333050 | 647.2558333050 | 1.137e-12 | 9.095e-13 |
| m55r6 | 55 | 6 | -3.7882054920 | -3.7882054920 | 5.329e-15 | 2.842e-14 |
| m55r14 | 55 | 14 | 3083.5511107014 | 3083.5511107013 | 6.821e-12 | 3.638e-12 |

## Summary

| system | arm | hits | median best | gap | median wall s | median inside s | median outside s | stops (hit/budget/wall) | cap s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r6 (38E, -157.477108) | for_cluster | 5/12 | -157.40690196 | 0.07020604 | 3.072299 | 2.439147 | 0.632570 | 5/7/0 |  |
| m38r6 (38E, -157.477108) | recommended | 12/12 | -157.47710811 | -0.00000011 | 0.253328 | 0.176140 | 0.077565 | 12/0/0 | 9.303891 |
| m38r6 (38E, -157.477108) | basin hopping | 0/12 | -154.01443883 | 3.46266917 | 9.304340 | 7.214937 | 2.089559 | 0/0/12 | 9.303891 |
| m38r6 (38E, -157.477108) | minima hopping | 0/12 | -153.77984449 | 3.69726351 | 9.305453 | 4.584193 | 4.757450 | 0/0/12 | 9.303891 |
| m38r14 (38D, -144.321054) | for_cluster | 0/12 | -140.31165132 | 4.00940268 | 3.047663 | 2.433580 | 0.612570 | 0/12/0 |  |
| m38r14 (38D, -144.321054) | recommended | 12/12 | -144.32105392 | 0.00000008 | 0.186840 | 0.130104 | 0.056788 | 12/0/0 | 9.161883 |
| m38r14 (38D, -144.321054) | basin hopping | 0/12 | -128.24517206 | 16.07588194 | 9.162566 | 7.110116 | 2.051959 | 0/0/12 | 9.161883 |
| m38r14 (38D, -144.321054) | minima hopping | 0/12 | -126.90310744 | 17.41794656 | 9.163075 | 4.939920 | 4.222710 | 0/0/12 | 9.161883 |
| m55r6 (55B, -250.286609) | for_cluster | 12/12 | -250.28660879 | 0.00000021 | 1.187034 | 1.033858 | 0.153176 | 12/0/0 |  |
| m55r6 (55B, -250.286609) | recommended | 12/12 | -250.28660879 | 0.00000021 | 0.248454 | 0.154829 | 0.093222 | 12/0/0 | 14.660946 |
| m55r6 (55B, -250.286609) | basin hopping | 0/12 | -238.62285608 | 11.66375292 | 14.662119 | 12.377780 | 2.284518 | 0/0/12 | 14.660946 |
| m55r6 (55B, -250.286609) | minima hopping | 0/12 | -237.90499068 | 12.38161832 | 14.809279 | 7.863144 | 7.334460 | 0/0/12 | 14.660946 |
| m55r14 (55C, -220.646208) | for_cluster | 0/12 | -218.66989772 | 1.97631028 | 15.380785 | 12.652870 | 2.718549 | 0/12/0 |  |
| m55r14 (55C, -220.646208) | recommended | 5/12 | -220.51853338 | 0.12767462 | 16.007071 | 12.564741 | 3.417884 | 5/7/0 | 46.785531 |
| m55r14 (55C, -220.646208) | basin hopping | 0/12 | -196.22780730 | 24.41840070 | 46.786409 | 39.517532 | 7.269090 | 0/0/12 | 46.785531 |
| m55r14 (55C, -220.646208) | minima hopping | 0/12 | -199.78820612 | 20.85800188 | 46.983019 | 23.638251 | 23.955337 | 0/0/12 | 46.785531 |

## Fisher and Mann-Whitney against ASE arms

| system | comparison | hits | Fisher p | U | Mann-Whitney p | wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m38r6 | for_cluster vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.330201 | 3.028462 |
| m38r6 | for_cluster vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.330161 | 3.028824 |
| m38r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.027227 | 36.728502 |
| m38r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.027224 | 36.732897 |
| m38r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.332621 | 3.006424 |
| m38r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.332603 | 3.006590 |
| m38r14 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.020392 | 49.039507 |
| m38r14 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.020391 | 49.042229 |
| m55r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.080959 | 12.351890 |
| m55r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.080155 | 12.475862 |
| m55r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.016945 | 59.013415 |
| m55r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.016777 | 59.605718 |
| m55r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.328745 | 3.041874 |
| m55r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.327369 | 3.054657 |
| m55r14 | recommended vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.342131 | 2.922859 |
| m55r14 | recommended vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.340699 | 2.935142 |

Medians are Hyndman-Fan type 7. Wall ratios are ratios of arm medians.
The Rust and NumPy random generators do not make a shared starting
geometry from the same numeric seed, so seed labels are not treated as pairs.
Fisher is two-sided exact. Mann-Whitney is two-sided asymptotic with
the tie correction and the continuity correction. It is not estimated
when either arm has wall-censored misses; assigning those misses the
larger paper budget would manufacture observations that were never run.
Inside is time in the potential. Outside is the rest of the wall.
