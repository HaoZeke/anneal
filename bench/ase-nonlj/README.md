Copyright (c) anneal developers

# Non-LJ ASE head-to-head (Cambridge Morse clusters)

Four Cambridge Morse systems across two sizes and two ranges: M38 and M55 at rho = 6 and rho = 14.
Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).
code_revision 5eb79cdcb0a986984dc7b487e5a3cc93224246f9.
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
| m38r6 (38E, -157.477108) | for_cluster | 5/12 | -157.40690196 | 0.07020604 | 3.097724 | 2.446911 | 0.650813 | 5/7/0 |  |
| m38r6 (38E, -157.477108) | recommended | 12/12 | -157.47710811 | -0.00000011 | 0.502146 | 0.360160 | 0.141986 | 12/0/0 | 10.185717 |
| m38r6 (38E, -157.477108) | basin hopping | 0/12 | -154.01443883 | 3.46266917 | 10.186500 | 8.407071 | 1.779809 | 0/0/12 | 10.185717 |
| m38r6 (38E, -157.477108) | minima hopping | 0/12 | -153.77984449 | 3.69726351 | 10.187733 | 5.525497 | 4.805421 | 0/0/12 | 10.185717 |
| m38r14 (38D, -144.321054) | for_cluster | 0/12 | -140.31165132 | 4.00940268 | 3.104056 | 2.466985 | 0.635762 | 0/12/0 |  |
| m38r14 (38D, -144.321054) | recommended | 9/12 | -144.32105392 | 0.00000008 | 2.136043 | 1.602577 | 0.533465 | 9/3/0 | 9.516357 |
| m38r14 (38D, -144.321054) | basin hopping | 0/12 | -128.24517206 | 16.07588194 | 9.517368 | 7.854948 | 1.662324 | 0/0/12 | 9.516357 |
| m38r14 (38D, -144.321054) | minima hopping | 0/12 | -126.90309396 | 17.41796004 | 9.517900 | 5.693238 | 3.825665 | 0/0/12 | 9.516357 |
| m55r6 (55B, -250.286609) | for_cluster | 12/12 | -250.28660879 | 0.00000021 | 1.267574 | 1.107996 | 0.161608 | 12/0/0 |  |
| m55r6 (55B, -250.286609) | recommended | 12/12 | -250.28660879 | 0.00000021 | 0.226116 | 0.167121 | 0.058996 | 12/0/0 | 15.636978 |
| m55r6 (55B, -250.286609) | basin hopping | 0/12 | -238.62285608 | 11.66375292 | 15.638122 | 13.798032 | 1.840064 | 0/0/12 | 15.636978 |
| m55r6 (55B, -250.286609) | minima hopping | 0/12 | -237.78163868 | 12.50497032 | 15.642038 | 9.161749 | 6.848896 | 0/0/12 | 15.636978 |
| m55r14 (55C, -220.646208) | for_cluster | 0/12 | -218.66989772 | 1.97631028 | 15.514131 | 12.800811 | 2.687787 | 0/12/0 |  |
| m55r14 (55C, -220.646208) | recommended | 8/12 | -220.64585570 | 0.00035230 | 11.332774 | 8.935748 | 2.294544 | 8/4/0 | 47.552319 |
| m55r14 (55C, -220.646208) | basin hopping | 0/12 | -196.22780729 | 24.41840071 | 47.553084 | 41.963894 | 5.590103 | 0/0/12 | 47.552319 |
| m55r14 (55C, -220.646208) | minima hopping | 0/12 | -200.14540501 | 20.50080299 | 47.556149 | 27.561762 | 20.625921 | 0/0/12 | 47.552319 |

## Fisher and Mann-Whitney against ASE arms

| system | comparison | hits | Fisher p | U | Mann-Whitney p | wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m38r6 | for_cluster vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.304101 | 3.288382 |
| m38r6 | for_cluster vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.304064 | 3.288780 |
| m38r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.049295 | 20.285932 |
| m38r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.049289 | 20.288388 |
| m38r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.326146 | 3.066107 |
| m38r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.326128 | 3.066278 |
| m38r14 | recommended vs basin hopping | 9/12 vs 0/12 | 0.000336519 | not estimated | not estimated | 0.224436 | 4.455608 |
| m38r14 | recommended vs minima hopping | 9/12 vs 0/12 | 0.000336519 | not estimated | not estimated | 0.224424 | 4.455857 |
| m55r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081057 | 12.337054 |
| m55r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081036 | 12.340143 |
| m55r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.014459 | 69.159581 |
| m55r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.014456 | 69.176898 |
| m55r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.326249 | 3.065146 |
| m55r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.326228 | 3.065344 |
| m55r14 | recommended vs basin hopping | 8/12 vs 0/12 | 0.00134608 | not estimated | not estimated | 0.238318 | 4.196067 |
| m55r14 | recommended vs minima hopping | 8/12 vs 0/12 | 0.00134608 | not estimated | not estimated | 0.238303 | 4.196338 |

Medians are Hyndman-Fan type 7. Wall ratios are ratios of arm medians.
The Rust and NumPy random generators do not make a shared starting
geometry from the same numeric seed, so seed labels are not treated as pairs.
Fisher is two-sided exact. Mann-Whitney is two-sided asymptotic with
the tie correction and the continuity correction. It is not estimated
when either arm has wall-censored misses; assigning those misses the
larger paper budget would manufacture observations that were never run.
Inside is time in the potential. Outside is the rest of the wall.
