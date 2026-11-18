Copyright (c) anneal developers

# Non-LJ ASE head-to-head (Cambridge Morse clusters)

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


Twelve Cambridge Morse systems: M38 and M55 at rho = 3, 6, 10 and 14, plus M75 and M80 at rho = 6 and 14.
Anneal `for_cluster` uses `Config::for_cluster_with_scales` at `PairKind::length_scale()` (earlier M38/M55 rho 6 and 14 rows retain their committed runs).
Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55, 4,000,000 at 75 and 80).
Additional code_revision values 2d0af145a04c19eeed01128ece49576db419e4d8 and 90b88235fe54abace3bde32789582eb6573ed40a cover the widened panel.

## Cross-code potential agreement (seed 2000 start)

| system | N | rho | E anneal | E ASE | |dE| | max |dF| |
| --- | --- | --- | --- | --- | --- | --- |
| m38r3 | 38 | 3 | -74.3990088498 | -74.3990088498 | 7.105e-14 | 2.665e-15 |
| m38r6 | 38 | 6 | -23.2866168438 | -23.2866168438 | 1.421e-14 | 1.421e-14 |
| m38r10 | 38 | 10 | 97.1815985541 | 97.1815985541 | 8.527e-14 | 2.274e-13 |
| m38r14 | 38 | 14 | 647.2558333050 | 647.2558333050 | 1.137e-12 | 9.095e-13 |
| m55r3 | 55 | 3 | -110.9398776959 | -110.9398776959 | 8.527e-14 | 4.441e-15 |
| m55r6 | 55 | 6 | -3.7882054920 | -3.7882054920 | 5.329e-15 | 2.842e-14 |
| m55r10 | 55 | 10 | 423.3159252828 | 423.3159252828 | 3.979e-13 | 4.547e-13 |
| m55r14 | 55 | 14 | 3083.5511107014 | 3083.5511107013 | 6.821e-12 | 3.638e-12 |
| m75r6 | 75 | 6 | -15.4214418034 | -15.4214418034 | 3.375e-14 | 3.553e-14 |
| m75r14 | 75 | 14 | 3033.8763401766 | 3033.8763401766 | 5.002e-12 | 1.819e-12 |
| m80r6 | 80 | 6 | -43.9930393173 | -43.9930393173 | 6.395e-14 | 1.421e-14 |
| m80r14 | 80 | 14 | 1777.2438323579 | 1777.2438323579 | 5.230e-12 | 1.364e-12 |
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


## Widened twelve-system summary

| system | arm | hits | median best | gap | median wall s | median inside s | median outside s | stops (hit/budget/wall) | cap s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r3 (38A, -249.159174) | for_cluster | 12/12 | -249.15917378 | 0.00000022 | 0.003353 | 0.002714 | 0.000686 | 12/0/0 |  |
| m38r3 (38A, -249.159174) | recommended | 12/12 | -249.15917378 | 0.00000022 | 0.007783 | 0.004637 | 0.002018 | 12/0/0 | 0.070920 |
| m38r3 (38A, -249.159174) | basin hopping | 0/12 | -243.77682340 | 5.38235060 | 0.071835 | 0.055396 | 0.016321 | 0/0/12 | 0.070920 |
| m38r3 (38A, -249.159174) | minima hopping | 0/12 | -239.46250287 | 9.69667113 | 0.071787 | 0.035616 | 0.035852 | 0/0/12 | 0.070920 |
| m38r6 (38E, -157.477108) | for_cluster | 5/12 | -157.40690196 | 0.07020604 | 3.394487 | 2.692119 | 0.702368 | 5/7/0 |  |
| m38r6 (38E, -157.477108) | recommended | 12/12 | -157.47710811 | -0.00000011 | 0.264529 | 0.187940 | 0.080779 | 12/0/0 | 10.465089 |
| m38r6 (38E, -157.477108) | basin hopping | 0/12 | -154.01443883 | 3.46266917 | 10.465763 | 8.621287 | 1.844740 | 0/0/12 | 10.465089 |
| m38r6 (38E, -157.477108) | minima hopping | 0/12 | -153.77984449 | 3.69726351 | 10.466373 | 5.687899 | 4.778787 | 0/0/12 | 10.465089 |
| m38r10 (38D, -145.849817) | for_cluster | 0/12 | -143.85477593 | 1.99504107 | 3.035279 | 2.415945 | 0.616753 | 0/12/0 |  |
| m38r10 (38D, -145.849817) | recommended | 12/12 | -145.84981727 | -0.00000027 | 0.116921 | 0.082227 | 0.035967 | 12/0/0 | 9.165219 |
| m38r10 (38D, -145.849817) | basin hopping | 0/12 | -136.00687063 | 9.84294637 | 9.166013 | 7.111828 | 2.054105 | 0/0/12 | 9.165219 |
| m38r10 (38D, -145.849817) | minima hopping | 0/12 | -135.11750639 | 10.73231061 | 9.166410 | 4.885374 | 4.281400 | 0/0/12 | 9.165219 |
| m38r14 (38D, -144.321054) | for_cluster | 0/12 | -140.31165132 | 4.00940268 | 3.505552 | 2.800238 | 0.705210 | 0/12/0 |  |
| m38r14 (38D, -144.321054) | recommended | 12/12 | -144.32105392 | 0.00000008 | 0.201223 | 0.138291 | 0.062129 | 12/0/0 | 11.142690 |
| m38r14 (38D, -144.321054) | basin hopping | 0/12 | -128.24517206 | 16.07588194 | 11.143400 | 9.191566 | 1.951881 | 0/0/12 | 11.142690 |
| m38r14 (38D, -144.321054) | minima hopping | 0/12 | -126.90309396 | 17.41796004 | 11.143864 | 6.678300 | 4.465999 | 0/0/12 | 11.142690 |
| m55r3 (55A, -417.918562) | for_cluster | 12/12 | -417.91856235 | -0.00000035 | 0.041023 | 0.035868 | 0.005155 | 12/0/0 |  |
| m55r3 (55A, -417.918562) | recommended | 11/12 | -417.91856235 | -0.00000035 | 0.037244 | 0.025684 | 0.014740 | 11/0/1 | 0.304524 |
| m55r3 (55A, -417.918562) | basin hopping | 0/12 | -414.49228439 | 3.42627762 | 0.305060 | 0.257515 | 0.047610 | 0/0/12 | 0.304524 |
| m55r3 (55A, -417.918562) | minima hopping | 0/12 | -413.54756073 | 4.37100127 | 0.306121 | 0.182675 | 0.123490 | 0/0/12 | 0.304524 |
| m55r6 (55B, -250.286609) | for_cluster | 12/12 | -250.28660879 | 0.00000021 | 1.295807 | 1.128095 | 0.167712 | 12/0/0 |  |
| m55r6 (55B, -250.286609) | recommended | 12/12 | -250.28660879 | 0.00000021 | 0.285501 | 0.177955 | 0.105386 | 12/0/0 | 15.803787 |
| m55r6 (55B, -250.286609) | basin hopping | 0/12 | -238.62285608 | 11.66375292 | 15.805353 | 13.925380 | 1.879936 | 0/0/12 | 15.803787 |
| m55r6 (55B, -250.286609) | minima hopping | 0/12 | -237.60803081 | 12.67857819 | 15.806368 | 9.297536 | 6.634145 | 0/0/12 | 15.803787 |
| m55r10 (55B, -225.814286) | for_cluster | 9/12 | -225.81428617 | -0.00000017 | 7.994217 | 6.670729 | 1.323488 | 9/3/0 |  |
| m55r10 (55B, -225.814286) | recommended | 12/12 | -225.81428617 | -0.00000017 | 1.694541 | 1.313819 | 0.380723 | 12/0/0 | 47.103933 |
| m55r10 (55B, -225.814286) | basin hopping | 0/12 | -209.62941423 | 16.18487178 | 47.105200 | 39.703813 | 7.400633 | 0/0/12 | 47.103933 |
| m55r10 (55B, -225.814286) | minima hopping | 0/12 | -208.64043202 | 17.17385398 | 47.148128 | 21.869480 | 25.757234 | 0/0/12 | 47.103933 |
| m55r14 (55C, -220.646208) | for_cluster | 0/12 | -218.66989772 | 1.97631028 | 16.474519 | 13.595308 | 2.957568 | 0/12/0 |  |
| m55r14 (55C, -220.646208) | recommended | 5/12 | -220.51853338 | 0.12767462 | 17.183232 | 13.410117 | 3.747864 | 5/7/0 | 54.130152 |
| m55r14 (55C, -220.646208) | basin hopping | 0/12 | -196.53148796 | 24.11472004 | 54.132262 | 47.735096 | 6.396609 | 0/0/12 | 54.130152 |
| m55r14 (55C, -220.646208) | minima hopping | 0/12 | -195.70211792 | 24.94409008 | 54.135009 | 34.419608 | 20.023580 | 0/0/12 | 54.130152 |
| m75r6 (75C, -351.472365) | for_cluster | 0/12 | -351.17704149 | 0.29532351 | 110.943570 | 93.121109 | 17.829197 | 0/12/0 |  |
| m75r6 (75C, -351.472365) | recommended | 11/12 | -351.47236546 | -0.00000046 | 24.802750 | 17.934775 | 6.867975 | 11/1/0 | 335.075712 |
| m75r6 (75C, -351.472365) | basin hopping | 0/12 | -343.03754528 | 8.43481972 | 335.077890 | 298.811159 | 36.266754 | 0/0/12 | 335.075712 |
| m75r6 (75C, -351.472365) | minima hopping | 0/12 | -343.73299875 | 7.73936625 | 336.492814 | 63.079442 | 276.915143 | 0/0/12 | 335.075712 |
| m75r14 (75C, -318.407330) | for_cluster | 1/12 | -315.77794494 | 2.62938506 | 115.141258 | 93.197923 | 21.905860 | 1/11/0 |  |
| m75r14 (75C, -318.407330) | recommended | 2/12 | -316.75965992 | 1.64767008 | 119.654969 | 92.463527 | 26.967623 | 2/10/0 | 352.064352 |
| m75r14 (75C, -318.407330) | basin hopping | 0/12 | -278.85585451 | 39.55147549 | 352.066569 | 313.869332 | 38.197577 | 0/0/12 | 352.064352 |
| m75r14 (75C, -318.407330) | minima hopping | 0/12 | -291.33770019 | 27.06962981 | 357.845411 | 98.718634 | 256.310218 | 0/0/12 | 352.064352 |
| m80r6 (80B, -378.333471) | for_cluster | 12/12 | -378.33347107 | -0.00000007 | 4.915536 | 4.361016 | 0.554520 | 12/0/0 |  |
| m80r6 (80B, -378.333471) | recommended | 6/12 | -378.15298417 | 0.18048683 | 105.723067 | 80.116942 | 25.209035 | 6/0/6 | 107.045721 |
| m80r6 (80B, -378.333471) | basin hopping | 0/12 | -366.39014441 | 11.94332659 | 107.046650 | 96.400219 | 10.647134 | 0/0/12 | 107.045721 |
| m80r6 (80B, -378.333471) | minima hopping | 0/12 | -366.74898775 | 11.58448325 | 108.608386 | 32.070216 | 79.553045 | 0/0/12 | 107.045721 |
| m80r14 (80G, -340.811371) | for_cluster | 1/12 | -338.84138377 | 1.96998723 | 129.243885 | 106.047811 | 23.148536 | 1/11/0 |  |
| m80r14 (80G, -340.811371) | recommended | 10/12 | -340.81137066 | 0.00000034 | 85.720585 | 66.391192 | 19.329392 | 10/2/0 | 390.960969 |
| m80r14 (80G, -340.811371) | basin hopping | 0/12 | -298.25214857 | 42.55922243 | 390.962527 | 351.511211 | 39.451886 | 0/0/12 | 390.960969 |
| m80r14 (80G, -340.811371) | minima hopping | 0/12 | -311.96694128 | 28.84442972 | 394.536203 | 109.153984 | 286.208670 | 0/0/12 | 390.960969 |
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

## Widened twelve-system Fisher and Mann-Whitney against ASE arms

| system | comparison | hits | Fisher p | U | Mann-Whitney p | wall ratio | inverse ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m38r3 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.046676 | 21.424098 |
| m38r3 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.046708 | 21.409633 |
| m38r3 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.108346 | 9.229731 |
| m38r3 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.108419 | 9.223500 |
| m38r6 | for_cluster vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.324342 | 3.083165 |
| m38r6 | for_cluster vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.324323 | 3.083344 |
| m38r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.025276 | 39.563762 |
| m38r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.025274 | 39.566068 |
| m38r10 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.331145 | 3.019826 |
| m38r10 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.331131 | 3.019956 |
| m38r10 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.012756 | 78.395260 |
| m38r10 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.012755 | 78.398655 |
| m38r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.314585 | 3.178786 |
| m38r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.314572 | 3.178919 |
| m38r14 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018058 | 55.378224 |
| m38r14 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018057 | 55.380532 |
| m55r3 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.134475 | 7.436316 |
| m55r3 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.134009 | 7.462180 |
| m55r3 | recommended vs basin hopping | 11/12 vs 0/12 | 9.61483e-06 | not estimated | not estimated | 0.122089 | 8.190740 |
| m55r3 | recommended vs minima hopping | 11/12 vs 0/12 | 9.61483e-06 | not estimated | not estimated | 0.121666 | 8.219227 |
| m55r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081985 | 12.197304 |
| m55r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.081980 | 12.198089 |
| m55r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018064 | 55.359963 |
| m55r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.018062 | 55.363522 |
| m55r10 | for_cluster vs basin hopping | 9/12 vs 0/12 | 0.000336519 | not estimated | not estimated | 0.169710 | 5.892410 |
| m55r10 | for_cluster vs minima hopping | 9/12 vs 0/12 | 0.000336519 | not estimated | not estimated | 0.169555 | 5.897779 |
| m55r10 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.035974 | 27.798206 |
| m55r10 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.035941 | 27.823539 |
| m55r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.304338 | 3.285817 |
| m55r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.304323 | 3.285984 |
| m55r14 | recommended vs basin hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.317431 | 3.150296 |
| m55r14 | recommended vs minima hopping | 5/12 vs 0/12 | 0.0372671 | not estimated | not estimated | 0.317414 | 3.150456 |
| m75r6 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.331098 | 3.020255 |
| m75r6 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | not estimated | not estimated | 0.329706 | 3.033009 |
| m75r6 | recommended vs basin hopping | 11/12 vs 0/12 | 9.61483e-06 | not estimated | not estimated | 0.074021 | 13.509707 |
| m75r6 | recommended vs minima hopping | 11/12 vs 0/12 | 9.61483e-06 | not estimated | not estimated | 0.073710 | 13.566754 |
| m75r14 | for_cluster vs basin hopping | 1/12 vs 0/12 | 1 | not estimated | not estimated | 0.327044 | 3.057693 |
| m75r14 | for_cluster vs minima hopping | 1/12 vs 0/12 | 1 | not estimated | not estimated | 0.321763 | 3.107882 |
| m75r14 | recommended vs basin hopping | 2/12 vs 0/12 | 0.478261 | not estimated | not estimated | 0.339865 | 2.942348 |
| m75r14 | recommended vs minima hopping | 2/12 vs 0/12 | 0.478261 | not estimated | not estimated | 0.334376 | 2.990644 |
| m80r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.045920 | 21.777208 |
| m80r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | not estimated | not estimated | 0.045259 | 22.094922 |
| m80r6 | recommended vs basin hopping | 6/12 vs 0/12 | 0.01373 | not estimated | not estimated | 0.987635 | 1.012519 |
| m80r6 | recommended vs minima hopping | 6/12 vs 0/12 | 0.01373 | not estimated | not estimated | 0.973434 | 1.027291 |
| m80r14 | for_cluster vs basin hopping | 1/12 vs 0/12 | 1 | not estimated | not estimated | 0.330579 | 3.024998 |
| m80r14 | for_cluster vs minima hopping | 1/12 vs 0/12 | 1 | not estimated | not estimated | 0.327584 | 3.052649 |
| m80r14 | recommended vs basin hopping | 10/12 vs 0/12 | 6.73038e-05 | not estimated | not estimated | 0.219255 | 4.560894 |
| m80r14 | recommended vs minima hopping | 10/12 vs 0/12 | 6.73038e-05 | not estimated | not estimated | 0.217269 | 4.602584 |

Medians are Hyndman-Fan type 7. Wall ratios are ratios of arm medians.
The Rust and NumPy random generators do not make a shared starting
geometry from the same numeric seed, so seed labels are not treated as pairs.
Fisher is two-sided exact. Mann-Whitney is two-sided asymptotic with
the tie correction and the continuity correction. It is not estimated
when either arm has wall-censored misses; assigning those misses the
larger paper budget would manufacture observations that were never run.
Inside is time in the potential. Outside is the rest of the wall.
