# no-size-constants r5 no-loss gate (part B)

**Verdict: PASS**

## Provenance and Protocol

- Arm A: `commit:61fa67cf380e46876b569da45e7cc091647a775b` (tree `cff796c507a8fa1d0bfb28888c3570b61988055e`)
- Arm B: `tree:64b12739c8707669e6ae203d8f8b99a160609b55` (tree `64b12739c8707669e6ae203d8f8b99a160609b55`)
- Sessions: 1 (first `20261120T043734Z`)
- Machine ID (SHA-256): `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- CPU: `Intel(R) Xeon(R) Processor` (`x86_64`, kernel `6.12.113+`, affinity `0`)
- Toolchains: `rustc 1.88.0 (6b00bc388 2025-06-23)`, `cargo 1.88.0 (873a06493 2025-05-10)`, Python `3.12.3`
- Seeds: `2000-2011` (n=12 per system, `PAPER_BUDGETS`, `H2H_BUDGET` unset)
- Order: `ABBA_balanced_per_seed`; `(slot, arm)` counts across 720 rows: slot 1 arm A=180, slot 1 arm B=180, slot 2 arm A=180, slot 2 arm B=180
- Provenance verification (`git cat-file -e`): 9/9 recorded library and harness blob ids resolved.
- The recorded harness blobs are the pre-strip harness and the only difference is the removed seat field.

### Release binary digests

| arm | binary | sha256 |
| --- | --- | --- |
| A | `h2h_sc` | `2f6f429b5af8d4ad972fec042eea807891990b8ff30bde21e6f5d3d29ae0a14c` |
| A | `h2h_seed` | `a90c449c3c9a19ac517a2adbc80c5673656330b4102f2c358006f0530958e76c` |
| A | `nsc-r5-gate-driver` | `bb19c02528cd92a42bcb3751e3d592545c9072ba76c114faa458b52bd7773352` |
| B | `h2h_sc` | `3af401b57e4839dc2968737210b2ad15d9c83635a365fb80d0a8f513cb52284b` |
| B | `h2h_seed` | `1f34df30451e4b8fd7ca8ed9704db144e663f34754b26f6ff27a6fcd1ae91089` |
| B | `nsc-r5-gate-driver` | `3f32d86175f18af1661e54382acd49a452f072f0afb2f36a9e55abbc42f92cc6` |

## Panels

| panel | systems | hits A | hits B | CMH exact Fisher p | van Elteren p (calls) | calls direction | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | outside/call med A (us) | outside/call med B (us) | outside/call med B/A | same | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sutton-Chen | 9 | 75/108 | 75/108 | 1 | 1 | equal | 4771.997 | 4785.392 | 4400.188 | 4412.006 | 371.809 | 373.387 | 4.4865 | 4.4980 | 1.0026 | 108/108 | PASS |
| BLJ | 7 | 46/84 | 46/84 | 1 | 1 | equal | 2852.536 | 2843.423 | 1513.726 | 1518.670 | 1338.810 | 1324.754 | 10.6532 | 10.3610 | 0.9726 | 84/84 | PASS |
| water/TIP4P rigid | 14 | 113/168 | 113/168 | 1 | 1 | equal | 1289.015 | 1285.411 | 976.868 | 973.717 | 312.147 | 311.694 | 1.9247 | 1.9123 | 0.9936 | 168/168 | PASS |

Multi-system panels report the two-sided stratified exact Fisher (CMH exact) p on hits and the two-sided tie-corrected van Elteren test on calls to hit (misses entered at budget). Wall, inside, and outside are panel sums in seconds; outside/call med is the Hyndman-Fan type 7 median of per-run `outside_s / charged` in microseconds.

## Systems

| panel | system | budget | hits A | hits B | Fisher p | calls A | calls B | MW p calls | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | out/call med A (us) | out/call med B (us) | out/call med B/A | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sutton-Chen | sc126-38 | 400000 | 12/12 | 12/12 | 1 | 113504 | 113504 | 1 | 4.091481 | 4.114253 | 3.867592 | 3.891406 | 0.237193 | 0.236275 | 2.1856 | 2.1789 | 0.9970 | PASS |
| Sutton-Chen | sc108-38 | 400000 | 7/12 | 7/12 | 1 | 361032 | 361032 | 1 | 13.711373 | 13.719442 | 12.710579 | 12.727054 | 0.988800 | 0.983494 | 2.6781 | 2.6647 | 0.9950 | PASS |
| Sutton-Chen | sc96-38 | 400000 | 12/12 | 12/12 | 1 | 50230 | 50230 | 1 | 1.897470 | 1.896504 | 1.780136 | 1.780039 | 0.117334 | 0.116465 | 2.2332 | 2.2176 | 0.9930 | PASS |
| Sutton-Chen | sc126-55 | 1000000 | 12/12 | 12/12 | 1 | 18379.5 | 18379.5 | 1 | 1.175609 | 1.175912 | 1.103285 | 1.103823 | 0.074046 | 0.074069 | 4.2440 | 4.2334 | 0.9975 | PASS |
| Sutton-Chen | sc108-55 | 1000000 | 5/12 | 5/12 | 1 | 1e+06 | 1e+06 | 1 | 66.697885 | 67.177345 | 61.890687 | 62.526037 | 4.615952 | 4.585768 | 5.1523 | 5.1672 | 1.0029 | PASS |
| Sutton-Chen | sc96-55 | 1000000 | 12/12 | 12/12 | 1 | 23157 | 23157 | 1 | 1.544558 | 1.544778 | 1.451016 | 1.451442 | 0.090205 | 0.089966 | 3.9143 | 3.9035 | 0.9972 | PASS |
| Sutton-Chen | sc126-80 | 1000000 | 5/12 | 5/12 | 1 | 1e+06 | 1e+06 | 1 | 130.425640 | 130.386571 | 119.530552 | 119.423995 | 10.717703 | 10.788771 | 10.8514 | 10.9191 | 1.0062 | PASS |
| Sutton-Chen | sc108-80 | 1000000 | 3/12 | 3/12 | 1 | 1e+06 | 1e+06 | 1 | 133.826354 | 134.457108 | 123.668909 | 124.760910 | 10.259197 | 10.102906 | 10.2592 | 10.1029 | 0.9848 | PASS |
| Sutton-Chen | sc96-80 | 1000000 | 7/12 | 7/12 | 1 | 989848 | 989848 | 1 | 132.001683 | 131.993775 | 122.111118 | 122.098772 | 9.656866 | 9.640684 | 9.7227 | 9.7169 | 0.9994 | PASS |
| BLJ | blj105n38 | 400000 | 2/12 | 2/12 | 1 | 400000 | 400000 | 1 | 5.570992 | 5.522501 | 1.623533 | 1.624002 | 3.944770 | 3.899003 | 9.8619 | 9.7475 | 0.9884 | PASS |
| BLJ | blj115n38 | 400000 | 12/12 | 12/12 | 1 | 82300.5 | 82300.5 | 1 | 0.835282 | 0.844203 | 0.361533 | 0.362366 | 0.303039 | 0.291385 | 2.4215 | 2.3493 | 0.9702 | PASS |
| BLJ | blj130n38 | 400000 | 11/12 | 11/12 | 1 | 134072 | 134072 | 1 | 1.111226 | 1.113634 | 0.542671 | 0.542882 | 0.670087 | 0.673259 | 6.4727 | 6.5053 | 1.0050 | PASS |
| BLJ | blj105n55 | 1000000 | 12/12 | 12/12 | 1 | 158055 | 158055 | 1 | 2.205353 | 2.208024 | 1.423716 | 1.426711 | 0.626707 | 0.634814 | 3.1681 | 3.2029 | 1.0110 | PASS |
| BLJ | blj115n55 | 1000000 | 0/12 | 0/12 | 1 | 1e+06 | 1e+06 | 1 | 33.077240 | 32.730833 | 8.495566 | 8.509076 | 24.583788 | 24.220941 | 24.5838 | 24.2209 | 0.9852 | PASS |
| BLJ | blj130n55 | 1000000 | 9/12 | 9/12 | 1 | 633042 | 633042 | 1 | 14.821162 | 14.844964 | 5.437327 | 5.471766 | 8.833642 | 8.737661 | 17.3303 | 17.3988 | 1.0040 | PASS |
| BLJ | blj130n98 | 4000000 | 0/12 | 0/12 | 1 | 4e+06 | 4e+06 | 1 | 164.061825 | 164.107145 | 107.777072 | 108.191817 | 56.295044 | 55.930073 | 14.0738 | 13.9825 | 0.9935 | PASS |
| water/TIP4P rigid | tip4p8 | 2000000 | 12/12 | 12/12 | 1 | 6339 | 6339 | 1 | 0.019731 | 0.019886 | 0.011936 | 0.011911 | 0.007796 | 0.008774 | 1.2190 | 1.2378 | 1.0154 | PASS |
| water/TIP4P rigid | tip4p9 | 2000000 | 12/12 | 12/12 | 1 | 5839.5 | 5839.5 | 1 | 0.020656 | 0.020567 | 0.013144 | 0.013111 | 0.007466 | 0.007452 | 1.2797 | 1.2872 | 1.0059 | PASS |
| water/TIP4P rigid | tip4p10 | 2000000 | 12/12 | 12/12 | 1 | 13138.5 | 13138.5 | 1 | 0.053700 | 0.053240 | 0.035034 | 0.034692 | 0.018665 | 0.018548 | 1.4118 | 1.4129 | 1.0008 | PASS |
| water/TIP4P rigid | tip4p11 | 2000000 | 12/12 | 12/12 | 1 | 109288 | 109288 | 1 | 0.505894 | 0.508457 | 0.342697 | 0.346089 | 0.163197 | 0.162368 | 1.5064 | 1.5022 | 0.9972 | PASS |
| water/TIP4P rigid | tip4p12 | 2000000 | 12/12 | 12/12 | 1 | 41418.5 | 41418.5 | 1 | 0.219714 | 0.218459 | 0.149404 | 0.148665 | 0.070309 | 0.069795 | 1.7281 | 1.7213 | 0.9961 | PASS |
| water/TIP4P rigid | tip4p13 | 2000000 | 12/12 | 12/12 | 1 | 216690 | 216690 | 1 | 1.291598 | 1.290298 | 0.890887 | 0.889084 | 0.400711 | 0.401214 | 1.8438 | 1.8300 | 0.9925 | PASS |
| water/TIP4P rigid | tip4p14 | 2000000 | 12/12 | 12/12 | 1 | 135853 | 135853 | 1 | 0.897599 | 0.894000 | 0.632871 | 0.630623 | 0.264728 | 0.263377 | 1.9761 | 1.9687 | 0.9963 | PASS |
| water/TIP4P rigid | tip4p15 | 2000000 | 7/12 | 7/12 | 1 | 1.48162e+06 | 1.48162e+06 | 1 | 10.449780 | 10.417092 | 7.747695 | 7.727701 | 2.702085 | 2.689392 | 1.8734 | 1.8614 | 0.9936 | PASS |
| water/TIP4P rigid | tip4p16 | 2000000 | 2/12 | 2/12 | 1 | 2e+06 | 2e+06 | 1 | 15.836004 | 15.798205 | 11.712910 | 11.677058 | 4.122199 | 4.123818 | 2.0863 | 2.0910 | 1.0022 | PASS |
| water/TIP4P rigid | tip4p17 | 2000000 | 5/12 | 5/12 | 1 | 2e+06 | 2e+06 | 1 | 16.978061 | 16.915425 | 12.997983 | 12.965091 | 3.980077 | 3.941591 | 2.1532 | 2.1452 | 0.9963 | PASS |
| water/TIP4P rigid | tip4p18 | 2000000 | 2/12 | 2/12 | 1 | 2e+06 | 2e+06 | 1 | 18.689296 | 18.663416 | 14.397544 | 14.348180 | 4.295225 | 4.308514 | 2.1598 | 2.1579 | 0.9991 | PASS |
| water/TIP4P rigid | tip4p19 | 2000000 | 7/12 | 7/12 | 1 | 1.06084e+06 | 1.06084e+06 | 1 | 10.762706 | 10.734412 | 8.397983 | 8.379608 | 2.376078 | 2.369046 | 2.4591 | 2.4643 | 1.0021 | PASS |
| water/TIP4P rigid | tip4p20 | 2000000 | 6/12 | 6/12 | 1 | 1.68625e+06 | 1.68625e+06 | 1 | 18.742952 | 18.630723 | 14.693010 | 14.582293 | 4.049942 | 4.048430 | 2.7130 | 2.6921 | 0.9923 | PASS |
| water/TIP4P rigid | tip4p21 | 2000000 | 0/12 | 0/12 | 1 | 2e+06 | 2e+06 | 1 | 24.803472 | 24.833251 | 18.947213 | 18.908351 | 5.855935 | 5.901727 | 2.9280 | 2.9509 | 1.0078 | PASS |

## Hit Status Changes

- `A hit -> B miss`: 0
- `A miss -> B hit`: 0

No seed changed hit status in either direction.

## TIP4P bit-identity

For every rigid TIP4P `(H2O)n` size (`n = 8..21`) and seed (`2000-2011`), both arms start from identical coordinates and compare `.trace.csv` bytes, `.xyz`/`.gmin` geometry bytes, best energy value/bits, and call counts (`evals_to_hit` and `charged`).

| system | trajectory byte-identical | best energy identical | calls identical | all bit-identical |
| --- | --- | --- | --- | --- |
| tip4p8 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p9 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p10 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p11 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p12 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p13 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p14 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p15 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p16 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p17 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p18 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p19 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p20 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p21 | 12/12 | 12/12 | 12/12 | 12/12 |

### TIP4P per-seed bit-identity matrix (`Y` = trajectory, best energy, and calls all byte-identical; `D` = differs)

| system | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p9 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p10 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p11 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p12 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p13 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p14 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p15 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p16 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p17 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p18 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p19 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p20 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p21 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |

Total TIP4P pairs all byte-identical: 168/168.

## Gate Rule

FIX if any cell loses hits (B below A), or loses on calls at `p < 0.05` (two-sided tie-corrected Mann-Whitney), or has outside-potential time per call ratio (B/A ratio of medians) above `1.10` (or any B-only failure), naming the cells; otherwise PASS.
