# PR 168 walklen3 gate, shard B, out of sample

Measured base `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec`, tree `e5ebda63f9f0117226d50c13abf33243ccf452cc`.
M is scratch merge `8824564b2479f0917c1e4927cd5ac938a86c2ac0`, tree `991d5afc6d7b7f7e6c392cb68aa351af1fc50179`.
`git merge-tree --write-tree` of the base and payload `fd5449766b4f1f693de2245933afa609a93a5c6b`
(tree `491819895e46a8fccbfc2449c18868780d1a090b`) exited 0 and wrote `7f2625597a167d3c7841b2266eec844997e53266`.
O is scratch merge `c96807c54e8cc602322255c75be61425c7bf722a` of that tree. Neither scratch merge is in the result history.

Seeds 2000-2011 stay in the in-sample A-versus-M record and are not inputs here.
This file uses seeds 2012-2119 for lj75-seed and seeds 2012-2023 for the other cells.
lj75-seed runs first. The arm order repeats AMO, OMA, MAO, OAM, MOA, AOM.

Bootstrap: 10000 paired resamples, seed 20261123, type-7 percentiles.
A fresh generator is used for each interval. A miss counts as budget plus one.
The ratio is the median of the first arm divided by the median of the second.
The bar is no hit loss, no calls-to-hit loss, and an upper interval at most 1.10
on outside time and on RSS. A hit loss is real when the one-sided Fisher
probability is below 0.05. A calls, outside, or RSS loss is real only when
its lower interval is above 1. This round has no second build of the base.

| cell | arm | hits | Wilson 95% | median calls | median outside | median RSS |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| lj75-seed | A | 107/108 | [0.9494, 0.9984] | 17926.0 | 0.195193 | 2808.0 |
| lj75-seed | M | 107/108 | [0.9494, 0.9984] | 23481.0 | 0.211857 | 2222.0 |
| lj75-seed | O | 105/108 | [0.9215, 0.9905] | 17926.0 | 0.198979 | 2160.0 |
| ase-lj98 | A | 9/12 | [0.4677, 0.9111] | 2135674.0 | 18.686910 | 92672.0 |
| ase-lj98 | M | 9/12 | [0.4677, 0.9111] | 2135674.0 | 18.529694 | 92636.0 |
| ase-lj98 | O | 9/12 | [0.4677, 0.9111] | 2135674.0 | 18.701929 | 92632.0 |
| ase-lj104 | A | 6/12 | [0.2538, 0.7462] | 3296794.5 | 33.702058 | 187940.0 |
| ase-lj104 | M | 6/12 | [0.2538, 0.7462] | 3296794.5 | 33.123193 | 187948.0 |
| ase-lj104 | O | 6/12 | [0.2538, 0.7462] | 3296794.5 | 34.174226 | 187948.0 |
| au55-seed | A | 4/12 | [0.1381, 0.6094] | 1000001.0 | 8.673059 | 123042.0 |
| au55-seed | M | 4/12 | [0.1381, 0.6094] | 1000001.0 | 8.509157 | 63972.0 |
| au55-seed | O | 4/12 | [0.1381, 0.6094] | 1000001.0 | 8.646138 | 63972.0 |
| lj38 | A | 12/12 | [0.7575, 1.0000] | 1803.5 | 0.006928 | 778.0 |
| lj38 | M | 12/12 | [0.7575, 1.0000] | 4426.0 | 0.015003 | 944.0 |
| lj38 | O | 12/12 | [0.7575, 1.0000] | 1803.5 | 0.007098 | 740.0 |
| m38r6 | A | 12/12 | [0.7575, 1.0000] | 43237.0 | 0.106918 | 1898.0 |
| m38r6 | M | 12/12 | [0.7575, 1.0000] | 43237.0 | 0.107234 | 1228.0 |
| m38r6 | O | 12/12 | [0.7575, 1.0000] | 43237.0 | 0.106723 | 1228.0 |
| m55r14 | A | 9/12 | [0.4677, 0.9111] | 579153.0 | 2.420077 | 17708.0 |
| m55r14 | M | 9/12 | [0.4677, 0.9111] | 579153.0 | 2.409893 | 7396.0 |
| m55r14 | O | 9/12 | [0.4677, 0.9111] | 579153.0 | 2.368501 | 7392.0 |

| cell | pair | Fisher p | MWU calls | MWU outside | MWU RSS | calls ratio | outside ratio | RSS ratio | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| lj75-seed | O/M | 0.3107 | 0.8744 | 0.7574 | 0.7621 | 0.7634 [0.6370, 1.1019] | 0.9392 [0.7114, 1.1778] | 0.9721 [0.9155, 1.0337] | UNRESOLVED |
| lj75-seed | O/A | 0.3107 | 0.497 | 0.4476 | 1 | 1.0000 [1.0000, 1.0000] | 1.0194 [0.9632, 1.0528] | 0.7692 [0.6871, 0.8502] | UNRESOLVED |
| lj75-seed | M/A | 0.7512 | 0.1256 | 0.2279 | 1 | 1.3099 [0.9075, 1.5699] | 1.0854 [0.8555, 1.4147] | 0.7913 [0.6812, 0.8875] | UNRESOLVED |
| ase-lj98 | O/M | 0.6798 | 0.5116 | 0.4199 | 0.5802 | 1.0000 [1.0000, 1.0000] | 1.0093 [0.9601, 1.0562] | 1.0000 [0.9999, 1.0000] | PASS |
| ase-lj98 | O/A | 0.6798 | 0.5116 | 0.4885 | 0.523 | 1.0000 [1.0000, 1.0000] | 1.0008 [0.9766, 1.0551] | 0.9996 [0.9995, 1.0013] | PASS |
| ase-lj98 | M/A | 0.6798 | 0.5116 | 0.5345 | 0.523 | 1.0000 [1.0000, 1.0000] | 0.9916 [0.9829, 1.0268] | 0.9996 [0.9996, 1.0013] | PASS |
| ase-lj104 | O/M | 0.6579 | 0.5123 | 0.4655 | 0.5 | 1.0000 [1.0000, 1.0000] | 1.0317 [0.9794, 1.0632] | 1.0000 [1.0000, 1.0000] | PASS |
| ase-lj104 | O/A | 0.6579 | 0.5123 | 0.6025 | 0.5345 | 1.0000 [1.0000, 1.0000] | 1.0140 [0.9446, 1.0835] | 1.0000 [0.9997, 1.0003] | PASS |
| ase-lj104 | M/A | 0.6579 | 0.5123 | 0.5801 | 0.5345 | 1.0000 [1.0000, 1.0000] | 0.9828 [0.9285, 1.0191] | 1.0000 [0.9997, 1.0003] | PASS |
| au55-seed | O/M | 0.6666 | 0.5137 | 0.2013 | 0.5346 | 1.0000 [1.0000, 1.0000] | 1.0161 [1.0024, 1.0391] | 1.0000 [0.9999, 1.0000] | PASS |
| au55-seed | O/A | 0.6666 | 0.5137 | 0.6463 | 0.999 | 1.0000 [1.0000, 1.0000] | 0.9969 [0.9693, 1.0263] | 0.5199 [0.5161, 0.5299] | PASS |
| au55-seed | M/A | 0.6666 | 0.5137 | 0.8573 | 0.999 | 1.0000 [1.0000, 1.0000] | 0.9811 [0.9510, 0.9881] | 0.5199 [0.5161, 0.5299] | PASS |
| lj38 | O/M | 1 | 0.8296 | 0.7278 | 0.6138 | 0.4075 [0.2228, 0.9945] | 0.4731 [0.2478, 1.0039] | 0.7839 [0.5508, 1.0103] | PASS |
| lj38 | O/A | 1 | 0.5115 | 0.5115 | 0.8762 | 1.0000 [1.0000, 1.0000] | 1.0247 [0.9073, 1.0371] | 0.9512 [0.5566, 0.9594] | PASS |
| lj38 | M/A | 1 | 0.1854 | 0.3325 | 0.7988 | 2.4541 [1.0055, 4.4888] | 2.1656 [0.9676, 4.0362] | 1.2134 [0.5547, 1.7417] | FIX |
| m38r6 | O/M | 1 | 0.5115 | 0.6025 | 0.5 | 1.0000 [1.0000, 1.0000] | 0.9952 [0.9741, 1.0033] | 1.0000 [0.9968, 1.0034] | PASS |
| m38r6 | O/A | 1 | 0.5115 | 0.6025 | 0.9996 | 1.0000 [1.0000, 1.0000] | 0.9982 [0.9776, 1.0019] | 0.6470 [0.4286, 0.6775] | PASS |
| m38r6 | M/A | 1 | 0.5115 | 0.5115 | 0.9996 | 1.0000 [1.0000, 1.0000] | 1.0030 [0.9884, 1.0123] | 0.6470 [0.4286, 0.6774] | PASS |
| m55r14 | O/M | 0.6798 | 0.5116 | 0.6463 | 0.546 | 1.0000 [1.0000, 1.0000] | 0.9828 [0.9652, 0.9983] | 0.9995 [0.9995, 1.0000] | PASS |
| m55r14 | O/A | 0.6798 | 0.5116 | 0.6463 | 0.9995 | 1.0000 [1.0000, 1.0000] | 0.9787 [0.9495, 1.0021] | 0.4174 [0.3618, 0.5252] | PASS |
| m55r14 | M/A | 0.6798 | 0.5116 | 0.5345 | 0.9995 | 1.0000 [1.0000, 1.0000] | 0.9958 [0.9758, 1.0294] | 0.4177 [0.3618, 0.5255] | PASS |

Overall: **FIX**.


## lj75-seed

Verdict M versus A: UNRESOLVED.
Notes: calls-to-hit loss not real; outside upper above 1.10 not real.
Verdict O versus A: UNRESOLVED.
Notes: hit loss not real.
Verdict O versus M: UNRESOLVED.
Notes: hit loss not real; outside upper above 1.10 not real.

Binary sha256 A f308f080875555ba69d080a2aa947eef61e7fda3472e4bde3ad39659fa7c8448 M e42df421483ccfa569dd868c3700add29c67218b8b35f6d18b82d2b19564c93e O e5b270b6b11c84582ce6d7f9ef138d4208896362b1e9c110b458326893e1c16f.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
M traces bit-identical to A: 1 of 108.
Seeds whose trace differs: 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119.
O traces bit-identical to A: 106 of 108.
Seeds whose trace differs: 2037, 2097.


## ase-lj98

Verdict M versus A: PASS.
Notes: every bar holds.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A 37a85db5ea63e6cdd2f9888773af31180066791ac5abc53f8cdab97d36073103 M f6a21468a3292329596732b517185c0cab40490d988c734a9342758caa6ccef1 O 0e20ca68dd6837d2d7cf3448807aad2380d5c268675b761e353ad4825f849109.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
This driver does not write a call trajectory.
Printed best-energy strings match on 12 of 12 seeds.


## ase-lj104

Verdict M versus A: PASS.
Notes: every bar holds.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A 37a85db5ea63e6cdd2f9888773af31180066791ac5abc53f8cdab97d36073103 M f6a21468a3292329596732b517185c0cab40490d988c734a9342758caa6ccef1 O 0e20ca68dd6837d2d7cf3448807aad2380d5c268675b761e353ad4825f849109.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
This driver does not write a call trajectory.
Printed best-energy strings match on 12 of 12 seeds.


## au55-seed

Verdict M versus A: PASS.
Notes: every bar holds.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A a3f8b66515d5fb39875888ad0509f31673726b077ee0715696331a255a430772 M 59dbb36bb18ff87eb224621bd85cdb99b607163244e2906fe9e125d6d646df7c O d17a0f7526725c2e8f7c05d5f08948bd739351e837232c8cabbb2e4429a9b16e.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
This driver does not write a call trajectory.
Printed best-energy strings match on 12 of 12 seeds.


## lj38

Verdict M versus A: FIX.
Notes: calls-to-hit loss real; outside upper above 1.10 not real; RSS upper above 1.10 not real.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A f308f080875555ba69d080a2aa947eef61e7fda3472e4bde3ad39659fa7c8448 M e42df421483ccfa569dd868c3700add29c67218b8b35f6d18b82d2b19564c93e O e5b270b6b11c84582ce6d7f9ef138d4208896362b1e9c110b458326893e1c16f.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
M traces bit-identical to A: 0 of 12.
Seeds whose trace differs: 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023.
O traces bit-identical to A: 12 of 12.


## m38r6

Verdict M versus A: PASS.
Notes: every bar holds.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A db8a935228b2df97f9540a13982ff900af845a84fbbdd7d32919927bf043de96 M 1ad304992e9b551f6353018d1687390bc2695cf6547ccbfde968bb259172d8a4 O ac31e673acf0a6e51e8c2ee6ce36d6f6e3c4cd797c21879ade43ef527454976c.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
This driver does not write a call trajectory.
Printed best-energy strings match on 12 of 12 seeds.


## m55r14

Verdict M versus A: PASS.
Notes: every bar holds.
Verdict O versus A: PASS.
Notes: every bar holds.
Verdict O versus M: PASS.
Notes: every bar holds.

Binary sha256 A db8a935228b2df97f9540a13982ff900af845a84fbbdd7d32919927bf043de96 M 1ad304992e9b551f6353018d1687390bc2695cf6547ccbfde968bb259172d8a4 O ac31e673acf0a6e51e8c2ee6ce36d6f6e3c4cd797c21879ade43ef527454976c.
Revision A 7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec M 8824564b2479f0917c1e4927cd5ac938a86c2ac0 O c96807c54e8cc602322255c75be61425c7bf722a.
This driver does not write a call trajectory.
Printed best-energy strings match on 12 of 12 seeds.

