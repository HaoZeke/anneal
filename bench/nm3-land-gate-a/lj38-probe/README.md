# Plain LJ38 probe

verdict: PASS

## Budgets

- mask-probe budget: `30000`
- `PAPER_BUDGETS` LJ38 budget: `400000`
- reference: `-173.928427`
- hit tolerance: `0.0001`

## Definition

Both budgets use the plain LJ38 walk: `Config::recommended(38)`, 200 relax steps,
`MoveLibrary::LeanBurst`, stall trail, escape, and restart off, Bayes screening off,
and no surface portfolio. A run spends its full budget. A hit is a recorded best at
or below the reference plus `1e-4`. Calls-to-hit are the first recorded improvement
inside that window, or the budget when no such improvement is present. Potential time
is time inside the Lennard-Jones kernel. Outside time per call divides the remaining
wall time by the charged call count. Probabilities are exact reduced ratios.

## Provenance

- session: `c9e94623c43eb004f8f7`
- machine id: `sha256:3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- arm A: commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`, tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- arm B: commit `8f1e0b5c5078062d5f241371554555421526452c`, tree `2ba43832b540c6e9005cfb6be31243b36ab086a7`
- kit: `sha256:cddaf32978ceeaeedb73d7ebff0f00d5006689c19d110ce8df69fb2881204134`
- probe blob: `7571436ba1cb868823070e94166475edcb66a87d`
- runner blob: `bca7b1d0e583717b91d7631903212ee9046d4db5`

## Protocol

The probe executable for each arm was linked to that arm's existing release library.
Library artifact hashes were unchanged by the link. One session ran every job serially
on CPU 0, with every listed thread-pool variable set to 1. Seeds 2012-2059 are common
to both arms. Even offsets use AB and odd offsets use BA. Each `(budget, slot, arm)`
count is 24. The verdict family is exact two-sided Fisher hits and exact two-sided
Mann-Whitney calls with non-hits censored at the budget, at both budgets. Fisher sums
every table whose probability is no larger than the observed table. Mann-Whitney uses
the exact midrank distribution and doubles the smaller tail, capped at 1. Bonferroni
uses family size 4. A hits loss requires fewer B hits. A calls loss requires a
larger B median. Either loss also requires its exact p to be below
0.0125. The hit-only Mann-Whitney is reported separately and is outside
that family. U counts B-versus-A pairs, with ties scored one half.

## Results

| budget | hits A/B | Fisher p | adjusted p | calls A/B | U | MW p | adjusted p | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30000 | 8/10 | 71263920677/89712093717 | 1/1 | 30000/30000 | 1103 | 4633821286181/7625527965945 | 1/1 | PASS |
| 400000 | 48/45 | 23/95 | 92/95 | 93083.5/82844 | 1034 | 125781523266395430519915419/321753350693314945421080155 | 1/1 | PASS |

| budget | hit-only n A/B | hit-only calls A/B | hit-only U | hit-only p | wall s A/B | potential s A/B | outside/call s A/B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 8/10 | 20553.5/19364.5 | 39 | 21122/21879 | 0.3690159285/0.4018841925 | 0.0629318175/0.062891257 | 1.02104295333e-05/1.128662445e-05 |
| 400000 | 48/45 | 93083.5/79509 | 890 | 3784425436807220071770154/25966059880513135946262609 | 4.731351086/4.928453454 | 0.8440954095/0.843962915 | 9.71438068875e-06/1.02137194037e-05 |

## Seeds at budget 30000

Hit outcomes differ on 14 seeds: 2012, 2015, 2018, 2020, 2025, 2028, 2031, 2032, 2038, 2042, 2045, 2046, 2056, 2057. Calls differ on 16 seeds.

| seed | A hit | A calls | B hit | B calls |
|---:|---:|---:|---:|---:|
| 2012 | 0 | 30000 | 1 | 14682 |
| 2013 | 1 | 23301 | 1 | 16927 |
| 2014 | 0 | 30000 | 0 | 30000 |
| 2015 | 0 | 30000 | 1 | 25442 |
| 2016 | 0 | 30000 | 0 | 30000 |
| 2017 | 0 | 30000 | 0 | 30000 |
| 2018 | 0 | 30000 | 1 | 21189 |
| 2019 | 0 | 30000 | 0 | 30000 |
| 2020 | 1 | 17806 | 0 | 30000 |
| 2021 | 0 | 30000 | 0 | 30000 |
| 2022 | 0 | 30000 | 0 | 30000 |
| 2023 | 0 | 30000 | 0 | 30000 |
| 2024 | 0 | 30000 | 0 | 30000 |
| 2025 | 1 | 10124 | 0 | 30000 |
| 2026 | 0 | 30000 | 0 | 30000 |
| 2027 | 0 | 30000 | 0 | 30000 |
| 2028 | 1 | 11238 | 0 | 30000 |
| 2029 | 0 | 30000 | 0 | 30000 |
| 2030 | 0 | 30000 | 0 | 30000 |
| 2031 | 0 | 30000 | 1 | 17540 |
| 2032 | 0 | 30000 | 1 | 1909 |
| 2033 | 0 | 30000 | 0 | 30000 |
| 2034 | 0 | 30000 | 0 | 30000 |
| 2035 | 0 | 30000 | 0 | 30000 |
| 2036 | 0 | 30000 | 0 | 30000 |
| 2037 | 0 | 30000 | 0 | 30000 |
| 2038 | 1 | 26726 | 0 | 30000 |
| 2039 | 0 | 30000 | 0 | 30000 |
| 2040 | 0 | 30000 | 0 | 30000 |
| 2041 | 0 | 30000 | 0 | 30000 |
| 2042 | 0 | 30000 | 1 | 28972 |
| 2043 | 1 | 28306 | 1 | 28610 |
| 2044 | 0 | 30000 | 0 | 30000 |
| 2045 | 0 | 30000 | 1 | 25194 |
| 2046 | 0 | 30000 | 1 | 2889 |
| 2047 | 0 | 30000 | 0 | 30000 |
| 2048 | 0 | 30000 | 0 | 30000 |
| 2049 | 0 | 30000 | 0 | 30000 |
| 2050 | 0 | 30000 | 0 | 30000 |
| 2051 | 0 | 30000 | 0 | 30000 |
| 2052 | 0 | 30000 | 0 | 30000 |
| 2053 | 0 | 30000 | 0 | 30000 |
| 2054 | 0 | 30000 | 0 | 30000 |
| 2055 | 0 | 30000 | 0 | 30000 |
| 2056 | 1 | 3858 | 0 | 30000 |
| 2057 | 1 | 27336 | 0 | 30000 |
| 2058 | 0 | 30000 | 0 | 30000 |
| 2059 | 0 | 30000 | 0 | 30000 |

## Seeds at budget 400000

Hit outcomes differ on 3 seeds: 2016, 2044, 2047. Calls differ on 48 seeds.

| seed | A hit | A calls | B hit | B calls |
|---:|---:|---:|---:|---:|
| 2012 | 1 | 110845 | 1 | 14682 |
| 2013 | 1 | 23301 | 1 | 16927 |
| 2014 | 1 | 103764 | 1 | 149659 |
| 2015 | 1 | 113442 | 1 | 25442 |
| 2016 | 1 | 126349 | 0 | 400000 |
| 2017 | 1 | 56707 | 1 | 114284 |
| 2018 | 1 | 95343 | 1 | 21189 |
| 2019 | 1 | 77130 | 1 | 83053 |
| 2020 | 1 | 17806 | 1 | 37409 |
| 2021 | 1 | 89370 | 1 | 168130 |
| 2022 | 1 | 90912 | 1 | 269282 |
| 2023 | 1 | 54799 | 1 | 53967 |
| 2024 | 1 | 83543 | 1 | 97745 |
| 2025 | 1 | 10124 | 1 | 120515 |
| 2026 | 1 | 66947 | 1 | 82635 |
| 2027 | 1 | 82330 | 1 | 120527 |
| 2028 | 1 | 11238 | 1 | 46197 |
| 2029 | 1 | 108852 | 1 | 79509 |
| 2030 | 1 | 163148 | 1 | 47178 |
| 2031 | 1 | 182315 | 1 | 17540 |
| 2032 | 1 | 134126 | 1 | 1909 |
| 2033 | 1 | 270529 | 1 | 184353 |
| 2034 | 1 | 141186 | 1 | 100179 |
| 2035 | 1 | 151091 | 1 | 49470 |
| 2036 | 1 | 326414 | 1 | 139097 |
| 2037 | 1 | 86239 | 1 | 32124 |
| 2038 | 1 | 26726 | 1 | 67678 |
| 2039 | 1 | 37423 | 1 | 109824 |
| 2040 | 1 | 90999 | 1 | 65179 |
| 2041 | 1 | 300775 | 1 | 56972 |
| 2042 | 1 | 219225 | 1 | 28972 |
| 2043 | 1 | 28306 | 1 | 28610 |
| 2044 | 1 | 77290 | 0 | 400000 |
| 2045 | 1 | 95168 | 1 | 25194 |
| 2046 | 1 | 160621 | 1 | 2889 |
| 2047 | 1 | 224508 | 0 | 400000 |
| 2048 | 1 | 74870 | 1 | 95865 |
| 2049 | 1 | 121055 | 1 | 233604 |
| 2050 | 1 | 138257 | 1 | 143862 |
| 2051 | 1 | 90616 | 1 | 232528 |
| 2052 | 1 | 134435 | 1 | 186227 |
| 2053 | 1 | 139877 | 1 | 163134 |
| 2054 | 1 | 75540 | 1 | 49168 |
| 2055 | 1 | 313102 | 1 | 36176 |
| 2056 | 1 | 3858 | 1 | 133055 |
| 2057 | 1 | 27336 | 1 | 31510 |
| 2058 | 1 | 78444 | 1 | 97838 |
| 2059 | 1 | 230477 | 1 | 124736 |
