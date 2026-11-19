# nm3-land gate B timing recheck

**Verdict: FAIL: sc126-55**

## Rule

For each cell, over the 24 seeds,

R(X/A) = (sum of outside_s for arm X / sum of charged for arm X) / (sum of outside_s for arm A / sum of charged for arm A).

The 95% interval is a paired percentile bootstrap with 10000 resamples, random.Random(12345) and randrange, endpoints at the type 7 quantiles 0.025 and 0.975.
A cell passes if the lower endpoint of the B/A interval is <= 1.10, or if the B/A point ratio lies inside the A2/A interval, endpoints included. Otherwise it fails.
The per-seed median is the type 7 median of the 24 per-seed outside-per-call ratios.
A and A2 outcomes match when both finished, the hit flag matches, evals_to_hit matches, charged matches, and best_bits_hex matches.

## Provenance

- Session `20261119T213412Z` at `2026-11-19T21:34:12Z`
- Machine id sha256 `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- CPU `Intel(R) Xeon(R) Processor` (x86_64, kernel `6.12.113+`, affinity `0`)
- Compilers `rustc 1.88.0 (6b00bc388 2025-06-23)`; `cargo 1.88.0 (873a06493 2025-05-10)`; Python `3.12.3`
- Arm A `commit:5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- Arm B `tree:2ba43832b540c6e9005cfb6be31243b36ab086a7` tree `2ba43832b540c6e9005cfb6be31243b36ab086a7`
- Arm A2 `commit:5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- Seeds `2100-2123`; cells `blj115n38 blj105n38 sc126-55`
- Order `six_permutation_williams_rotation`; timeout `43200.0` s; seat `benchbot`
- Bootstrap resamples `10000`; generator seed `12345`; threshold `1.10`
- Sutton-Chen library sha256 `ee31dc58507142399dec4cedcba593e33eac6b934449fa09c775036b1cb912d6`
- Slot counts: slot 1 A=24, slot 1 B=24, slot 1 A2=24, slot 2 A=24, slot 2 B=24, slot 2 A2=24, slot 3 A=24, slot 3 B=24, slot 3 A2=24

### Release binary digests

| arm | binary | sha256 |
| --- | --- | --- |
| A | `h2h_sc` | `37c765cb8b068bd35e69ed38964c6d817d4e40eeda296cea03ca525b02bcc53c` |
| A | `nm3-gate-driver` | `f46c43acdfd2a29f5000546198792abd0b58e6bd3519b3f303f0b98acd441979` |
| B | `h2h_sc` | `f7a29cdd533e67d1aa1adb6ce5be8bb235348f6d448cc40d3215b63fcba0e5e8` |
| B | `nm3-gate-driver` | `04f996fddf207b99aaf40c157404c6bb3d30d715c9325413b3b5b5fa53a631b4` |
| A2 | `h2h_sc` | `b09c159c144c86f601aee9eb9c7a6e06d6c8535321891df19affc0c3040e867e` |
| A2 | `nm3-gate-driver` | `d28554a7a9fbecfe24da4cecf6650f1071391a0714c6b9160d5088cff308930d` |

## Cells

| cell | R B/A | B/A 95% CI | R A2/A | A2/A 95% CI | median B/A | median A2/A | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blj115n38 | 1.63738406 | [0.91240488, 3.14879670] | 0.99392735 | [0.98284827, 0.99949772] | 1.04723963 | 0.99414740 | PASS |
| blj105n38 | 0.97774363 | [0.89264975, 1.07061113] | 0.98868093 | [0.97298835, 1.00195317] | 0.98833118 | 0.99307540 | PASS |
| sc126-55 | 1.20198371 | [1.10084288, 1.33069061] | 0.97647024 | [0.94879261, 1.00515977] | 1.16413799 | 0.99151251 | FAIL |

## Hits and calls

| cell | hits A | hits B | hits A2 | charged A | charged B | charged A2 | outside A (s) | outside B (s) | outside A2 (s) | evals median A | evals median B | evals median A2 | failed A | failed B | failed A2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blj115n38 | 22/24 | 19/24 | 22/24 | 4005242 | 5258290 | 4005242 | 34.797810 | 74.802867 | 34.586495 | 133708 | 191301 | 133708 | 0 | 0 | 0 |
| blj105n38 | 1/24 | 4/24 | 1/24 | 9523534 | 8608601 | 9523534 | 152.283411 | 134.589765 | 150.559705 | 400000 | 400000 | 400000 | 0 | 0 | 0 |
| sc126-55 | 24/24 | 24/24 | 24/24 | 836068 | 1056208 | 836068 | 3.556432 | 5.400338 | 3.472750 | 16238 | 26627.5 | 16238 | 0 | 0 | 0 |

## A versus A2 outcome identity

| cell | matching seeds | seeds that differ |
| --- | --- | --- |
| blj115n38 | 24/24 |  |
| blj105n38 | 24/24 |  |
| sc126-55 | 24/24 |  |

Outside seconds are the sums of recorded outside_s. Charged values are the sums of recorded charged calls. R divides those sums; it is not the mean of the per-seed ratios.
