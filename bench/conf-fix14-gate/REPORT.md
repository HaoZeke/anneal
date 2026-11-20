Copyright (c) anneal developers

Same-session comparison of develop and the conf-fix14 tree.
A is develop 5eb79cdcb0a986984dc7b487e5a3cc93224246f9.
B is that commit plus the conf-fix14 kit, tree 2ca02a84d3a5ed6452263d23c0cb810d6a0d3dcc, kit sha256 8e5600c1070c8d98d2d9936a4c9d70f5afa46f10d35beb61c7f723171cd5fd85.
A' is a second release build of A. The kit is not part of this tree.
The rows were produced with run_gate.py blob 4b0c45fe, which differs from the committed harness only by one removed environment-variable line that had no effect on the measurement.

Seeds are 2000-2023, twenty-four per arm. Molecule caps are the per-seed
anneal budgets of the paired conformer benchmark; a seed with no recorded
budget uses the median of that molecule's recorded budgets. The search is
cluster_search on the recommended molecular preset. No preset field is overridden.

A molecule hit is a reported conformer that polishes to the lowest valid
conformer of the union of the session's arms and seeds. Polish is MMFF94
at dielectric one until the minimiser reports convergence. A conformer
whose stereocentres or double bonds differ from the SMILES is invalid and
is dropped. Two valid geometries are the same conformer when their
heavy-atom RMSD, minimised over graph automorphisms after a Kabsch fit,
is under 0.25 angstrom and their energies differ by less than 0.5 kcal/mol.
Molecule calls are energy plus gradient force-field calls. A guard call
is the charged count at the published minimum, or the whole budget on a miss.
A guard hit uses DEFAULT_HIT_TOL (0.001) against the published minimum.
The guard pass rule is bit-identity of the trajectory hash, the charged
count, the call count and the best-energy bits on every seed.

Tests are one-sided and exact: Fisher on hits, in the direction of fewer
B hits, and Mann-Whitney on calls and wall, in the direction of larger B
values. The family is the eight molecules' three endpoints and the five
guard wall endpoints, 29 tests. A loss requires an adjusted p below
0.05, a worse median or hit count, and a median ratio outside the A/A
or placement band. The band limit is the largest per-seed A'/A ratio on
that cell, or the largest such ratio pooled from the measured A/A cells
when the cell has no A'.

## Verdict

| cell | hits A | hits B | hits A' | p hits | calls A | calls B | p calls | wall A | wall B | p wall | wall limit | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 98591 | 98584 | 9.4702e-01 | 4.16748 | 34.4411 | 3.1010e-14 | 1.21943 | FAIL |
| 915 | 16/24 | 24/24 |  | 1.0000e+00 | 48791.5 | 48800.5 | 4.7964e-01 | 1.73427 | 13.4126 | 3.1010e-14 | 1.21943 | FAIL |
| xld | 20/24 | 20/24 | 20/24 | 6.4961e-01 | 85301.5 | 85233.5 | 6.0865e-01 | 4.55598 | 50.8373 | 3.1010e-14 | 1.20222 | FAIL |
| fluvastatin | 10/24 | 21/24 |  | 9.9991e-01 | 78099 | 78099.5 | 5.8492e-01 | 4.29255 | 39.8039 | 3.1010e-14 | 1.21943 | FAIL |
| z34 | 0/24 | 5/24 |  | 1.0000e+00 | 86811.5 | 86757 | 5.9287e-01 | 5.50625 | 53.7655 | 1.7871e-10 | 1.21943 | FAIL |
| amprenavir | 3/24 | 20/24 |  | 1.0000e+00 | 75754 | 75748 | 9.1407e-01 | 5.72314 | 70.3046 | 3.1010e-14 | 1.21943 | FAIL |
| lopinavir | 0/24 | 9/24 |  | 1.0000e+00 | 93058 | 93054.5 | 5.6088e-01 | 10.5907 | 154.707 | 3.1010e-14 | 1.21943 | FAIL |
| lorlatinib | 11/24 | 9/24 |  | 3.8508e-01 | 83114 | 83103.5 | 4.9593e-01 | 3.78906 | 4.24867 | 8.6939e-04 | 1.21943 | PASS |
| lj38 | 23/24 | 23/24 | 23/24 | 1.0000e+00 | 117832 | 117832 | 1.0000e+00 | 0.589099 | 0.58539 | 5.3662e-01 | 1.15622 | PASS |
| lj75 | 7/24 | 7/24 |  | 1.0000e+00 | 4000000 | 4000000 | 1.0000e+00 | 58.9279 | 58.5037 | 5.9287e-01 | 3.54502 | PASS |
| m80r6 | 24/24 | 24/24 |  | 1.0000e+00 | 72907 | 72907 | 1.0000e+00 | 2.83662 | 2.81514 | 5.0407e-01 | 3.54502 | PASS |
| au55 | 13/24 | 13/24 |  | 1.0000e+00 | 774530 | 774530 | 1.0000e+00 | 28.3901 | 28.4639 | 5.7693e-01 | 3.54502 | PASS |
| hexamer | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 4578.5 | 4578.5 | 1.0000e+00 | 0.0153675 | 0.0154452 | 5.1222e-01 | 3.54502 | PASS |

Overall: FAIL

## Guard bit-identity

| cell | identical to A | mismatched seeds |
|---|---|---|
| lj38 | true |  |
| lj75 | true |  |
| m80r6 | true |  |
| au55 | true |  |
| hexamer | true |  |

## A/A band

| cell | arm | median wall ratio to A | median call ratio to A | hit difference |
|---|---|---|---|---|
| rq3 | A' | 1.04207 | 1 | 0 |
| rq3 | B | 8.26424 | 0.999929 | 0 |
| xld | A' | 1.02722 | 1 | 0 |
| xld | B | 11.1584 | 0.999203 | 0 |
| lj38 | A' | 1.01455 | 1 | 0 |
| lj38 | B | 0.993704 | 1 | 0 |
| hexamer | A' | 1.01572 | 1 | 0 |
| hexamer | B | 1.00506 | 1 | 0 |

## Session

- base_ref: 5eb79cdcb0a986984dc7b487e5a3cc93224246f9
- base_tree: 7a2ba49b37241a8fe42b2c45c226374de6d92274
- b_tree: 2ca02a84d3a5ed6452263d23c0cb810d6a0d3dcc
- kit_sha256: 8e5600c1070c8d98d2d9936a4c9d70f5afa46f10d35beb61c7f723171cd5fd85
- harness_blobs: run_gate.py:4b0c45fe273bb8d91258710b7c77f3e78c362255 guard_trace.rs:56f354180171104420b2ffc4bb19027f3884d710
- machine_id: 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873
- seeds: 2000-2023
- n: 24
- core: 0
- threads: 1
- family: 29
- alpha: 0.05
- build_a: 45a1f5ea5097ed6f0a6112ca3ef92d4e56ffa62e5dba963f30aff971d61d1d7f
- build_b: df49b50cd72e4a3b9480a86558a1928cc91d2f88f6865309918f9d190bce55e6
- build_ap: 40be2432a787ef6273f49aafd9dd1e96594699e3a6e46f0d291a2077a1d754a1
- balance_rq3: 0:A=12 1:B=12 2:Ap=12 3:Ap=12 4:B=12 5:A=12
- balance_915: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_xld: 0:A=12 1:B=12 2:Ap=12 3:Ap=12 4:B=12 5:A=12
- balance_fluvastatin: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_z34: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_amprenavir: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_lopinavir: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_lorlatinib: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_lj38: 0:A=12 1:B=12 2:Ap=12 3:Ap=12 4:B=12 5:A=12
- balance_lj75: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_m80r6: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_au55: 0:A=12 1:B=12 2:B=12 3:A=12
- balance_hexamer: 0:A=12 1:B=12 2:Ap=12 3:Ap=12 4:B=12 5:A=12
- budget_lj38: 400000
- budget_lj75: 4000000
- budget_m80r6: 4000000
- budget_au55: 1000000
- budget_hexamer: 500000
- a_tree: 7a2ba49b37241a8fe42b2c45c226374de6d92274
- placement: none
