Copyright (c) anneal developers

Same-session comparison of develop and the conf-miss4 tree.
A is develop 61fa67cf380e46876b569da45e7cc091647a775b.
B is that commit plus the conf-miss4 kit, tree 91af99fceaf0591edddb071988c149cc711eaf37, kit sha256 5ca02856b67ecd4a0e938d50d91f73536660d6536cd739befdbcea53d05406f1.
A' is a second release build of A. The kit is not part of this tree.

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

Outside time per call is wall minus the time inside potential energy
and gradient calls, divided by the call count. Molecule inside time is
the phase-0 force-field counter, on the MMFF energy and gradient entry
points. Guard inside time is the clock around value_and_gradient.
Wall for a molecule is the job clock from the embedding through the
search. Wall for a guard is the search clock. The ratio below is the
ratio of the two medians.

Tests were fixed before the session. They are one-sided and exact:
Fisher on hits, in the direction of fewer B hits, and Mann-Whitney on
calls, wall and outside time per call, in the direction of larger B
values. Paired McNemar over seeds is reported beside Fisher and is not
in the family. The family is the eight molecules' four endpoints and
the five guard wall endpoints, 37 tests. A loss requires an
adjusted p below 0.05, a worse median or hit count, and a median
ratio outside the A/A or placement band. The band limit is the largest
per-seed A'/A ratio on that cell, or the largest such ratio pooled from
the measured A/A cells when the cell has no A'.

On every molecule cell the median B/A ratio of outside time per call
must be at most 1.10. A cell above that cap fails even
when its adjusted p is not below the threshold.

## Verdict

| cell | hits A | hits B | hits A' | p hits | p mcnemar | calls A | calls B | p calls | wall A | wall B | p wall | outside A | outside B | ratio | p outside | wall limit | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 98591 | 98591 | 5.0407e-01 | 4.04411 | 4.04885 | 3.8352e-01 | 1.7918e-05 | 1.79307e-05 | 1.00071 | 3.6033e-01 | 1.15839 | PASS |
| 915 | 17/24 | 17/24 |  | 6.2418e-01 | 1.0000e+00 | 48799 | 48799 | 5.0407e-01 | 1.71861 | 1.7613 | 4.7964e-01 | 1.57645e-05 | 1.60675e-05 | 1.01922 | 7.3597e-02 | 1.15839 | PASS |
| xld | 17/24 | 17/24 | 17/24 | 6.2418e-01 | 1.0000e+00 | 85303.5 | 85303.5 | 5.0407e-01 | 4.53028 | 4.56595 | 5.6088e-01 | 2.20475e-05 | 2.19566e-05 | 0.995878 | 5.9287e-01 | 1.0621 | PASS |
| fluvastatin | 8/24 | 8/24 |  | 6.1995e-01 | 1.0000e+00 | 78099 | 78099 | 5.0407e-01 | 4.07789 | 4.16828 | 3.9135e-01 | 2.20659e-05 | 2.23837e-05 | 1.0144 | 3.3763e-01 | 1.15839 | PASS |
| z34 | 3/24 | 3/24 |  | 6.6691e-01 | 1.0000e+00 | 86813.5 | 86813.5 | 5.0407e-01 | 5.36845 | 5.32519 | 4.3912e-01 | 2.52841e-05 | 2.55422e-05 | 1.01021 | 1.9779e-01 | 1.15839 | PASS |
| amprenavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 75753 | 75753 | 5.0407e-01 | 5.45795 | 5.31838 | 6.9891e-01 | 3.05172e-05 | 2.94819e-05 | 0.966074 | 6.6237e-01 | 1.15839 | PASS |
| lopinavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 93058 | 93058 | 5.0407e-01 | 10.5098 | 10.4267 | 5.6088e-01 | 4.74705e-05 | 4.6759e-05 | 0.985013 | 5.9287e-01 | 1.15839 | PASS |
| lorlatinib | 9/24 | 9/24 |  | 6.1694e-01 | 1.0000e+00 | 83112.5 | 83112.5 | 5.0407e-01 | 3.64923 | 3.67524 | 3.6801e-01 | 1.90627e-05 | 1.93237e-05 | 1.01369 | 9.9715e-02 | 1.15839 | PASS |
| lj38 | 23/24 | 23/24 | 23/24 | 1.0000e+00 | 1.0000e+00 | 117832 | 117832 | 1.0000e+00 | 0.551952 | 0.55151 | 5.4472e-01 | 2.56105e-06 | 2.60804e-06 | 1.01835 | 1.0000e+00 | 1.05794 | PASS |
| lj75 | 7/24 | 7/24 |  | 1.0000e+00 | 1.0000e+00 | 4000000 | 4000000 | 1.0000e+00 | 58.9838 | 58.7912 | 6.9891e-01 | 6.48432e-06 | 6.29118e-06 | 0.970215 | 1.0000e+00 | 4.43617 | PASS |
| m80r6 | 24/24 | 24/24 |  | 1.0000e+00 | 1.0000e+00 | 72907 | 72907 | 1.0000e+00 | 2.67939 | 2.63765 | 4.7151e-01 | 1.17809e-05 | 1.17112e-05 | 0.994087 | 1.0000e+00 | 4.43617 | PASS |
| au55 | 13/24 | 13/24 |  | 1.0000e+00 | 1.0000e+00 | 774530 | 774530 | 1.0000e+00 | 28.9094 | 28.8517 | 6.1648e-01 | 1.04545e-05 | 1.03595e-05 | 0.990914 | 1.0000e+00 | 4.43617 | PASS |
| hexamer | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 4578.5 | 4578.5 | 1.0000e+00 | 0.0147624 | 0.0147556 | 6.5486e-01 | 1.0984e-06 | 1.07784e-06 | 0.981284 | 1.0000e+00 | 4.43617 | PASS |

Overall: PASS

## Guard bit-identity

| cell | identical to A | mismatched seeds |
|---|---|---|
| lj38 | true |  |
| lj75 | true |  |
| m80r6 | true |  |
| au55 | true |  |
| hexamer | true |  |

## A/A band

| cell | arm | median wall ratio to A | median call ratio to A | median outside ratio to A | hit difference |
|---|---|---|---|---|---|
| rq3 | A' | 1.01152 | 1 | 0.999265 | 0 |
| rq3 | B | 1.00117 | 1 | 1.00071 | 0 |
| xld | A' | 1.01603 | 1 | 0.99478 | 0 |
| xld | B | 1.00787 | 1 | 0.995878 | 0 |
| lj38 | A' | 0.999288 | 1 | 1.0154 | 0 |
| lj38 | B | 0.9992 | 1 | 1.01835 | 0 |
| hexamer | A' | 1.0741 | 1 | 1.02561 | 0 |
| hexamer | B | 0.999541 | 1 | 0.981284 | 0 |

## Session

- base_ref: 61fa67cf380e46876b569da45e7cc091647a775b
- base_tree: cff796c507a8fa1d0bfb28888c3570b61988055e
- b_tree: 91af99fceaf0591edddb071988c149cc711eaf37
- kit_sha256: 5ca02856b67ecd4a0e938d50d91f73536660d6536cd739befdbcea53d05406f1
- harness_blobs: run_gate.py:8b1273f175a34ffb5d639fea6f299d4e84323c05 guard_trace.rs:f376d4f3f356c935ac50a6c8d037558454d5f339
- machine_id: 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873
- seeds: 2000-2023
- n: 24
- core: 0
- threads: 1
- family: 37
- alpha: 0.05
- outside_cap: 1.10
- build_a: 06a04a09c09e5090fc93165541dd8bf864684a8adbb8e24ce95e5133d1fe3f3b
- build_b: 1f3bb0ea454e6227698d20faf23fac1e482a5afc7fa03225c37682cb51560876
- build_ap: 69fc4719091431172a30a0bd62a6cfc52629611fe7d3812e190eba7355d96d95
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
- a_tree: cff796c507a8fa1d0bfb28888c3570b61988055e
- placement: none
