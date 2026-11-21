Copyright (c) anneal developers

Rescore of the stored conf-miss4 conformers. No search was repeated.
Both arms keep the geometries recorded in the gate session.

Polish is MMFF94 at dielectric one until the minimiser reports
convergence, the same polished cache the gate scored. A conformer
whose stereocentres or double bonds differ from the SMILES is
invalid and is dropped. Two valid geometries are the same conformer
when their heavy-atom RMSD, minimised over graph automorphisms
after a Kabsch fit, is under 0.25 angstrom and their energies
differ by less than 0.5 kcal/mol.

R-A applies that comparison. R-K applies it and also accepts a
candidate whose mirror image meets it. The mirror is a reflection
through the xy plane. The Kabsch fit still keeps proper rotations.

The reference is the lowest valid conformer of the union of that
cell's arms and seeds, including the second develop arm where the
session recorded one. R-K uses the same reference. It does not
replace the reference with a mirror image, and the reference
energy and geometry do not change under R-K.

## Hits

| molecule | A R-A | B R-A | A R-K | B R-K |
|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 24/24 | 24/24 |
| 915 | 17/24 | 17/24 | 17/24 | 17/24 |
| xld | 17/24 | 17/24 | 24/24 | 24/24 |
| fluvastatin | 8/24 | 8/24 | 8/24 | 8/24 |
| z34 | 3/24 | 3/24 | 3/24 | 3/24 |
| amprenavir | 2/24 | 2/24 | 2/24 | 2/24 |
| lopinavir | 2/24 | 2/24 | 2/24 | 2/24 |
| lorlatinib | 9/24 | 9/24 | 9/24 | 9/24 |

## Reference

Changed is no on every molecule: one geometry, one energy, both rules.

| molecule | ref kcal | geometry | changed |
|---|---|---|---|
| rq3 | 62.087577 | b42eeb6c97fa68e1ac3c3f1683c35b83681e6bc1d77056791525dc018ee7f532 | no |
| 915 | 9.442466 | 39c6ab5a79c35a385cb22ab202797fdfff47026c03e7bab0519d31b3a54d86aa | no |
| xld | -57.472585 | f183b0c7b1e256a051731ca13582ee972b0991d1986f44ec82eeaea4426f31de | no |
| fluvastatin | 14.478926 | 9384a67c5f58fd39d1d6ea1dbe50b452eb517d76106f346ffbbbe8f6b0280fac | no |
| z34 | 117.150066 | 72878c0afe3a2948eca7a73e504afc5648cb766239af000772ca7f6989934b35 | no |
| amprenavir | -26.291601 | e755d9d1a37ae0723a937996f5e2bacb378d1c2d60c73afb2a5a5d59de07faff | no |
| lopinavir | 11.626932 | 146e8a8c0169a86c1da770dd68df1eda6476db1b3b59ce60489cb6c7e8ceab66 | no |
| lorlatinib | 52.181037 | 3a91ef15cf1184010d530acc0cabda561858bf9d1d69ba8bb8465da955d6ddf9 | no |

## Flips

A flip is a seed whose hit status differs between R-A and R-K.
Mirror means the inverted heavy-atom RMSD is under 0.25 angstrom,
the direct RMSD is not, and the energy gap to the reference is
under 0.5 kcal/mol. Anything else is a distinct minimum.

| molecule | arm | seed | R-A | R-K | class | rmsd | rmsd inverted | energy gap |
|---|---|---|---|---|---|---|---|---|
| xld | A | 2002 | 0 | 1 | mirror | 2.018045 | 0.000039 | 0.000000 |
| xld | B | 2002 | 0 | 1 | mirror | 2.018045 | 0.000039 | 0.000000 |
| xld | B | 2007 | 0 | 1 | mirror | 2.018042 | 0.000033 | 0.000000 |
| xld | A | 2007 | 0 | 1 | mirror | 2.018042 | 0.000033 | 0.000000 |
| xld | A | 2008 | 0 | 1 | mirror | 2.018049 | 0.000012 | 0.000000 |
| xld | B | 2008 | 0 | 1 | mirror | 2.018049 | 0.000012 | 0.000000 |
| xld | B | 2013 | 0 | 1 | mirror | 2.018060 | 0.000047 | 0.000000 |
| xld | A | 2013 | 0 | 1 | mirror | 2.018060 | 0.000047 | 0.000000 |
| xld | A | 2016 | 0 | 1 | mirror | 2.018049 | 0.000058 | 0.000000 |
| xld | B | 2016 | 0 | 1 | mirror | 2.018049 | 0.000058 | 0.000000 |
| xld | B | 2017 | 0 | 1 | mirror | 2.018045 | 0.000015 | 0.000000 |
| xld | A | 2017 | 0 | 1 | mirror | 2.018045 | 0.000015 | 0.000000 |
| xld | B | 2023 | 0 | 1 | mirror | 2.018048 | 0.000010 | 0.000000 |
| xld | A | 2023 | 0 | 1 | mirror | 2.018048 | 0.000010 | 0.000000 |

Flips: 14. Mirror images: 14. Distinct minima: 0.

## Tests

The family is unchanged. It is the eight molecules' four endpoints
(hits, calls, wall, outside time) and the five guard wall endpoints, 37 tests.
Bonferroni multiplies a family probability by 37 and caps the product at 1.
A loss needs that adjusted probability below 0.05, a worse hit count
or median, and a median ratio outside the develop band. McNemar is
paired over seeds, is reported beside Fisher, and is not in the family.

Calls, wall and outside time enter the Mann-Whitney test with a miss
replaced by positive infinity. The medians, the outside-time ratio and
the 1.10 cap use the recorded finite times. On every molecule
the median ratio of outside time per call must be at most that cap.

Guard call and outside probabilities below are the same one-sided
exact Mann-Whitney test, including the tied case. They are reported
and they are not members of the 37-test family, so they do not move
a guard verdict. The guard wall endpoint remains the family member.

### R-A

| cell | hits A | hits B | hits A' | p hits | p mcnemar | calls A | calls B | p calls | wall A | wall B | p wall | outside A | outside B | ratio | p outside | wall limit | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 98591 | 98591 | 5.0407e-01 | 4.04411 | 4.04885 | 3.8352e-01 | 1.7918e-05 | 1.79307e-05 | 1.00071 | 3.6033e-01 | 1.15839 | PASS |
| 915 | 17/24 | 17/24 |  | 6.2418e-01 | 1.0000e+00 | 48799 | 48799 | 5.0407e-01 | 1.71861 | 1.7613 | 4.7964e-01 | 1.57645e-05 | 1.60675e-05 | 1.01922 | 2.1538e-01 | 1.15839 | PASS |
| xld | 17/24 | 17/24 | 17/24 | 6.2418e-01 | 1.0000e+00 | 85303.5 | 85303.5 | 5.0407e-01 | 4.53028 | 4.56595 | 5.4472e-01 | 2.20475e-05 | 2.19566e-05 | 0.995878 | 5.7693e-01 | 1.0621 | PASS |
| fluvastatin | 8/24 | 8/24 |  | 6.1995e-01 | 1.0000e+00 | 78099 | 78099 | 5.0407e-01 | 4.07789 | 4.16828 | 5.0407e-01 | 2.20659e-05 | 2.23837e-05 | 1.0144 | 4.5528e-01 | 1.15839 | PASS |
| z34 | 3/24 | 3/24 |  | 6.6691e-01 | 1.0000e+00 | 86813.5 | 86813.5 | 5.0407e-01 | 5.36845 | 5.32519 | 5.0407e-01 | 2.52841e-05 | 2.55422e-05 | 1.01021 | 5.0407e-01 | 1.15839 | PASS |
| amprenavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 75753 | 75753 | 5.0407e-01 | 5.45795 | 5.31838 | 5.1222e-01 | 3.05172e-05 | 2.94819e-05 | 0.966074 | 5.0407e-01 | 1.15839 | PASS |
| lopinavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 93058 | 93058 | 5.0407e-01 | 10.5098 | 10.4267 | 5.0407e-01 | 4.74705e-05 | 4.6759e-05 | 0.985013 | 4.9593e-01 | 1.15839 | PASS |
| lorlatinib | 9/24 | 9/24 |  | 6.1694e-01 | 1.0000e+00 | 83112.5 | 83112.5 | 5.0407e-01 | 3.64923 | 3.67524 | 4.3912e-01 | 1.90627e-05 | 1.93237e-05 | 1.01369 | 4.0713e-01 | 1.15839 | PASS |
| lj38 | 23/24 | 23/24 | 23/24 | 1.0000e+00 | 1.0000e+00 | 117832 | 117832 | 5.0407e-01 | 0.551952 | 0.55151 | 5.4472e-01 | 2.56105e-06 | 2.60804e-06 | 1.01835 | 5.2036e-01 | 1.05794 | PASS |
| lj75 | 7/24 | 7/24 |  | 1.0000e+00 | 1.0000e+00 | 4000000 | 4000000 | 5.0407e-01 | 58.9838 | 58.7912 | 5.2036e-01 | 6.48432e-06 | 6.29118e-06 | 0.970215 | 5.3662e-01 | 4.43617 | PASS |
| m80r6 | 24/24 | 24/24 |  | 1.0000e+00 | 1.0000e+00 | 72907 | 72907 | 5.0407e-01 | 2.67939 | 2.63765 | 4.7151e-01 | 1.17809e-05 | 1.17112e-05 | 0.994087 | 4.3108e-01 | 4.43617 | PASS |
| au55 | 13/24 | 13/24 |  | 1.0000e+00 | 1.0000e+00 | 774530 | 774530 | 5.0407e-01 | 28.9094 | 28.8517 | 5.2849e-01 | 1.04545e-05 | 1.03595e-05 | 0.990914 | 5.3662e-01 | 4.43617 | PASS |
| hexamer | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 4578.5 | 4578.5 | 5.0407e-01 | 0.0147624 | 0.0147556 | 6.5486e-01 | 1.0984e-06 | 1.07784e-06 | 0.981284 | 7.5340e-01 | 4.43617 | PASS |

Overall: PASS

### R-K

| cell | hits A | hits B | hits A' | p hits | p mcnemar | calls A | calls B | p calls | wall A | wall B | p wall | outside A | outside B | ratio | p outside | wall limit | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 98591 | 98591 | 5.0407e-01 | 4.04411 | 4.04885 | 3.8352e-01 | 1.7918e-05 | 1.79307e-05 | 1.00071 | 3.6033e-01 | 1.15839 | PASS |
| 915 | 17/24 | 17/24 |  | 6.2418e-01 | 1.0000e+00 | 48799 | 48799 | 5.0407e-01 | 1.71861 | 1.7613 | 4.7964e-01 | 1.57645e-05 | 1.60675e-05 | 1.01922 | 2.1538e-01 | 1.15839 | PASS |
| xld | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 85303.5 | 85303.5 | 5.0407e-01 | 4.53028 | 4.56595 | 5.6088e-01 | 2.20475e-05 | 2.19566e-05 | 0.995878 | 5.9287e-01 | 1.0621 | PASS |
| fluvastatin | 8/24 | 8/24 |  | 6.1995e-01 | 1.0000e+00 | 78099 | 78099 | 5.0407e-01 | 4.07789 | 4.16828 | 5.0407e-01 | 2.20659e-05 | 2.23837e-05 | 1.0144 | 4.5528e-01 | 1.15839 | PASS |
| z34 | 3/24 | 3/24 |  | 6.6691e-01 | 1.0000e+00 | 86813.5 | 86813.5 | 5.0407e-01 | 5.36845 | 5.32519 | 5.0407e-01 | 2.52841e-05 | 2.55422e-05 | 1.01021 | 5.0407e-01 | 1.15839 | PASS |
| amprenavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 75753 | 75753 | 5.0407e-01 | 5.45795 | 5.31838 | 5.1222e-01 | 3.05172e-05 | 2.94819e-05 | 0.966074 | 5.0407e-01 | 1.15839 | PASS |
| lopinavir | 2/24 | 2/24 |  | 6.9574e-01 | 1.0000e+00 | 93058 | 93058 | 5.0407e-01 | 10.5098 | 10.4267 | 5.0407e-01 | 4.74705e-05 | 4.6759e-05 | 0.985013 | 4.9593e-01 | 1.15839 | PASS |
| lorlatinib | 9/24 | 9/24 |  | 6.1694e-01 | 1.0000e+00 | 83112.5 | 83112.5 | 5.0407e-01 | 3.64923 | 3.67524 | 4.3912e-01 | 1.90627e-05 | 1.93237e-05 | 1.01369 | 4.0713e-01 | 1.15839 | PASS |
| lj38 | 23/24 | 23/24 | 23/24 | 1.0000e+00 | 1.0000e+00 | 117832 | 117832 | 5.0407e-01 | 0.551952 | 0.55151 | 5.4472e-01 | 2.56105e-06 | 2.60804e-06 | 1.01835 | 5.2036e-01 | 1.05794 | PASS |
| lj75 | 7/24 | 7/24 |  | 1.0000e+00 | 1.0000e+00 | 4000000 | 4000000 | 5.0407e-01 | 58.9838 | 58.7912 | 5.2036e-01 | 6.48432e-06 | 6.29118e-06 | 0.970215 | 5.3662e-01 | 4.43617 | PASS |
| m80r6 | 24/24 | 24/24 |  | 1.0000e+00 | 1.0000e+00 | 72907 | 72907 | 5.0407e-01 | 2.67939 | 2.63765 | 4.7151e-01 | 1.17809e-05 | 1.17112e-05 | 0.994087 | 4.3108e-01 | 4.43617 | PASS |
| au55 | 13/24 | 13/24 |  | 1.0000e+00 | 1.0000e+00 | 774530 | 774530 | 5.0407e-01 | 28.9094 | 28.8517 | 5.2849e-01 | 1.04545e-05 | 1.03595e-05 | 0.990914 | 5.3662e-01 | 4.43617 | PASS |
| hexamer | 24/24 | 24/24 | 24/24 | 1.0000e+00 | 1.0000e+00 | 4578.5 | 4578.5 | 5.0407e-01 | 0.0147624 | 0.0147556 | 6.5486e-01 | 1.0984e-06 | 1.07784e-06 | 0.981284 | 7.5340e-01 | 4.43617 | PASS |

Overall: PASS

## Session

- base_ref: 61fa67cf380e46876b569da45e7cc091647a775b
- base_tree: cff796c507a8fa1d0bfb28888c3570b61988055e
- b_tree: 91af99fceaf0591edddb071988c149cc711eaf37
- kit_sha256: 5ca02856b67ecd4a0e938d50d91f73536660d6536cd739befdbcea53d05406f1
- machine_id: 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873
- seeds: 2000-2023
- n: 24
- family: 37
- family_unchanged: yes
- guard_calls_outside_in_family: no
- miss_rank: inf
- reference_changes_under_R-K: no
