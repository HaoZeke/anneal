# nmpins7 gate, shard mw

Cells m75r14, m80r14 and tip4p20. Seeds 2000-2011. Arms A, B and C ran in one serial session on core 0, one thread, release builds. A is develop plus the M75 table row and the two-hour driver cap. B and C are the nmpins7 kit plus that same cap. B leaves general_symmetry off. C sets general_symmetry. The kit head is an ancestor of the B and C build and is not an ancestor of A.

A hit is the reference minimum plus 0.001. m80r14 uses the panel reference and paper budget. m75r14 uses the bold Cambridge 75C rho 14 energy and the same 4_000_000 budget the Morse driver assigns from 75 atoms, equal to the m80r14 paper budget. tip4p20 uses the Cambridge TIP4P header and the water-panel budget. Medians and IQRs are Hyndman-Fan type 7. Calls are charged evaluations. Outside time per call is outside-potential time divided by those calls. The ratio is the ratio of medians.

Fisher p is the exact one-sided test that the named arm has fewer hits than A. Mann-Whitney p is the one-sided normal approximation with the tie and continuity corrections, testing that the named arm is higher than A. A sample that is entirely tied has p = 1. Bonferroni uses m = 30. B loses on hits or calls when the adjusted p is below 0.05, and loses on time when the outside-per-call median ratio exceeds 1.1. C is reported with the same statistics and is not gated.

## Provenance

- record base `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- kit `4cd811004c67f26b0e216664cb51352f04654c09` tree `51c1a7e0180585d078c3457b5550b2963d1ab752`
- arm A build `02598d4d52c3f43d8297cc636a9345b9f558e174` tree `7c60bc5489dbc15a3a6df0fb99c8f5526b3e8d5a`
- kit build `6212f3c7d258b4c222b5c7847ddb0ff334da11c5` tree `11d900b43e669af4fcd79260c6edfe2cb961668a`
- data commit `a9c2270a79e07553581a4a2ab2750bfdfa084367`
- rows blob `a8c8b90f22f371c161a60861f5d8fe1c4ad57876`
- sessions blob `36ecdeec54cacaa180bf09342ec2d31d1ba1144b`
- harness blob `735c86dfa68ed98686b1194a44804473fba4cbdd`
- session `6de4851e203cb7d2` machine `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873` core 0 affinity 0
- rustc `rustc 1.100.0 (0ef1bc386 2026-11-09)`

Package versions for arm A, then the kit build when the version or revision differs:

- anneal-core 0.12.0
- rgmin 0.5.0
- readcon-core 0.14.8
- readcon-db 0.1.5
- eindir-core 0.6.0
- rgpot-core 3.5.1
- rgpot-pair 3.7.0 revision 8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e
- rgsaddle 0.3.0

Binary digests:

- A Morse `00aa822dcf1aa6592dc8fb875bb347f855258f0681a493c54ff440958d63f5ce`
- A TIP4P `34f96603e06a533fefac9ad772ecc9890c642eed16b27a02f592c312ee9c87ce`
- kit Morse `cd2df303701446eeb0c89060e1d80c5886527dbc8b5aea59c113493ba7587839`
- kit TIP4P `ab2ff30c280b0d3b61d268df81e70aa33ff55f0d4267257c6b2257173aea8fca`

## Balance

Each gate cell has four rows for every (slot, arm) pair. The m75r14 noise block runs replica 1 first on six seeds and replica 2 first on six seeds.

## Results

| cell | A hits | B hits | C hits | A calls | B calls | C calls | A outside/call | B outside/call | C outside/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m75r14 | 11/12 | 11/12 | 11/12 | 1.88845e+06 (646842) | 1.88845e+06 (646842) | 1.88845e+06 (646842) | 9.2406e-06 | 9.2146e-06 | 9.2458e-06 |
| m80r14 | 7/12 | 7/12 | 5/12 | 2374233 (2.08602e+06) | 2374233 (2.08602e+06) | 4000000 (2.08607e+06) | 9.9881e-06 | 9.8015e-06 | 9.7649e-06 |
| tip4p20 | 6/12 | 2/12 | 2/12 | 1.68625e+06 (1.37663e+06) | 2000000 (0) | 2000000 (0) | 3.3833e-06 | 2.9917e-06 | 3.0997e-06 |

| cell | Fisher B | Fisher B adj | MW calls B | MW calls B adj | outside B/A | MW outside B | MW outside B two-sided | verdict B | Fisher C | MW calls C | outside C/A | MW outside C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m75r14 | 7.608696e-01 | 1.000000e+00 | 5.115450e-01 | 1.000000e+00 | 0.997189 | 4.426170e-01 | 8.852339e-01 | pass | 7.608696e-01 | 5.115450e-01 | 1.000564 | 3.325028e-01 |
| m80r14 | 6.599139e-01 | 1.000000e+00 | 5.119711e-01 | 1.000000e+00 | 0.981322 | 4.654937e-01 | 9.309874e-01 | pass | 3.421363e-01 | 3.440855e-01 | 0.977658 | 6.881976e-01 |
| tip4p20 | 9.651366e-02 | 1.000000e+00 | 3.158514e-02 | 9.475542e-01 | 0.884236 | 9.824554e-01 | 4.040412e-02 | pass | 9.651366e-02 | 3.158514e-02 | 0.916151 | 9.797979e-01 |

m75r14: calls-to-hit medians A 1.88845e+06, B 1.88845e+06, C 1.88845e+06. Capped seeds A 0, B 0, C 0. The B/A outside median ratio is inside the A/A per-seed ratio IQR and inside the A/A per-seed ratio range.
m80r14: calls-to-hit medians A 2374233, B 2374233, C 4000000. Capped seeds A 0, B 0, C 0. The B/A outside median ratio is outside the A/A per-seed ratio IQR and inside the A/A per-seed ratio range.
tip4p20: calls-to-hit medians A 1.68625e+06, B 2000000, C 2000000. Capped seeds A 0, B 0, C 0. The B/A outside median ratio is outside the A/A per-seed ratio IQR and outside the A/A per-seed ratio range.

## A/A noise on m75r14

Same arm A binary, n = 12, ABBA order. Per-seed later/earlier outside-per-call ratios have median 0.992992, type-7 IQR 0.983252 to 1.014175, and range 0.963611 to 1.062112. Two-sided Mann-Whitney p of replica 2 against replica 1 on outside time per call is 7.074540e-01.
