# Walklen4 gate, shard B

Arms are A (develop `4feb0699d4e61c9531283e82196fcb60baf66932`) and B
(merge-tree of that commit and `ddd53f2f60d53aebdf2e97f03d3ade07c2905738`).
LJ75 budget 4000000, reference -397.492331. LJ38 budget 400000,
reference -173.928427. Seeds 2120-2131. Stop at the reference plus
1e-3. Core 0. Misses count as budget + 1. Ratios use first hits only.
Bootstrap: 10000 resamples, numpy Generator seed 20261123, percentiles
2.5 and 97.5. Significant means two-sided p < 0.05.

PASS: B has at least as many hits as A, calls-to-hit are not a
significant loss for B, and the outside-time upper CI is <= 1.10.
FIX: a hit loss with Fisher p < 0.05, a significant calls-to-hit loss
(B slower), or an outside-time CI entirely above 1.10.
UNRESOLVED: anything else.

| system | hits A | hits B | Fisher p | MWU p | calls B/A | outside B/A | verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| lj75 | 12 | 12 | 1.000000 | 0.644095 | 1.0361 [1.0154, 1.0618] n=12 | 1.0167 [0.9911, 1.0440] n=12 | PASS |
| lj38 | 12 | 12 | 1.000000 | 0.930867 | 1.0031 [1.0003, 1.0072] n=12 | 1.0036 [0.9682, 1.0482] n=12 | PASS |

## Notes

- lj75: PASS. none.
- lj38: PASS. none.

Host id host-9ea4a204732fbdb8. Core 0.
Rows 48.
Arm A commit 4feb0699d4e61c9531283e82196fcb60baf66932 tree 788d36860fb10d5863299acef0d0538618267d6d binary 25e9dbe1df300e19a0ed0e0ca4947faf5797deaa0b61c0737a132a9579e51de4.
Arm B tree 2fb57f0db81b5c3d1ec78077881b2982b8b78a55 binary 65bddc86045f27a7d0d6ccb6ab57491562cb5b519590863e86bc6e52857d9ce5.
numpy 2.5.5. scipy 1.18.1.
