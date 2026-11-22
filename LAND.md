# Landing note: anneal #179 r8 no-hit-loss gate kit

- Target repo: HaoZeke/anneal
- PR: #179 (record lane; harness and data only, no source change)
- Base: cb0cfbfa0cfae21b62cc32037defc5327dc52a51
- Payload: 3 commits on base, tree 4160e5ba07953278059f87e787349005d104e619
  (payload HEAD 1a16c7c1 in the kit; differs here by committer date only)
- Builder family of the gate harness: unverified (not stated in transcript)
- Arm B: base + 19 commits, tree dd97e61fbe57f633fa131794918d647a3a20b6c3
- Kit sha256: mbox ba48965d611a92ea55b159a466058b45b4efeaca57460e9fa2fcf1ca7c25ca05

## Verdict: FIX

n=12 per arm per cell, 8 cells, 384 rows, serial, taskset -c 0.
Hit regressions (B < A): lj38 11/12, lj104 2/5, blj130n38 7/11
(Fisher p 1, 0.371, 0.155: not significant, but the bar is B >= A).
Outside-us B/A > 1.10 in all 8 cells. AA/A and AP/A stay near 1.

| cell | hits A/B/AA/AP | Fisher | B/A (p) | AA/A (p) | AP/A (p) |
|---|---|---|---|---|---|
| lj38 | 12/11/12/12 | 1 | 1.1117 (0.00426) | 0.9816 (0.707) | 0.9320 (0.583) |
| lj75 | 5/5/5/5 | 1 | 1.1850 (0.00426) | 1.0225 (0.840) | 1.0645 (0.583) |
| lj98 | 7/11/7/7 | 0.155 | 1.2078 (0.00355) | 0.9986 (0.977) | 0.9994 (0.977) |
| lj102 | 6/6/6/6 | 1 | 1.3038 (7.66e-05) | 1.0140 (0.665) | 1.0166 (0.977) |
| lj104 | 5/2/5/5 | 0.371 | 1.2378 (0.00244) | 1.0197 (0.665) | 0.9949 (0.977) |
| m80r14 | 6/8/6/6 | 0.680 | 1.2325 (0.000123) | 0.9960 (0.544) | 1.0058 (0.507) |
| sc108-80 | 0/1/0/0 | 1 | 1.1312 (0.285) | 0.9982 (0.665) | 1.0586 (0.436) |
| blj130n38 | 11/7/11/11 | 0.155 | 1.2716 (0.751) | 1.0698 (0.977) | 0.9055 (0.885) |

B has no B-only path counter (removed in commit 17); move-sym counts
differ between A and B in every cell, which shows the path ran.
