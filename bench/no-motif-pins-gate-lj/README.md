# No-motif-pin LJ no-loss gate

Arm A is revision `ef1e1d07ad214d545cfa0bcb0c2e15b8be3bcbcb` (tree `8d9182531acda5cb8f265d70605f06e57714e334`).
Arm B is the six-patch candidate at revision `b5037da1c6d8ea35b87a75171929ca7c4e264fa6` (tree `b1b89bd25f04bede362c41e65b0eed914d438489`).
The input patch set checksum is `0d96ae7e73d1c10c6e17229fa2bf4617048bbdb5`.

Each panel used seeds 2000-2011 at the paper budget. The 120 jobs ran serially on one core in alternating A/B order, without prewarming, retries, rescue runs, or parameter changes. Misses enter the call test at the budget. Fisher is two-sided exact; Mann-Whitney is two-sided with tie and continuity corrections. Times are type-7 medians in seconds.

| panel | hits A/B (Fisher p) | calls A/B (U_B, p) | wall A/B | inside A/B | outside A/B | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 12/12 vs 10/12 (0.478261) | 127,839.5 vs 141,995.5 (71.0, 0.976965) | 0.709049 / 0.764765 | 0.330433 / 0.353116 | 0.378616 / 0.411650 | PASS |
| LJ75 | 5/12 vs 0/12 (0.0372671) | 4,000,000.0 vs 4,000,000.0 (102.0, 0.0164732) | 59.677097 / 112.271766 | 34.466513 / 36.531184 | 25.406937 / 75.882407 | FIX |
| LJ98 | 7/12 vs 11/12 (0.154953) | 3,166,811.5 vs 2,506,645.5 (55.5, 0.351921) | 77.830039 / 100.137722 | 46.824246 / 36.762425 | 30.652822 / 63.375298 | PASS |
| LJ103 | 6/12 vs 8/12 (0.680172) | 3,905,072.5 vs 2,646,501.0 (67.0, 0.787422) | 106.026737 / 136.333808 | 65.497784 / 46.611787 | 39.657648 / 88.749534 | PASS |
| LJ104 | 5/12 vs 7/12 (0.684273) | 4,000,000.0 vs 3,316,075.5 (74.5, 0.901793) | 104.419036 / 169.674070 | 65.378209 / 54.634146 | 38.647223 / 112.899141 | PASS |

Verdict: **FIX — LJ75**.

`rows.csv` contains the 120 per-seed records; `summary.csv` is derived from those records.
