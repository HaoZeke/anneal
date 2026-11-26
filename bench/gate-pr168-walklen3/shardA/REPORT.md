# PR 168 walklen3 out-of-sample gate — shard A

## Measurement

- A: revision `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec`, tree `e5ebda63f9f0117226d50c13abf33243ccf452cc`.
- M: candidate `4ab2681ae6bf72d61c633197e88047f00cdd7efa`, merged tree `991d5afc6d7b7f7e6c392cb68aa351af1fc50179`.
- O: candidate `fd5449766b4f1f693de2245933afa609a93a5c6b`, original tree `491819895e46a8fccbfc2449c18868780d1a090b`, merged tree `7f2625597a167d3c7841b2266eec844997e53266`. The bundle SHA-256 was `c20089cff9ebd73d9237ca6c4cf7a270e99404863addd9a6555d5f7cb80092e3`; its prerequisite check passed and it contained three commits after `52af8e47`.
- Both merge-tree operations exited 0. M and O were built from scratch merge commits outside the delivered history.
- Runs were serial on one core. Order rotated AMO, OMA, MAO, OAM, MOA, AOM by seed.
- Every arm used its own `h2h_seed`, recommended configuration, and the same release build flags. LJ used 4,000,000 calls and TIP4P used 2,000,000; stops were reference + 0.001.
- The same RSS-only in-process `RssAnon` reader sampled every 4,096 potential calls and at process end in all arms. No GMIN executable or data were used.
- Every cell reached target n.
- Bootstrap intervals use 10,000 paired resamples and seed 20261123. Calls misses are budget + 1. Fisher and MWU tests are two-sided.

## Per-arm results

| cell | arm | hits/n (Wilson 95% CI) | median calls | median outside s | median RssAnon KiB | verdict |
|---|---|---:|---:|---:|---:|---|
| lj98 | A | 35/36 [0.858, 0.995] | 1294787.0 | 16.507876 | 144952.0 | **PASS** |
| lj98 | M | 33/36 [0.782, 0.971] | 918064.0 | 11.804829 | 24708.0 | **FIX** |
| lj98 | O | 36/36 [0.904, 1.000] | 1335241.5 | 18.218353 | 41258.0 | **FIX** |
| lj102 | A | 7/12 [0.320, 0.807] | 3124644.5 | 59.013071 | 558404.0 | **PASS** |
| lj102 | M | 12/12 [0.758, 1.000] | 16055.5 | 0.306174 | 2632.0 | **PASS** |
| lj102 | O | 12/12 [0.758, 1.000] | 16886.0 | 0.295190 | 2706.0 | **FIX** |
| lj103 | A | 7/12 [0.320, 0.807] | 3687663.5 | 59.044422 | 565206.0 | **PASS** |
| lj103 | M | 11/12 [0.646, 0.985] | 29175.0 | 0.452818 | 3176.0 | **PASS** |
| lj103 | O | 12/12 [0.758, 1.000] | 33927.0 | 0.576434 | 3388.0 | **FIX** |
| lj104 | A | 9/12 [0.468, 0.911] | 1767897.0 | 28.536345 | 247860.0 | **PASS** |
| lj104 | M | 12/12 [0.758, 1.000] | 43720.0 | 0.667812 | 3654.0 | **PASS** |
| lj104 | O | 12/12 [0.758, 1.000] | 40099.5 | 0.668932 | 3536.0 | **UNRESOLVED** |
| lj150 | A | 10/12 [0.552, 0.953] | 194556.0 | 6.867979 | 42052.0 | **PASS** |
| lj150 | M | 12/12 [0.758, 1.000] | 21346.0 | 0.698292 | 5124.0 | **PASS** |
| lj150 | O | 12/12 [0.758, 1.000] | 193787.0 | 4.493448 | 11718.0 | **FIX** |
| tip4p15 | A | 8/12 [0.391, 0.862] | 953820.0 | 1.728858 | 51196.0 | **PASS** |
| tip4p15 | M | 8/12 [0.391, 0.862] | 953820.0 | 1.790392 | 21836.0 | **PASS** |
| tip4p15 | O | 8/12 [0.391, 0.862] | 953820.0 | 1.784179 | 21836.0 | **PASS** |
| tip4p20 | A | 0/12 [0.000, 0.242] | 2000001.0 | 5.436136 | 132654.0 | **PASS** |
| tip4p20 | M | 0/12 [0.000, 0.242] | 2000001.0 | 5.581673 | 40924.0 | **PASS** |
| tip4p20 | O | 0/12 [0.000, 0.242] | 2000001.0 | 5.557533 | 40924.0 | **PASS** |

A is the reference arm and is labelled PASS. M is judged against A. O must clear both O/A and O/M to PASS; a FIX in either comparison makes O FIX.

## Pairwise results

| cell | comparison | hits candidate/base | paired hit losses | Fisher p | MWU U, p | median calls ratio (95% CI) | median outside ratio (95% CI) | median RSS ratio (95% CI) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| lj98 | O/M | 36/33 | 0 | 0.2394 | 761.5, 0.2031 | 1.454 [1.097, 2.106] | 1.543 [1.125, 2.206] | 1.670 [1.130, 2.118] | **FIX** |
| lj98 | O/A | 36/35 | 0 | 1 | 696.5, 0.5888 | 1.031 [0.765, 3.175] | 1.104 [0.771, 2.947] | 0.285 [0.199, 0.751] | **FIX** |
| lj98 | M/A | 33/35 | 3 | 0.6142 | 630.0, 0.8437 | 0.709 [0.501, 2.233] | 0.715 [0.488, 2.015] | 0.170 [0.121, 0.462] | **FIX** |
| lj102 | O/M | 12/12 | 0 | 1 | 69.0, 0.8852 | 1.052 [0.697, 2.035] | 0.964 [0.767, 2.668] | 1.028 [0.972, 1.211] | **FIX** |
| lj102 | O/A | 12/7 | 0 | 0.03727 | 0.0, 3.382e-05 | 0.005 [0.004, 0.012] | 0.005 [0.003, 0.015] | 0.005 [0.004, 0.016] | **PASS** |
| lj102 | M/A | 12/7 | 0 | 0.03727 | 0.0, 3.382e-05 | 0.005 [0.003, 0.011] | 0.005 [0.002, 0.014] | 0.005 [0.004, 0.015] | **PASS** |
| lj103 | O/M | 12/11 | 0 | 1 | 82.5, 0.5636 | 1.163 [0.693, 2.162] | 1.273 [0.869, 2.125] | 1.067 [0.936, 1.245] | **FIX** |
| lj103 | O/A | 12/7 | 0 | 0.03727 | 22.5, 0.004483 | 0.009 [0.007, 0.031] | 0.010 [0.005, 0.034] | 0.006 [0.004, 0.020] | **PASS** |
| lj103 | M/A | 11/7 | 1 | 0.155 | 27.0, 0.00961 | 0.008 [0.004, 0.028] | 0.008 [0.003, 0.025] | 0.006 [0.003, 0.018] | **PASS** |
| lj104 | O/M | 12/12 | 0 | 1 | 74.0, 0.931 | 0.917 [0.499, 3.764] | 1.002 [0.556, 3.426] | 0.968 [0.786, 1.515] | **UNRESOLVED** |
| lj104 | O/A | 12/9 | 0 | 0.2174 | 13.0, 0.0007237 | 0.023 [0.011, 0.075] | 0.023 [0.013, 0.079] | 0.014 [0.008, 0.031] | **PASS** |
| lj104 | M/A | 12/9 | 0 | 0.2174 | 1.0, 4.624e-05 | 0.025 [0.010, 0.066] | 0.023 [0.012, 0.064] | 0.015 [0.008, 0.030] | **PASS** |
| lj150 | O/M | 12/12 | 0 | 1 | 138.0, 0.0001558 | 9.078 [3.123, 20.166] | 6.435 [2.533, 21.163] | 2.287 [1.400, 3.277] | **FIX** |
| lj150 | O/A | 12/10 | 0 | 0.4783 | 61.0, 0.5443 | 0.996 [0.058, 3.616] | 0.654 [0.050, 2.385] | 0.279 [0.026, 0.782] | **UNRESOLVED** |
| lj150 | M/A | 12/10 | 0 | 0.4783 | 11.0, 0.0004763 | 0.110 [0.011, 0.312] | 0.102 [0.010, 0.322] | 0.122 [0.013, 0.312] | **PASS** |
| tip4p15 | O/M | 8/8 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 0.997 [0.993, 1.014] | 1.000 [1.000, 1.000] | **PASS** |
| tip4p15 | O/A | 8/8 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 1.032 [1.026, 1.042] | 0.427 [0.280, 0.501] | **PASS** |
| tip4p15 | M/A | 8/8 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 1.036 [1.022, 1.042] | 0.427 [0.281, 0.501] | **PASS** |
| tip4p20 | O/M | 0/0 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 0.996 [0.991, 1.004] | 1.000 [1.000, 1.000] | **PASS** |
| tip4p20 | O/A | 0/0 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 1.022 [0.988, 1.025] | 0.309 [0.305, 0.321] | **PASS** |
| tip4p20 | M/A | 0/0 | 0 | 1 | 72.0, 1 | 1.000 [1.000, 1.000] | 1.027 [0.985, 1.033] | 0.309 [0.305, 0.321] | **PASS** |

The comparison bar is no lower hit count, median calls ratio at most 1.00, and upper confidence bounds at most 1.10 for outside time and RSS. A directly observed hit-count or median-calls loss is FIX; a lower confidence bound above a resource bar is FIX; other unmet bars are UNRESOLVED. Verdicts are the direct first assignment.

## TIP4P identity

- tip4p15: **bit-identical** across A, M and O for every measured seed.
- tip4p20: **bit-identical** across A, M and O for every measured seed.

## In-sample record

`in_sample_rows.csv` retains 336 prior A-vs-M rows on seeds 2000–2011 with `sample_scope=in-sample-a-vs-m`. They are excluded from every out-of-sample statistic and verdict above.

## Builder claims for M

- LJ98 geometric 0.972 and median 1.102: measured 1.054 and 0.709.
- LJ102–LJ150 geometric 0.027–0.080: measured 0.007–0.076.
- Calls per walk up to 4.5× on LJ150: measured 3.820×.

Builder family: `gpt`.
