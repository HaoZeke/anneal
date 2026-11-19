# No-size-constants-r3 regression gate

Paired head-to-head of arm A (`dd054749d4703923f4cf4c34f50589c574ee1276`, tree `f1705d6f02f7f257f50e831c16057740283efea5`) and arm B (`33c799031109a32ff9f3e61e9cdd94774adf54c6`, tree `3fc07c20d323a7ae5b34304720da7d7656309ba4`).

## Protocol

- Panel, presets, budgets and seeds are the no-size-constants-b2 gate.
- Every supported system is n=12. That gate's measured rows are n=12 on every supported system, and this run keeps that count, fixed before any result.
- Seeds present in `rows.csv`: 2000-2011 (12 values).
- LJ and Morse use `Config::recommended`. TIP4P uses `Config::recommended_tip4p`. The slab arm is `recommended`. Conformers use `cluster_search` with `recommended=True`.
- Paper budgets are the `budget` value on each summary row.
- Runs are serial, one fresh process per arm and seed, with no warm-up and no rescue rerun.
- Even seeds run A then B. Odd seeds run B then A.
- OpenMP, OpenBLAS, MKL and Rayon threads are 1. Each process is pinned to CPU core 0.
- Both arms are release builds. Every row records both commit SHAs, both trees, and a hashed host id.
- Binary LJ is unsupported, as in the b2 gate.
- Hashed host id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

## Verdict: `B LOSES`

B loses if any system has fewer hits at two-sided Fisher exact p < 0.05, more calls at tie-corrected two-sided Mann-Whitney p < 0.05, or a median outside-potential time ratio B/A above 1.1. Otherwise the verdict is PASS.

Loss conditions:

- `lj38`: outside ratio B/A = 1.17121.
- `lj98`: outside ratio B/A = 2.15846.

## Focus

The b2 losses and the size-sensitive rows are tip4p12 (calls), tip4p20 (hits), rq3 and xld (hits and outside time), lj103, lj104 and lj98.

### tip4p12

Hits 12/12 (A) and 12/12 (B), Fisher p = 1. Median calls 41418.5 (A) and 41418.5 (B), Mann-Whitney U = 72.0, p = 1. Median time inside the potential 0.153776 s (A) and 0.154801 s (B). Median time outside the potential 0.074433 s (A) and 0.075452 s (B), ratio B/A = 1.01368. Bit-identical seeds 12/12.

### tip4p20

Hits 6/12 (A) and 6/12 (B), Fisher p = 1. Median calls 1686249.5 (A) and 1686249.5 (B), Mann-Whitney U = 72.0, p = 1. Median time inside the potential 15.073408 s (A) and 15.029179 s (B). Median time outside the potential 4.377007 s (A) and 4.299823 s (B), ratio B/A = 0.982366. Bit-identical seeds 12/12.

### rq3

Hits 12/12 (A) and 12/12 (B), Fisher p = 1. Median calls 12738.0 (A) and 12738.0 (B), Mann-Whitney U = 72.0, p = 1. Median time inside the potential 1.272537 s (A) and 1.269113 s (B). Median time outside the potential 1.028998 s (A) and 1.025146 s (B), ratio B/A = 0.996256. Bit-identical seeds 12/12.

### xld

Hits 10/12 (A) and 10/12 (B), Fisher p = 1. Median calls 19412.5 (A) and 19412.5 (B), Mann-Whitney U = 72.0, p = 1. Median time inside the potential 1.304438 s (A) and 1.303704 s (B). Median time outside the potential 1.017991 s (A) and 1.014896 s (B), ratio B/A = 0.99696. Bit-identical seeds 12/12.

### lj103

Hits 8/12 (A) and 12/12 (B), Fisher p = 0.0931677. Median calls 2581470.5 (A) and 54139.5 (B), Mann-Whitney U = 8.0, p = 0.000238624. Median time inside the potential 41.580705 s (A) and 0.864303 s (B). Median time outside the potential 43.142789 s (A) and 0.938273 s (B), ratio B/A = 0.0217481. Bit-identical seeds 0/12.

### lj104

Hits 9/12 (A) and 12/12 (B), Fisher p = 0.217391. Median calls 1136745.5 (A) and 59448.5 (B), Mann-Whitney U = 7.0, p = 0.000193656. Median time inside the potential 18.818257 s (A) and 0.967646 s (B). Median time outside the potential 19.614578 s (A) and 0.986290 s (B), ratio B/A = 0.0502835. Bit-identical seeds 0/12.

### lj98

Hits 10/12 (A) and 9/12 (B), Fisher p = 1. Median calls 1321410.0 (A) and 2561612.0 (B), Mann-Whitney U = 101.0, p = 0.0984022. Median time inside the potential 19.135656 s (A) and 37.173623 s (B). Median time outside the potential 17.888939 s (A) and 38.612642 s (B), ratio B/A = 2.15846. Bit-identical seeds 0/12.

## Any hit drop, call increase, or outside ratio above 1

- `lj38`: hits 12/12 -> 12/12 (Fisher p = 1), median calls 6357.0 -> 6357.0 (U = 72.0, p = 1), outside ratio B/A = 1.17121, gate_fail = 1.
- `lj98`: hits 10/12 -> 9/12 (Fisher p = 1), median calls 1321410.0 -> 2561612.0 (U = 101.0, p = 0.0984022), outside ratio B/A = 2.15846, gate_fail = 1.
- `m80r6`: hits 12/12 -> 12/12 (Fisher p = 1), median calls 238328.0 -> 238328.0 (U = 72.0, p = 1), outside ratio B/A = 1.00392, gate_fail = 0.
- `tip4p8`: hits 12/12 -> 12/12 (Fisher p = 1), median calls 6339.0 -> 6339.0 (U = 72.0, p = 1), outside ratio B/A = 1.04893, gate_fail = 0.
- `tip4p12`: hits 12/12 -> 12/12 (Fisher p = 1), median calls 41418.5 -> 41418.5 (U = 72.0, p = 1), outside ratio B/A = 1.01368, gate_fail = 0.
- `tip4p15`: hits 7/12 -> 7/12 (Fisher p = 1), median calls 1481619.5 -> 1481619.5 (U = 72.0, p = 1), outside ratio B/A = 1.01462, gate_fail = 0.
- `tip4p16`: hits 2/12 -> 2/12 (Fisher p = 1), median calls 2000000.0 -> 2000000.0 (U = 72.0, p = 1), outside ratio B/A = 1.00814, gate_fail = 0.
- `tip4p21`: hits 0/12 -> 0/12 (Fisher p = 1), median calls 2000000.0 -> 2000000.0 (U = 72.0, p = 1), outside ratio B/A = 1.01192, gate_fail = 0.
- `cuh2_fcc_slab`: hits 0/12 -> 0/12 (Fisher p = 1), median calls 2500.0 -> 2500.0 (U = 72.0, p = 1), outside ratio B/A = 1.02029, gate_fail = 0.
- `cuh2_fcc_slab_h6`: hits 0/12 -> 0/12 (Fisher p = 1), median calls 5000.0 -> 5000.0 (U = 72.0, p = 1), outside ratio B/A = 1.00115, gate_fail = 0.
- `fluvastatin`: hits 5/12 -> 5/12 (Fisher p = 1), median calls 43258.0 -> 43258.0 (U = 72.0, p = 1), outside ratio B/A = 1.00051, gate_fail = 0.
- `z34`: hits 1/12 -> 1/12 (Fisher p = 1), median calls 46026.0 -> 46026.0 (U = 72.0, p = 1), outside ratio B/A = 1.00071, gate_fail = 0.

## Per-system summary

| Panel | System | Status | n | Hits A | Hits B | Fisher p | Med calls A | Med calls B | MW p | In (s) A | In (s) B | Out (s) A | Out (s) B | Out B/A | Bit-ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj | lj38 | measured | 12 | 12/12 | 12/12 | 1 | 6357.0 | 6357.0 | 1 | 0.014338 | 0.014462 | 0.032743 | 0.038349 | 1.17121 | 12/12 |
| lj | lj75 | measured | 12 | 12/12 | 12/12 | 1 | 44176.0 | 44176.0 | 1 | 0.376248 | 0.374369 | 0.430142 | 0.429786 | 0.999171 | 12/12 |
| lj | lj98 | measured | 12 | 10/12 | 9/12 | 1 | 1321410.0 | 2561612.0 | 0.0984022 | 19.135656 | 37.173623 | 17.888939 | 38.612642 | 2.15846 | 0/12 |
| lj | lj103 | measured | 12 | 8/12 | 12/12 | 0.0931677 | 2581470.5 | 54139.5 | 0.000238624 | 41.580705 | 0.864303 | 43.142789 | 0.938273 | 0.0217481 | 0/12 |
| lj | lj104 | measured | 12 | 9/12 | 12/12 | 0.217391 | 1136745.5 | 59448.5 | 0.000193656 | 18.818257 | 0.967646 | 19.614578 | 0.986290 | 0.0502835 | 0/12 |
| binary-lj | binary_lj | unsupported | 0 | - | - | - | - | - | - | - | - | - | - | - | - |
| morse | m75r3 | measured | 12 | 11/12 | 11/12 | 1 | 767869.5 | 767869.5 | 1 | 18.333296 | 18.226869 | 7.092081 | 6.994411 | 0.986228 | 12/12 |
| morse | m75r6 | measured | 12 | 12/12 | 12/12 | 1 | 39942.5 | 39942.5 | 1 | 0.943980 | 0.943900 | 0.448230 | 0.447198 | 0.9977 | 12/12 |
| morse | m75r10 | measured | 12 | 12/12 | 12/12 | 1 | 34286.5 | 34286.5 | 1 | 0.810053 | 0.812205 | 0.348525 | 0.345067 | 0.99008 | 12/12 |
| morse | m75r14 | measured | 12 | 12/12 | 12/12 | 1 | 36120.5 | 36120.5 | 1 | 0.855727 | 0.852496 | 0.352850 | 0.351052 | 0.994906 | 12/12 |
| morse | m80r3 | measured | 12 | 9/12 | 9/12 | 1 | 1897020.0 | 1897020.0 | 1 | 51.107558 | 50.952511 | 19.944885 | 19.938847 | 0.999697 | 12/12 |
| morse | m80r6 | measured | 12 | 12/12 | 12/12 | 1 | 238328.0 | 238328.0 | 1 | 6.415033 | 6.484892 | 2.528199 | 2.538105 | 1.00392 | 12/12 |
| morse | m80r10 | measured | 12 | 3/12 | 3/12 | 1 | 4000000.0 | 4000000.0 | 1 | 106.551387 | 106.169298 | 51.216455 | 49.673557 | 0.969875 | 12/12 |
| morse | m80r14 | measured | 12 | 8/12 | 8/12 | 1 | 3477864.0 | 3477864.0 | 1 | 93.246670 | 93.160963 | 42.635044 | 42.131775 | 0.988196 | 12/12 |
| tip4p | tip4p8 | measured | 12 | 12/12 | 12/12 | 1 | 6339.0 | 6339.0 | 1 | 0.012403 | 0.012362 | 0.008237 | 0.008639 | 1.04893 | 12/12 |
| tip4p | tip4p12 | measured | 12 | 12/12 | 12/12 | 1 | 41418.5 | 41418.5 | 1 | 0.153776 | 0.154801 | 0.074433 | 0.075452 | 1.01368 | 12/12 |
| tip4p | tip4p15 | measured | 12 | 7/12 | 7/12 | 1 | 1481619.5 | 1481619.5 | 1 | 7.992353 | 7.972300 | 2.911384 | 2.953934 | 1.01462 | 12/12 |
| tip4p | tip4p16 | measured | 12 | 2/12 | 2/12 | 1 | 2000000.0 | 2000000.0 | 1 | 12.091662 | 12.078309 | 4.437927 | 4.474032 | 1.00814 | 12/12 |
| tip4p | tip4p20 | measured | 12 | 6/12 | 6/12 | 1 | 1686249.5 | 1686249.5 | 1 | 15.073408 | 15.029179 | 4.377007 | 4.299823 | 0.982366 | 12/12 |
| tip4p | tip4p21 | measured | 12 | 0/12 | 0/12 | 1 | 2000000.0 | 2000000.0 | 1 | 19.495351 | 19.514339 | 6.358723 | 6.434537 | 1.01192 | 12/12 |
| tip4p | tip4p25 | measured | 12 | 1/12 | 1/12 | 1 | 2000000.0 | 2000000.0 | 1 | 26.757451 | 26.759287 | 8.607924 | 8.543888 | 0.992561 | 12/12 |
| slab | cuh2_fcc_slab | measured | 12 | 0/12 | 0/12 | 1 | 2500.0 | 2500.0 | 1 | 1.408426 | 1.436262 | 129.249093 | 131.871436 | 1.02029 | 12/12 |
| slab | cuh2_fcc_slab_h6 | measured | 12 | 0/12 | 0/12 | 1 | 5000.0 | 5000.0 | 1 | 1.956018 | 1.958542 | 339.528911 | 339.920449 | 1.00115 | 12/12 |
| conformer | rq3 | measured | 12 | 12/12 | 12/12 | 1 | 12738.0 | 12738.0 | 1 | 1.272537 | 1.269113 | 1.028998 | 1.025146 | 0.996256 | 12/12 |
| conformer | 915 | measured | 12 | 7/12 | 7/12 | 1 | 20330.0 | 20330.0 | 1 | 0.530334 | 0.532008 | 0.477232 | 0.476180 | 0.997796 | 12/12 |
| conformer | xld | measured | 12 | 10/12 | 10/12 | 1 | 19412.5 | 19412.5 | 1 | 1.304438 | 1.303704 | 1.017991 | 1.014896 | 0.99696 | 12/12 |
| conformer | fluvastatin | measured | 12 | 5/12 | 5/12 | 1 | 43258.0 | 43258.0 | 1 | 1.253436 | 1.253220 | 0.975658 | 0.976158 | 1.00051 | 12/12 |
| conformer | z34 | measured | 12 | 1/12 | 1/12 | 1 | 46026.0 | 46026.0 | 1 | 1.567150 | 1.566108 | 1.213475 | 1.214333 | 1.00071 | 12/12 |
| conformer | amprenavir | measured | 12 | 1/12 | 1/12 | 1 | 39154.0 | 39154.0 | 1 | 1.583912 | 1.574151 | 1.290192 | 1.264282 | 0.979918 | 12/12 |
| conformer | lopinavir | measured | 12 | 1/12 | 1/12 | 1 | 44336.0 | 44336.0 | 1 | 2.859217 | 2.859647 | 2.333083 | 2.327311 | 0.997526 | 12/12 |
| conformer | lorlatinib | measured | 12 | 4/12 | 4/12 | 1 | 43676.0 | 43676.0 | 1 | 1.039169 | 1.038156 | 0.863356 | 0.859855 | 0.995945 | 12/12 |

Per-seed bit identity (trace hash, charged count, best energy, hit and calls) is in `identity.csv`.
