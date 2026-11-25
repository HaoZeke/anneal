# PR 211 r-next8 gate

Rules and arm definitions are in `PROTOCOL.md`. GMIN was not run.

## Per-cell results

| Cell | n | Hits A/B/A2/B2 | Physical A/B | Fisher p | Mann-Whitney p | Outside B/A | Outside A2/A | RSS B/A | RSS A2/A | Time >1.10 | Time > noise bar | Real time regression | Memory failure | Verdict |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sw10 | 12 | 12/12/12/12 | 12/12 | 1 | 1 | 0.9781 [0.9357, 1.0168] n=12 | 0.8476 [0.6932, 0.9570] n=12 | 1.0009 [0.9984, 1.0034] n=12 | 1.0000 [0.9975, 1.0025] n=12 | False | False | False | False | PASS |
| sw13 | 12 | 12/12/12/12 | 12/12 | 1 | 1 | 0.9888 [0.9693, 1.0058] n=12 | 0.9812 [0.9650, 0.9958] n=12 | 1.0009 [0.9963, 1.0048] n=12 | 0.9985 [0.9948, 1.0024] n=12 | False | False | False | False | PASS |
| sw20 | 12 | 12/12/12/12 | 12/12 | 1 | 1 | 0.9978 [0.9692, 1.0233] n=12 | 0.9906 [0.9571, 1.0168] n=12 | 0.9999 [0.9983, 1.0014] n=12 | 0.9993 [0.9977, 1.0007] n=12 | False | False | False | False | PASS |
| tc10 | 12 | 12/12/12/12 | 12/12 | 1 | 1 | 0.9941 [0.9630, 1.0268] n=12 | 0.9765 [0.9354, 1.0061] n=12 | 0.9985 [0.9957, 1.0012] n=12 | 0.9979 [0.9943, 1.0021] n=12 | False | False | False | False | PASS |
| tc20 | 12 | 3/3/3/3 | 3/3 | 1 | 1 | 1.0119 [0.9903, 1.0440] n=12 | 1.0043 [0.9938, 1.0149] n=12 | 0.9999 [0.9996, 1.0002] n=12 | 1.0002 [1.0000, 1.0003] n=12 | False | False | False | False | PASS |
| gfn2-si10 | 5 | 5/5/5/5 | 0/0 | 1 | 0.09524 | 3.7486 [2.5392, 5.3447] n=5 | 2.2627 [1.5504, 3.2042] n=6 | 1.0171 [1.0043, 1.0302] n=5 | 1.0007 [0.9986, 1.0029] n=6 | True | False | False | False | FIX |
| gfn2-si13 | 0 | 0/0/0/0 | 0/0 |  |  |  |  |  |  | False | False | False | False | PARTIAL |
| gfn2-c10 | 0 | 0/0/0/0 | 0/0 |  |  |  |  |  |  | False | False | False | False | PARTIAL |

Hit losses, expected non-physical rejections, and B hits that are not physical:

- sw10: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 12/12; B2 hit or energy mismatches none; B2 same-energy call differences none
- sw13: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 12/12; B2 hit or energy mismatches none; B2 same-energy call differences none
- sw20: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 12/12; B2 hit or energy mismatches none; B2 same-energy call differences none
- tc10: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 12/12; B2 hit or energy mismatches none; B2 same-energy call differences none
- tc20: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 12/12; B2 hit or energy mismatches none; B2 same-energy call differences none
- gfn2-si10: censored rows 2; losses none; expected rejections ['2003']; B non-physical ['2000', '2001', '2002', '2004', '2005']; energy agree 0/5; B2 hit or energy mismatches none; B2 same-energy call differences ['2002', '2003']
  gfn2-si10 outside-time lower bound is above 1.10. It is not a time failure: that bound does not also exceed the A2/A upper bound on at least six pairs.
- gfn2-si13: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 0/0; B2 hit or energy mismatches none; B2 same-energy call differences none
- gfn2-c10: censored rows 0; losses none; expected rejections none; B non-physical none; energy agree 0/0; B2 hit or energy mismatches none; B2 same-energy call differences none
No rows were recorded for gfn2-si13, gfn2-c10. Those cells were not started.

## Slab call accounting

Verdict: PARTIAL. Ledger pattern 1 then 2: True.
Grad-only engine calls are the budget-2 total minus the combined total.
The calls column is the engine count.
Grad-only counts are A 1, B 0, B2 0, A2 1.
The FIX rule requires B's grad-only count to be strictly greater than A's.

| Arm | Combined engine calls | Budget-2 engine calls | Grad-only engine calls |
| --- | ---: | ---: | ---: |
| A | 1 | 2 | 1 |
| B | 1 | 1 | 0 |
| B2 | 1 | 1 | 0 |
| A2 | 1 | 2 | 1 |

## GFN2 best geometries

Si10 seed 2000, 300 K. Arm A best -541.20924786 eV, detached true, calls 123. Arm B best -779.70709560 eV, detached false, calls 1457, fresh SCC false.
The arms recorded different best energies, so B did not keep A's geometry.
The earlier false hit to compare is -475.186 eV with a 4.43 A separation. Arm A matches that point: False. Arm B matches that point: False.

| Cell | Seed | Arm | Hit | Physical | Best | Calls | Censored | Detached | NN min | NN max | SCC | Force | Fresh eV |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| gfn2-si10 | 2000 | A | true | false | -541.20924786 | 123 | false | true | 1.745804 | 3.408542 | false |  |  |
| gfn2-si10 | 2000 | B | true | false | -779.70709560 | 1457 | false | false | 1.682095 | 2.706696 | false |  |  |
| gfn2-si10 | 2000 | B2 | true | false | -779.70709560 | 1457 | false | false | 1.682095 | 2.706696 | false |  |  |
| gfn2-si10 | 2000 | A2 | true | false | -541.20924786 | 123 | false | true | 1.745804 | 3.408542 | false |  |  |
| gfn2-si10 | 2001 | A2 | true | false | -723.70246534 | 442 | false | true | 2.076765 | 2.981089 | true | 5.70599829e+00 | -453.00446891 |
| gfn2-si10 | 2001 | B2 | true | false | -676.82765186 | 1004 | false | false | 1.734372 | 2.287765 | false |  |  |
| gfn2-si10 | 2001 | B | true | false | -676.82765186 | 1004 | false | false | 1.734372 | 2.287765 | false |  |  |
| gfn2-si10 | 2001 | A | true | false | -723.70246534 | 442 | false | true | 2.076765 | 2.981089 | true | 5.70599829e+00 | -453.00446891 |
| gfn2-si10 | 2002 | A | true | false | -931.74600907 | 609 | false | true | 1.766269 | 3.185276 | false |  |  |
| gfn2-si10 | 2002 | B | true | false | -936.36646735 | 2095 | false | false | 1.594877 | 2.229428 | false |  |  |
| gfn2-si10 | 2002 | B2 | true | false | -936.36646735 | 2074 | false | false | 1.594877 | 2.229428 | false |  |  |
| gfn2-si10 | 2002 | A2 | true | false | -931.74600907 | 609 | false | true | 1.766269 | 3.185276 | false |  |  |
| gfn2-si10 | 2003 | A2 | true | false | -893.71325752 | 915 | false | true | 1.578488 | 2.283852 | false |  |  |
| gfn2-si10 | 2003 | B2 | false | false | -459.11871885 | 2382 | true | false | 2.228941 | 2.228941 | true | 3.93069268e-07 | -459.11869867 |
| gfn2-si10 | 2003 | B | false | false | -459.11871885 | 2396 | true | false | 2.228941 | 2.228941 | true | 3.93069268e-07 | -459.11869867 |
| gfn2-si10 | 2003 | A | true | false | -893.71325752 | 915 | false | true | 1.578488 | 2.283852 | false |  |  |
| gfn2-si10 | 2004 | A | true | false | -949.54524982 | 445 | false | false | 1.562411 | 2.426248 | false |  |  |
| gfn2-si10 | 2004 | B | true | false | -718.97114375 | 684 | false | false | 1.763729 | 2.478977 | false |  |  |
| gfn2-si10 | 2004 | B2 | true | false | -718.97114375 | 684 | false | false | 1.763729 | 2.478977 | false |  |  |
| gfn2-si10 | 2004 | A2 | true | false | -949.54524982 | 445 | false | false | 1.562411 | 2.426248 | false |  |  |
| gfn2-si10 | 2005 | A2 | true | false | -944.69732213 | 527 | false | false | 1.586447 | 1.728974 | true | 1.59173566e+02 | -302.99322771 |
| gfn2-si10 | 2005 | B2 | true | false | -544.80779298 | 419 | false | false | 1.761646 | 2.571096 | false |  |  |
| gfn2-si10 | 2005 | B | true | false | -544.80779298 | 419 | false | false | 1.761646 | 2.571096 | false |  |  |
| gfn2-si10 | 2005 | A | true | false | -944.69732213 | 527 | false | false | 1.586447 | 1.728974 | true | 1.59173566e+02 | -302.99322771 |

A physical hit is an energy hit that is connected and, for GFN2,
whose fresh 300 K single point converged with maximum force below 1e-3 eV/A.
B rejecting a non-physical A hit is listed above and is not a hit loss.
