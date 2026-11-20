# b7 20k miss diagnosis

This is the diagnosis made before changing search code from b6
`53d98cd20fd7614c4cb70571dccc57056589d569`. The release `h2h_seed`
driver ran the recommended rigid preset for 20,000 calls on seeds 0–5.
The Cambridge energy appears only as the driver's stop level, with the
usual 0.001 kJ/mol tolerance. Raw logs and traces are under
`kits/tip4p-hits-b7-evidence/b6-diagnosis/`.

## Progress through the budget

The entries at 1k, 5k, 10k, and 20k are the median best-energy gap to
the stop reference in kJ/mol. Final gaps list seeds 0–5.

| Bodies | 1k | 5k | 10k | 20k | Final gaps |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 17.925 | 7.500 | 2.936 | 1.890 | 1.548, 1.890, 1.548, 5.514, 1.890, 2.207 |
| 14 | 45.944 | 24.297 | 22.164 | 15.148 | 11.896, 0.447, 23.044, 23.211, 18.400, 6.313 |
| 17 | 51.090 | 32.895 | 25.516 | 21.314 | 20.227, 23.690, 25.668, 22.400, 11.732, 9.050 |
| 21 | 77.931 | 48.339 | 42.102 | 36.048 | 31.899, 43.963, 19.909, 42.622, 38.720, 33.376 |

At 17 and 21 bodies every seed except 21/3 found its last lower raw
energy after call 15,000. Five of those twelve last improvements were
after call 19,000. Those searches are still descending when the budget
ends rather than sitting on a long final plateau. Eleven bodies is
different: five seeds stop improving between calls 3,616 and 12,889,
and their final cages are only 1.5–5.5 kJ/mol above the reference.

## Where calls go

| Bodies | Median calls/hop | Median potential µs/call | Median outside µs/call | Median basins |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 79.7 | 3.181 | 1.557 | 39.5 |
| 14 | 78.1 | 4.735 | 2.011 | 32.0 |
| 17 | 83.2 | 6.655 | 2.622 | 34.5 |
| 21 | 84.4 | 9.650 | 3.526 | 35.0 |

The budget buys only about 237–256 proposals. Nearly all calls are in
the quench, not in search bookkeeping. Across the six seeds, visit
classifications `(return, known, new)` are:

| Bodies | Return | Known | New |
| ---: | ---: | ---: | ---: |
| 11 | 818 | 604 | 98 |
| 14 | 876 | 535 | 121 |
| 17 | 769 | 552 | 158 |
| 21 | 767 | 423 | 235 |

The share of new basins rises with body count, but the run still rejects
many proposals which changed the rigid cage. More calls alone would
continue the observed descent at 17 and 21, but the task fixes the budget.

## Move outcomes

The following rows aggregate seeds 0–5. `record` means the quench was a
new run-wide best before acceptance. Percentages are per draw.

| Bodies | Move | Draws | Accept | Record |
| ---: | --- | ---: | ---: | ---: |
| 11 | rtrans | 262 | 100.0% | 15.3% |
| 11 | rrot | 241 | 100.0% | 14.9% |
| 11 | rcycle | 229 | 58.1% | 12.7% |
| 11 | rreloc | 221 | 15.4% | 13.1% |
| 11 | rreor | 199 | 56.3% | 11.6% |
| 11 | rtrans1 | 196 | 65.3% | 12.8% |
| 11 | rkick | 178 | 100.0% | 11.2% |
| 14 | rtrans | 324 | 99.4% | 26.2% |
| 14 | rcycle | 255 | 43.9% | 23.1% |
| 14 | rreor | 236 | 52.5% | 22.0% |
| 14 | rrot | 206 | 99.0% | 18.4% |
| 14 | rkick | 202 | 100.0% | 18.8% |
| 14 | rtrans1 | 183 | 62.8% | 16.9% |
| 14 | rreloc | 132 | 14.4% | 12.9% |
| 17 | rkick | 268 | 100.0% | 23.5% |
| 17 | rcycle | 249 | 42.2% | 24.9% |
| 17 | rrot | 222 | 99.1% | 27.5% |
| 17 | rreloc | 204 | 13.7% | 21.6% |
| 17 | rreor | 201 | 48.3% | 20.4% |
| 17 | rtrans1 | 171 | 62.6% | 23.4% |
| 17 | rtrans | 170 | 98.8% | 22.4% |
| 21 | rrot | 271 | 99.3% | 43.5% |
| 21 | rcycle | 244 | 45.5% | 41.0% |
| 21 | rtrans | 234 | 98.7% | 32.9% |
| 21 | rkick | 202 | 100.0% | 36.1% |
| 21 | rreloc | 174 | 13.8% | 33.3% |
| 21 | rreor | 167 | 52.1% | 35.9% |
| 21 | rtrans1 | 139 | 56.8% | 28.8% |

No move is dead: every move type produces records at every measured
size. Translation and rotation accept almost every quench. Relocation,
the move which explicitly changes a body's neighbours, accepts only
13.7–15.4% despite producing records on 12.9–33.3% of its draws.
Cycle reversal and random reorientation are accepted on roughly half
their draws.

The sparse-improvement allocator is not starving one decisive move.
Its draw shares remain broad, and the best-producing move changes with
size. A fixed arm weight would therefore be unsupported.

## Diagnosis and next experiment

The measured limitation is not lack of downhill proposals. It is that a
20k budget buys few fully quenched cage transitions, while the live chain
often refuses topology-changing destinations. The ledger keeps a rejected
record, but the following proposal still starts from the old cage and cannot
continue descending from the new one.

The supported acceptance experiment is therefore to keep the existing
energy rule for known destinations but accept a newly observed rigid basin.
Novelty is already computed from the run's basin index; it introduces no
species, size, motif, reference-energy, or fitted numeric parameter. This
tests whether chaining measured cage discoveries converts the same quench
budget into hits. If it does not improve the six-seed panel without losses,
the code change will be discarded and this diagnosis will be delivered as
the negative result.
