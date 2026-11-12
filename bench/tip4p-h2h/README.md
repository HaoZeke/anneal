# TIP4P head-to-head

Six seeds, 2000–2005, serial, one core, hostname `bench`, working directory `/work`. The driver is `scripts/tip4p_h2h/enter_bench.sh`. Both sizes use a budget of 2_000_000 charged evaluations, the paper ceiling for (H₂O)₂₀. A hit is a quenched energy at or below the GminTip4p header plus 0.001 kJ/mol. A miss is the full budget. Medians are Hyndman–Fan type 7. The wall ratio is the GMIN median wall divided by the anneal median wall. `pot_s` is time inside the potential. `outside_s` is the rest of the wall.

The existing rate table stops three (H₂O)₁₅ seeds on one minimum, −623.877 kJ/mol, 4.5 kJ/mol above the Wales–Hodges value. Cycle reversal is accepted far more often than translation or rotation and does not move the oxygen centres. On a chart of more than ten bodies those new proton arrangements were counted as discoveries, which shrank the kick, and an 80-step quench often stopped short of the record tolerance.

The rigid chart now proposes a centre step and a rotation together, accepts that quench by Metropolis at the preset temperature, and scales the step toward a one-half acceptance. A stalled chain relocates one body once, then returns to the library. A rigid quench continues until the root-mean-square force is at the record tolerance. The lengths are the preset step and the preset minimum separation.

GMIN is NEWTIP with TIPID 4, temperature 2 kJ/mol, and step 0.40 / 0.51, with the step and the temperature held fixed. Both arms open the same rigid start. The Lennard-Jones kernel already times each evaluation. The same pair of calls was added around NEWTIP in the local GMIN build, which is what fills `pot_s` for that arm.

| Size | Arm | Hits | Median evals | Median wall (s) | Ratio | Inside (s) | Outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₁₅ | anneal | 4/6 | 988080.5 | 8.035 | 5.004 | 5.564 | 2.471 |
| (H₂O)₁₅ | GMIN | 0/6 | 2000000 | 40.212 | 5.004 | 36.866 | 3.359 |
| (H₂O)₂₀ | anneal | 3/6 | 1396031.5 | 18.768 | 3.639 | 12.896 | 5.851 |
| (H₂O)₂₀ | GMIN | 0/6 | 2000000 | 68.305 | 3.639 | 64.713 | 3.580 |

On the same binary, LJ38 at 400_000 calls is 6/6 and LJ98 at 4_000_000 calls is 5/6, seeds 2000–2005. The library tests passed, 1163 of them, and `tests/hop_stop.rs` passed.

Per-seed rows are `rows.csv`.

## Follow-up

`scripts/tip4p_h2h/enter_followup.sh` appends three comparisons. The budget, seeds, core, and hostname are the same. The develop arm is `4fdc37e8`. The adaptive GMIN arm is `examples/h2h_gmin_tip4p_accrat.data`: `ACCEPTRATIO 0.5`, temperature held at 2 kJ/mol, step free to move from 0.40 / 0.51.

A GMIN call is one entry to `POTENTIAL`. The L-BFGS quench asks for the gradient on that entry, so the call is one energy and one gradient, and `NPCALL` advances by one. Anneal's `calls` column equals `charged`, and each charge is one `value_and_gradient`. On a full (H₂O)₁₅ budget the potential time is 5.64 µs per anneal call and 18.3 µs per GMIN call. The adaptive arm has the same 18.3 µs, so the gap is the kernel. `NEWTIP` builds the rotation matrix and its three derivatives for every molecule before the pair loop. The fixed-step and adaptive arms spend about 3.4 s outside the potential; a full-budget anneal miss spends about 6 s outside it.

On seed 2000 the adaptive step finished at an acceptance ratio of 0.016 and a step of 1e-5. The best energy did not move after that collapse. The fixed step is the stronger of the two GMIN arms.

| Size | Arm | Hits | Median evals | Median wall (s) | Inside (s) | Outside (s) |
| --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₁₅ | anneal, this tip | 4/6 | 988081 | 8.035 | 5.564 | 2.471 |
| (H₂O)₁₅ | anneal, 4fdc37e8 | 0/6 | 2000000 | 18.319 | | |
| (H₂O)₁₅ | GMIN, adaptive step | 0/6 | 2000000 | 40.015 | 36.657 | 3.362 |
| (H₂O)₂₀ | anneal, this tip | 3/6 | 1396032 | 18.768 | 12.896 | 5.851 |
| (H₂O)₂₀ | anneal, 4fdc37e8 | 0/6 | 2000000 | 34.675 | | |
| (H₂O)₂₀ | GMIN, adaptive step | 0/6 | 2000000 | 68.284 | 64.709 | 3.575 |

LJ75 and LJ104 at 4_000_000 calls, seeds 2000–2005, are 6/6 on this tip and on `4fdc37e8`. Each seed spends the same number of evaluations on both binaries. The medians are 979291 calls at 75 atoms and 1_812_258 at 104.

The new rows are in `rows.csv`. The medians of those rows are `summary-followup.csv`. The 4fdc37e8 water binary does not time the potential, so those inside and outside cells are blank.
