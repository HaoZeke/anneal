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
