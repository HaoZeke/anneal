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

## Cage walk

`scripts/tip4p_h2h/enter_cage.sh` appends the `anneal-cage` arm. The budget, seeds, core, and hostname are the same. The five misses on the previous tip are different oxygen cages, not proton arrangements of the global-minimum cage. (H₂O)₁₅ misses have 24 oxygen contacts against 25 in the global minimum. (H₂O)₂₀ misses differ by one contact, including a five-coordinate water.

The energy archive was keeping proton isomers of the lowest cage, so a stall had no other cage to walk. The cage archive keeps one structure per oxygen framework. After 160 patience windows with no new lowest minimum, the calls that remain are spent on another stored cage. That walk uses its own generator. The main walker is not moved, and a seed that is still descending keeps the same proposals and the same evaluation count.

(H₂O)₁₅ seed 2002, previously −623.022 kJ/mol, now reaches −628.372. Seed 2004 stays at −623.877. The three (H₂O)₂₀ misses stay at −864.746, −859.404 and −863.975. Every previous hit keeps its evaluation count.

The descriptor of a rigid minimum is reused when the centres and rotations are unchanged. A full-budget (H₂O)₁₅ miss drops from 5.90 s outside the potential to 4.57 s. The three (H₂O)₂₀ misses drop from 9.58, 8.79 and 9.26 s to 6.37, 8.87 and 5.72 s. GMIN's full-budget outside time stays near 3.4 s at 15 molecules and 3.6 s at 20.

| Size | Arm | Hits | Median evals | Median wall (s) | Ratio | Inside (s) | Outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₁₅ | anneal-cage | 5/6 | 988081 | 7.679 | 5.236 | 5.279 | 2.400 |
| (H₂O)₁₅ | GMIN, fixed step | 0/6 | 2000000 | 40.212 | 5.236 | 36.866 | 3.359 |
| (H₂O)₂₀ | anneal-cage | 3/6 | 1396032 | 16.656 | 4.101 | 12.364 | 4.291 |
| (H₂O)₂₀ | GMIN, fixed step | 0/6 | 2000000 | 68.305 | 4.101 | 64.713 | 3.580 |

LJ38 at 400_000 calls is 6/6, LJ75 at 4_000_000 is 6/6, LJ98 at 4_000_000 is 5/6, and LJ104 at 4_000_000 is 6/6. Each seed spends the same number of evaluations as on `4fdc37e8` and on `91c8a9c3`. The medians are 10_190, 979_291, 1_387_169 and 1_812_258 calls.

The new rows are in `rows.csv`. The medians of those rows are `summary-cage.csv`.

## Shell turn

`scripts/tip4p_h2h/enter_shell.sh` appends the `anneal-shell` arm and, for (H₂O)₂₁, a fixed-step GMIN arm in the same session. (H₂O)₂₁ is the next Wales–Hodges size. Its reference is the Cambridge header, −916.7074302940 kJ/mol. Seeds, budget, core, and hostname match the earlier arms.

A stalled rigid chain turns the outer half of the cage about the core by three orientation steps, on a generator of its own, and also walks a stored cage before turning that shell. One molecule moved into a hollow quenches back into the same oxygen cage. The shell turn does not. The main walker is left where it is.

(H₂O)₁₅ is 6/6. Seeds 2002 and 2004, the two one-contact misses, reach −628.372. The three (H₂O)₂₀ misses stay one contact off, at −864.746, −859.404 and −863.975. (H₂O)₂₁ is 0/6 on both arms. Anneal's six bests lie between −915.230 and −907.019. GMIN's lie between −903.037 and −893.409.

A full-budget call on (H₂O)₂₁ costs 9.68 µs inside the potential and 3.55 µs outside it for anneal, and 35.67 µs inside and 1.92 µs outside for GMIN. On (H₂O)₂₀ the three full-budget anneal misses are 8.85 µs inside and about 3.1 µs outside. On (H₂O)₁₅, where every seed hits, a call is 5.33 µs inside.

| Size | Arm | Hits | Median evals | Median wall (s) | Ratio | Inside (s) | Outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₁₅ | anneal-shell | 6/6 | 1158631 | 8.140 | 4.941 | 6.180 | 1.960 |
| (H₂O)₁₅ | GMIN, fixed step | 0/6 | 2000000 | 40.212 | 4.941 | 36.866 | 3.359 |
| (H₂O)₂₀ | anneal-shell | 3/6 | 1696341 | 19.749 | 3.459 | 15.022 | 4.728 |
| (H₂O)₂₀ | GMIN, fixed step | 0/6 | 2000000 | 68.305 | 3.459 | 64.713 | 3.580 |
| (H₂O)₂₁ | anneal-shell | 0/6 | 2000000 | 26.470 | 2.842 | 19.366 | 7.093 |
| (H₂O)₂₁ | GMIN, fixed step | 0/6 | 2000000 | 75.215 | 2.842 | 71.339 | 3.835 |

LJ38, LJ75, LJ98 and LJ104 keep the same evaluation counts as on `4fdc37e8`, including on the outside-cut at `2d00f044`. LJ38 is 6/6, LJ75 is 6/6, LJ98 is 5/6, and LJ104 is 6/6.

The new rows are in `rows.csv`. The medians of those rows are `summary-shell.csv`.

## Container radius

`examples/h2h_gmin_tip4p.data` sets no `RADIUS`. For `NTIP`, `io1.f90` then applies the Lennard-Jones container, \((2 + (3N/17.77)^{1/3})\,2^{1/6}\) with `NATOMS` counting the centre and angle-axis rows, two per water. GMIN prints `io1> Container radius=        4.1724973071` for (H₂O)₁₅, seed 2000. After every step `takestep.f90` moves any molecular centre beyond the radius one diameter toward the origin. The quench is not confined, because `NEWTIP` does not call `RAD`, and a hit is scored on the quenched energy.

The (H₂O)₂₀ Wales–Hodges minimum has centres of mass 4.61 Å from its centroid, outside the 4.37 Å container. At 15 molecules the margin is 0.37 Å. On seed 2000, over 100_000 calls with `DEBUG`, every (H₂O)₁₅ and (H₂O)₂₀ step moved centres, 2140 times in 453 steps and 1718 times in 427. A quench took 221 and 234 calls. With the radius below it took 64 and 106, and no centre was moved.

`examples/h2h_gmin_tip4p_radius<n>.data` add `RADIUS`: the largest oxygen distance from the oxygen centroid of the Wales–Hodges minimum, plus 3 Å, at every size. `scripts/tip4p_h2h/gm_radius.py` prints it from the Cambridge Cluster Database files, `radius/TIP4P-4.xyz`, `radius/TIP4P-8.xyz` and the 15 and 20 fixtures. With `--gmin` it writes each minimum as a NEWTIP block; GMIN scores all four as hits within two calls.

| Size | LJ container (Å) | O extent (Å) | Centre-of-mass extent (Å) | `RADIUS` (Å) |
| --- | --- | --- | --- | --- |
| (H₂O)₄ | 3.4856 | 1.9229 | 1.9257 | 4.9229 |
| (H₂O)₈ | 3.8081 | 2.4544 | 2.4791 | 5.4544 |
| (H₂O)₁₅ | 4.1725 | 3.7744 | 3.8031 | 6.7744 |
| (H₂O)₂₀ | 4.3665 | 4.5769 | 4.6099 | 7.5769 |

`scripts/tip4p_h2h/enter_radius.sh` appends two fixed-step GMIN arms in one session. `gmin-radius` reads the radius file. `gmin-control` reads the unchanged data file. Seeds are 2000–2005, serial, `taskset -c 0`, hostname `bench` under `sudo unshare --uts`, working directory `/work`. The budget is 2_000_000 calls; 4 and 8 are not in `PAPER_BUDGETS` and use the (H₂O)₂₀ ceiling. References are the Cambridge headers, −116.5904186160 and −305.5183221676 kJ/mol for 4 and 8. A seed's two arms run back to back from the start `h2h_seed` writes for it, at `3cab6d7e`. That start is byte-identical to the one `1ba7417e` writes.

GMIN is `scripts/gmin_h2h/build_gmin.sh` at revision `aac1548a`, gfortran 13.3.0, Ubuntu reference BLAS and LAPACK 3.12.0. This build does not time `NEWTIP`, so the inside and outside cells are blank. It does not reproduce the earlier GMIN rows: (H₂O)₁₅ seed 2000 ends at −619.701 kJ/mol here and −621.166 in the fixed-step row. The control arm is the comparison for the radius.

| Size | GMIN, fixed step | `gmin-control` | `gmin-radius` | anneal-shell | Control best, median | Radius best, median |
| --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₄ | | 6/6 | 6/6 | | −116.591 | −116.591 |
| (H₂O)₈ | | 1/6 | 6/6 | | −297.794 | −305.519 |
| (H₂O)₁₅ | 0/6 | 1/6 | 0/6 | 6/6 | −622.349 | −613.067 |
| (H₂O)₂₀ | 0/6 | 0/6 | 0/6 | 3/6 | −853.324 | −838.325 |

At 8 molecules four control seeds stop at −297.794 kJ/mol and the radius arm hits all six, median 367_389.5 calls. At 15 and 20 the control's lowest energy is below the radius arm's on all twelve seeds, by 0.11 to 20.72 kJ/mol. The (H₂O)₁₅ radius-arm minima are compact cages with 24 or 25 oxygen contacts, against 25 in the global minimum, and no molecule apart. Moving a centre by one diameter places a surface water on the far side of the cage; that may be the move the control is using. The earlier 0/6 at 15 and 20 is not the container.

`h2h_seed` writes each angle-axis vector negated. NEWTIP's site placement is the Rodrigues map of the vector as written: a Wales–Hodges minimum written without the negation quenches in zero steps, at −628.3735 kJ/mol for (H₂O)₁₅. The GMIN arms therefore start from anneal's centres with each orientation inverted. On (H₂O)₁₅ seed 2000, on the GminTip4p constants, anneal's start is −13.193 kJ/mol and GMIN's is −5.754. These rows keep the existing start.

The new rows are in `rows.csv`. The medians of those rows are `summary-radius.csv`.
