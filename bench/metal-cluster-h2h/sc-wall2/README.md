# Sutton-Chen wall with rgpot 14680ddb

rgpot 14680ddb20e3d8e80010ac8e6f1c07c44234319e is the head of rgpot's `feat/sc-call-overhead` branch, seven commits on its develop at 251aaade, which keep the Sutton-Chen C entry's potential between calls and read the pair powers back in the force pass. Anneal's `Cargo.lock` still pins rgpot 8efdd246; 14680ddb is not merged, so it is only recorded here and in the rows. `scripts/metal_h2h/enter_sc_wall2.sh` builds its Sutton-Chen entry with `build_rgpot_sc.sh` from a clean checkout, runs `sc_wall.sh` against it, appending to `../sc-wall/seeds.csv` and `../sc-wall/micro.csv`, and then runs anneal alone on the 55- and 80-atom sc-n12 systems, seeds 2000-2011 at the paper budgets, into `seeds.csv`. Harness 91c8dbf0, one job at a time on core 0 in a UTS namespace with hostname `bench`. GMIN is not rerun: its rows are the corrected-file reruns in `../sc-n12/seeds.csv`. Per-call times are totals over the twelve seeds divided by charged calls. Harness 91c8dbf0 was a local commit, pushed as 0e11e9ae with the same patch; `../../revision-aliases.csv` maps it to its pushed commit.

## sc_wall.sh

| System | Seeds | Same as 8efdd246 | Inside 8efdd246 (µs/call) | Inside 14680ddb (µs/call) | Outside 8efdd246 (µs/call) | Outside 14680ddb (µs/call) | Wall 8efdd246 (s) | Wall 14680ddb (s) |
|---|---|---|---|---|---|---|---|---|
| sc126-55 | 12 | 12 | 62.69 | 13.63 | 5.31 | 4.63 | 3.414 | 0.946 |
| sc96-55 | 12 | 12 | 62.01 | 14.41 | 5.20 | 4.85 | 1.917 | 0.552 |

On sc126-55 each of the 12 14680ddb rows of `sc_wall.sh` has the calls to the hit, calls charged and best energy of its 8efdd246 row. Inside the potential a call takes 13.63 µs on sc126-55 with 14680ddb, against 62.69 µs with 8efdd246. On sc96-55 each of the 12 14680ddb rows of `sc_wall.sh` has the calls to the hit, calls charged and best energy of its 8efdd246 row. Inside the potential a call takes 14.41 µs on sc96-55 with 14680ddb, against 62.01 µs with 8efdd246.

`sc_wall_micro` medians in µs per call, as in `../sc-wall/README.md`:

| System | Rgpot | Entry | Entry with dlsym and allocation | Kept | Kept, streaming | Plain |
|---|---|---|---|---|---|---|
| sc126-55 | 8efdd246 | 58.19 | 58.44 | 30.95 | 23.80 | 9.86 |
| sc126-55 | 14680ddb | 13.61 | 13.61 | 13.52 | 13.49 | 9.55 |
| sc96-55 | 8efdd246 | 59.78 | 60.12 | 32.54 | 25.23 | 10.85 |
| sc96-55 | 14680ddb | 14.36 | 14.25 | 14.05 | 14.08 | 10.22 |

## Anneal against GMIN

On sc126-55, sc108-55, sc96-55, sc126-80, sc108-80 and sc96-80 every seed matches its c89653df row of `../sc-n12/seeds.csv`, rgpot 8efdd246, in hit, calls to the hit, calls charged, best and final energy, hops, escapes, restarts and paths.

Fisher compares hits; Mann-Whitney, two-sided with the tie correction and the continuity correction, compares calls to the hit and wall. The wall ratio is GMIN's median wall over anneal's. The code ahead on wall is the one lower in more of the 144 seed pairs; n.s. marks p >= 0.05.

| System | GMIN harness | Anneal hits | GMIN hits | Fisher p | Anneal calls | GMIN calls | Mann-Whitney p calls | Anneal wall (s) | GMIN wall (s) | Mann-Whitney p wall | Wall ratio | Wall ahead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sc126-55 | a50f4b02 | 12/12 | 12/12 | 1 | 50803 | 30565.5 | 0.285 | 0.948 | 0.576 | 0.312 | 0.608 | GMIN, n.s. |
| sc108-55 | a50f4b02 | 6/12 | 2/12 | 0.193 | 511393.5 | 1000000 | 0.0424 | 9.907 | 21.165 | 0.000731 | 2.136 | anneal |
| sc96-55 | a50f4b02 | 12/12 | 12/12 | 1 | 28767 | 32261 | 0.507 | 0.549 | 0.652 | 0.47 | 1.189 | anneal, n.s. |
| sc126-80 | a50f4b02 | 6/12 | 0/12 | 0.0137 | 828059.5 | 1000000 | 0.00705 | 32.706 | 38.015 | 1 | 1.162 | tie |
| sc108-80 | a50f4b02 | 0/12 | 0/12 | 1 | 1000000 | 1000000 | 1 | 40.941 | 42.888 | 3.66e-05 | 1.048 | anneal |
| sc96-80 | a50f4b02 | 7/12 | 0/12 | 0.00458 | 981181 | 1000000 | 0.00286 | 40.285 | 40.355 | 0.312 | 1.002 | anneal, n.s. |

Anneal beats GMIN on wall time on sc108-55 and sc108-80, at Mann-Whitney p < 0.05. Anneal's lead on wall time on sc96-55 and sc96-80 is not significant. GMIN's lead on wall time on sc126-55 is not significant. Neither code leads on wall time on sc126-80.

| System | Code | Rgpot | Inside (µs/call) | Outside (µs/call) | Wall (µs/call) |
|---|---|---|---|---|---|
| sc126-55 | Anneal | 8efdd246 | 58.42 | 4.76 | 63.18 |
| sc126-55 | Anneal | 14680ddb | 13.87 | 4.74 | 18.61 |
| sc126-55 | GMIN | - | 16.72 | 2.12 | 18.84 |
| sc108-55 | Anneal | 8efdd246 | 59.15 | 6.08 | 65.23 |
| sc108-55 | Anneal | 14680ddb | 14.40 | 6.00 | 20.40 |
| sc108-55 | GMIN | - | 19.17 | 2.08 | 21.25 |
| sc96-55 | Anneal | 8efdd246 | 60.64 | 5.01 | 65.64 |
| sc96-55 | Anneal | 14680ddb | 14.31 | 4.82 | 19.13 |
| sc96-55 | GMIN | - | 17.99 | 2.15 | 20.14 |
| sc126-80 | Anneal | 8efdd246 | 115.00 | 11.56 | 126.57 |
| sc126-80 | Anneal | 14680ddb | 28.23 | 11.34 | 39.57 |
| sc126-80 | GMIN | - | 35.31 | 2.69 | 38.01 |
| sc108-80 | Anneal | 8efdd246 | 116.18 | 11.44 | 127.62 |
| sc108-80 | Anneal | 14680ddb | 29.60 | 11.41 | 41.01 |
| sc108-80 | GMIN | - | 40.21 | 2.66 | 42.87 |
| sc96-80 | Anneal | 8efdd246 | 117.04 | 11.12 | 128.16 |
| sc96-80 | Anneal | 14680ddb | 29.64 | 11.00 | 40.65 |
| sc96-80 | GMIN | - | 37.81 | 2.70 | 40.51 |

On sc126-55 anneal's wall per call is 0.99 times GMIN's with 14680ddb, against 3.35 with 8efdd246. On sc108-55 anneal's wall per call is 0.96 times GMIN's with 14680ddb, against 3.07 with 8efdd246. On sc96-55 anneal's wall per call is 0.95 times GMIN's with 14680ddb, against 3.26 with 8efdd246. On sc126-80 anneal's wall per call is 1.04 times GMIN's with 14680ddb, against 3.33 with 8efdd246. On sc108-80 anneal's wall per call is 0.96 times GMIN's with 14680ddb, against 2.98 with 8efdd246. On sc96-80 anneal's wall per call is 1.00 times GMIN's with 14680ddb, against 3.16 with 8efdd246.
