# Sutton-Chen, twelve seeds

Anneal against GMIN on reduced Sutton-Chen 12-6, 10-8 and 9-6 at N=38, 55
and 80, seeds 2000-2011. `seeds.csv` has one row per system, arm, seed and
harness revision. `results.csv` is written from it by
`scripts/metal_h2h/run_sc_n12.py`. All rows ran one at a time on core 0 in a
UTS namespace with hostname `bench`.

The GMIN rows of harness e3ed7dad were run from a data file with
`RADIUS 30.0`, `TEMPERATURE 0.8` and `STEP 0.40 0.51`. GMIN squares
`RADIUS` and reseeds inside it, so every reseed drew a gas in a box of side
34.6 a. The temperature and step are the LJ38 tutorial values, in units of
ε and σ, not of the Sutton-Chen ε and a. From harness 2d10325c on, the file
leaves the container to GMIN, which sizes it for Sutton-Chen. It sets the
temperature of Doye and Wales (30 for 12-6, 5 for 10-8, 10 for 9-6) and
the step 0.40 σ in units of a, 0.251984. GMIN is linked against OpenBLAS,
with which the twelve e3ed7dad sc96-38 GMIN rows reproduce.

The GMIN rows rerun with the corrected file are sc126-38 and sc108-38 under harness f9d1ac05; sc96-38 under harness 9149d1b0; sc126-55, sc108-55, sc96-55, sc126-80, sc108-80 and sc96-80 under harness a50f4b02. Every system now has GMIN rows from the corrected file. The e3ed7dad GMIN rows are kept in `seeds.csv` and shown below next to their reruns. Each rerun starts from the anneal row's start files: the e3ed7dad and a50f4b02 builds of `h2h_sc` write the same bytes for every system and seed.

The anneal rows are rerun on all nine systems under harness c89653df, one seed at a time on core 0, with `scripts/metal_h2h/enter_sc_anneal.sh`. Its anneal code is that of develop. The e3ed7dad anneal rows are kept in `seeds.csv` and shown below. The c89653df rows call rgpot's Sutton-Chen entry at the rgpot revision of `Cargo.lock`, built by `scripts/metal_h2h/build_rgpot_sc.sh`, from the same start files: the c89653df build of `h2h_sc` writes the same bytes as the e3ed7dad build.

| System | Arm | Harness | Hits | Calls median | Wall median (s) |
|---|---|---|---|---|---|
| sc126-38 | anneal | e3ed7dad | 12/12 | 10344.5 | 0.518 |
| sc126-38 | anneal | c89653df | 12/12 | 9861 | 0.414 |
| sc126-38 | gmin | e3ed7dad | 1/12 | 400000 | 4.662 |
| sc126-38 | gmin | f9d1ac05 | 12/12 | 48872.5 | 0.471 |
| sc108-38 | anneal | e3ed7dad | 12/12 | 11193 | 0.504 |
| sc108-38 | anneal | c89653df | 12/12 | 9762.5 | 0.427 |
| sc108-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.570 |
| sc108-38 | gmin | f9d1ac05 | 0/12 | 400000 | 4.305 |
| sc96-38 | anneal | e3ed7dad | 12/12 | 9177 | 0.399 |
| sc96-38 | anneal | c89653df | 12/12 | 8185 | 0.374 |
| sc96-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.291 |
| sc96-38 | gmin | 9149d1b0 | 12/12 | 44974 | 0.458 |
| sc126-55 | anneal | e3ed7dad | 12/12 | 65397.5 | 5.104 |
| sc126-55 | anneal | c89653df | 12/12 | 50803 | 3.216 |
| sc126-55 | gmin | e3ed7dad | 0/12 | 1000000 | 22.940 |
| sc126-55 | gmin | a50f4b02 | 12/12 | 30565.5 | 0.576 |
| sc108-55 | anneal | e3ed7dad | 5/12 | 1000000 | 81.861 |
| sc108-55 | anneal | c89653df | 6/12 | 511393.5 | 32.844 |
| sc108-55 | gmin | e3ed7dad | 0/12 | 1000000 | 24.381 |
| sc108-55 | gmin | a50f4b02 | 2/12 | 1000000 | 21.165 |
| sc96-55 | anneal | e3ed7dad | 12/12 | 17280 | 1.316 |
| sc96-55 | anneal | c89653df | 12/12 | 28767 | 1.867 |
| sc96-55 | gmin | e3ed7dad | 0/12 | 1000000 | 21.334 |
| sc96-55 | gmin | a50f4b02 | 12/12 | 32261 | 0.652 |
| sc126-80 | anneal | e3ed7dad | 5/12 | 1000000 | 147.453 |
| sc126-80 | anneal | c89653df | 6/12 | 828059.5 | 104.625 |
| sc126-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.538 |
| sc126-80 | gmin | a50f4b02 | 0/12 | 1000000 | 38.015 |
| sc108-80 | anneal | e3ed7dad | 4/12 | 1000000 | 155.295 |
| sc108-80 | anneal | c89653df | 0/12 | 1000000 | 128.014 |
| sc108-80 | gmin | e3ed7dad | 0/12 | 1000000 | 51.184 |
| sc108-80 | gmin | a50f4b02 | 0/12 | 1000000 | 42.888 |
| sc96-80 | anneal | e3ed7dad | 7/12 | 773712.5 | 117.332 |
| sc96-80 | anneal | c89653df | 7/12 | 981181 | 126.882 |
| sc96-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.931 |
| sc96-80 | gmin | a50f4b02 | 0/12 | 1000000 | 40.355 |

The c89653df anneal rows against the e3ed7dad anneal rows of the same seeds: Fisher on hits, Mann-Whitney on calls to the hit.

| System | Old harness | New harness | Old hits | New hits | Fisher p | Old calls median | New calls median | Mann-Whitney p calls |
|---|---|---|---|---|---|---|---|---|
| sc126-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 10344.5 | 9861 | 0.507 |
| sc108-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 11193 | 9762.5 | 0.194 |
| sc96-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 9177 | 8185 | 0.403 |
| sc126-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 65397.5 | 50803 | 0.403 |
| sc108-55 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 511393.5 | 0.329 |
| sc96-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 17280 | 28767 | 0.237 |
| sc126-80 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 828059.5 | 0.432 |
| sc108-80 | e3ed7dad | c89653df | 4/12 | 0/12 | 0.0932 | 1000000 | 1000000 | 0.0367 |
| sc96-80 | e3ed7dad | c89653df | 7/12 | 7/12 | 1 | 773712.5 | 981181 | 0.59 |

The c89653df rows need more calls on sc108-80 than the e3ed7dad rows, at Mann-Whitney p < 0.05. On no system do their hits differ at Fisher p < 0.05.

On sc96-55, seeds 2000-2011, e3ed7dad hits 12/12 and c89653df 12/12, Fisher p 1, and the median calls move from 17280 to 28767, Mann-Whitney p 0.237. `sc96-55-seeds.csv` has seeds 2012-2059 under both builds from the same starts. There e3ed7dad hits 48 of 48 at a median of 24011.5 calls and c89653df hits 48 of 48 at a median of 23506.5, Fisher p 1 and Mann-Whitney p 0.58; c89653df charges more calls on 19 of 48 seeds and fewer on 29. Over all 60 seeds e3ed7dad hits 60 and c89653df 60, Fisher p 1, at median calls charged of 23441 and 23506.5, Mann-Whitney p 0.923. The move on seeds 2000-2011 does not hold over the 60 seeds, so it is seed noise and no develop commit was bisected. Their calls are the calls charged; both builds stop at the hit, and the e3ed7dad build prints the budget for the calls to some hits.

On sc108-80, seeds 2000-2011, e3ed7dad hits 4/12 and c89653df 0/12, Fisher p 0.0932, and the median calls stay at 1000000, Mann-Whitney p 0.0367. `sc108-80-seeds.csv` has seeds 2012-2059 under both builds from the same starts. Its 5c9dd1b3 build has the search code of c89653df. There e3ed7dad hits 15 of 48 at a median of 1000000 calls and 5c9dd1b3 hits 21 of 48 at a median of 1000000, Fisher p 0.292 and Mann-Whitney p 0.143; 5c9dd1b3 charges more calls on 14 of 48 seeds and fewer on 16. Over all 60 seeds e3ed7dad hits 19 and c89653df and 5c9dd1b3 21, Fisher p 0.847, at median calls charged of 1000000 and 1000000, Mann-Whitney p 0.582. The move on seeds 2000-2011 does not hold over the 60 seeds, so it is seed noise and no develop commit was bisected. Their calls are the calls charged; both builds stop at the hit, and the e3ed7dad build prints the budget for the calls to some hits.

Each rerun is tested against the c89653df anneal rows of its system. The wall ratio is the GMIN median wall over the anneal median wall. On calls and wall the code ahead is the one lower in more of the 144 seed pairs, the direction of the Mann-Whitney test; on hits it is the one with more hits. n.s. marks a lead at p >= 0.05.

| System | Anneal harness | GMIN harness | Fisher p | Mann-Whitney p calls | Mann-Whitney p wall | Wall ratio | Hits ahead | Calls ahead | Wall ahead |
|---|---|---|---|---|---|---|---|---|---|
| sc126-38 | c89653df | f9d1ac05 | 1 | 0.00111 | 0.285 | 1.139 | tie | anneal | anneal, n.s. |
| sc108-38 | c89653df | f9d1ac05 | 7.40e-07 | 1.03e-05 | 3.66e-05 | 10.087 | anneal | anneal | anneal |
| sc96-38 | c89653df | 9149d1b0 | 1 | 0.000592 | 0.341 | 1.223 | tie | anneal | anneal, n.s. |
| sc126-55 | c89653df | a50f4b02 | 1 | 0.285 | 0.000592 | 0.179 | tie | GMIN, n.s. | GMIN |
| sc108-55 | c89653df | a50f4b02 | 0.193 | 0.0424 | 1 | 0.644 | anneal, n.s. | anneal | tie |
| sc96-55 | c89653df | a50f4b02 | 1 | 0.507 | 0.0141 | 0.349 | tie | anneal, n.s. | GMIN |
| sc126-80 | c89653df | a50f4b02 | 0.0137 | 0.00705 | 0.0404 | 0.363 | anneal | anneal | GMIN |
| sc108-80 | c89653df | a50f4b02 | 1 | 1 | 3.66e-05 | 0.335 | tie | tie | GMIN |
| sc96-80 | c89653df | a50f4b02 | 0.00458 | 0.00286 | 0.000592 | 0.318 | anneal | anneal | GMIN |

GMIN ties anneal on hits on sc126-38, sc96-38, sc126-55, sc96-55 and sc108-80. Anneal hits more seeds than GMIN on sc108-38, sc126-80 and sc96-80 at Fisher p < 0.05, and on sc108-55 not significantly.

GMIN beats anneal on wall time on sc126-55, sc96-55 and sc108-80, at Mann-Whitney p < 0.05. Anneal's lead on wall time on sc126-38 and sc96-38 is not significant. GMIN's lead on calls on sc126-55 is not significant. Anneal beats GMIN on calls on sc126-38 and sc96-38, at Mann-Whitney p < 0.05. Anneal's lead on calls on sc96-55 is not significant. On sc108-55, sc126-80 and sc96-80 GMIN's median wall is lower while it hits fewer seeds, and its median calls are the full budget.

On sc108-38 every f9d1ac05 GMIN seed ends at -10108.5598. On sc126-80 the a50f4b02 GMIN seeds end between -78402.0224 and -78238.3887, against -78414.6271. On sc108-80 the a50f4b02 GMIN seeds end between -21811.8772 and -21799.3447, against -21819.9952. On sc96-80 the a50f4b02 GMIN seeds end between -19560.0322 and -19531.9554, against -19560.8966.

Inside is the time in the potential and outside the rest of the wall. The medians are over the twelve seeds; per call is the sum over the twelve seeds over the calls charged.

| System | Code | Harness | Wall median (s) | Inside median (s) | Outside median (s) | Inside (µs/call) | Outside (µs/call) |
|---|---|---|---|---|---|---|---|
| sc126-38 | anneal | c89653df | 0.414 | 0.385 | 0.030 | 33.61 | 2.86 |
| sc126-38 | gmin | f9d1ac05 | 0.471 | 0.396 | 0.076 | 8.11 | 1.56 |
| sc108-38 | anneal | c89653df | 0.427 | 0.390 | 0.034 | 34.03 | 2.88 |
| sc108-38 | gmin | f9d1ac05 | 4.305 | 3.692 | 0.611 | 9.27 | 1.52 |
| sc96-38 | anneal | c89653df | 0.374 | 0.342 | 0.031 | 34.21 | 2.94 |
| sc96-38 | gmin | 9149d1b0 | 0.458 | 0.387 | 0.071 | 8.63 | 1.56 |
| sc126-55 | anneal | c89653df | 3.216 | 2.952 | 0.255 | 58.42 | 4.76 |
| sc126-55 | gmin | a50f4b02 | 0.576 | 0.511 | 0.065 | 16.72 | 2.12 |
| sc108-55 | anneal | c89653df | 32.844 | 29.721 | 2.650 | 59.15 | 6.08 |
| sc108-55 | gmin | a50f4b02 | 21.165 | 19.077 | 2.072 | 19.17 | 2.08 |
| sc96-55 | anneal | c89653df | 1.867 | 1.724 | 0.143 | 60.64 | 5.01 |
| sc96-55 | gmin | a50f4b02 | 0.652 | 0.582 | 0.070 | 17.99 | 2.15 |
| sc126-80 | anneal | c89653df | 104.625 | 94.912 | 9.674 | 115.00 | 11.56 |
| sc126-80 | gmin | a50f4b02 | 38.015 | 35.324 | 2.695 | 35.31 | 2.69 |
| sc108-80 | anneal | c89653df | 128.014 | 116.275 | 11.414 | 116.18 | 11.44 |
| sc108-80 | gmin | a50f4b02 | 42.888 | 40.225 | 2.654 | 40.21 | 2.66 |
| sc96-80 | anneal | c89653df | 126.882 | 114.997 | 10.898 | 117.04 | 11.12 |
| sc96-80 | gmin | a50f4b02 | 40.355 | 37.650 | 2.696 | 37.81 | 2.70 |

Per call, anneal spends 33.6-117.0 µs inside the potential and GMIN 8.1-40.2 µs; outside it, anneal spends 2.9-11.6 µs and GMIN 1.5-2.7 µs.

GMIN still quenches to an RMS force of 1e-4 (`SLOPPYCONV`) and 1e-5 (`TIGHTCONV`). Anneal stops its quench at its own default.
