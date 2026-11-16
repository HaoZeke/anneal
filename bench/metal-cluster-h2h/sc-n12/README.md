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

The GMIN rows rerun with the corrected file are sc126-38 and sc108-38 under
harness f9d1ac05; sc96-38 under harness 9149d1b0; sc126-55, sc108-55,
sc96-55, sc126-80, sc108-80 and sc96-80 under harness a50f4b02. Every system
now has GMIN rows from the corrected file. The e3ed7dad GMIN rows are kept
in `seeds.csv` and shown below next to their reruns. The anneal rows are
not rerun. Each rerun starts from the anneal row's start files: the
e3ed7dad and a50f4b02 builds of `h2h_sc` write the same bytes for every
system and seed. Harnesses f9d1ac05, 9149d1b0 and a50f4b02 were local
commits, pushed as bc3e0cb1, 016b6911 and 79ec1b7f with the same patches;
`bench/revision-aliases.csv` maps each to its pushed commit.

| System | Arm | Harness | Hits | Calls median | Wall median (s) |
|---|---|---|---|---|---|
| sc126-38 | anneal | e3ed7dad | 12/12 | 10344.5 | 0.518 |
| sc126-38 | gmin | e3ed7dad | 1/12 | 400000 | 4.662 |
| sc126-38 | gmin | f9d1ac05 | 12/12 | 48872.5 | 0.471 |
| sc108-38 | anneal | e3ed7dad | 12/12 | 11193 | 0.504 |
| sc108-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.570 |
| sc108-38 | gmin | f9d1ac05 | 0/12 | 400000 | 4.305 |
| sc96-38 | anneal | e3ed7dad | 12/12 | 9177 | 0.399 |
| sc96-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.291 |
| sc96-38 | gmin | 9149d1b0 | 12/12 | 44974 | 0.458 |
| sc126-55 | anneal | e3ed7dad | 12/12 | 65397.5 | 5.104 |
| sc126-55 | gmin | e3ed7dad | 0/12 | 1000000 | 22.940 |
| sc126-55 | gmin | a50f4b02 | 12/12 | 30565.5 | 0.576 |
| sc108-55 | anneal | e3ed7dad | 5/12 | 1000000 | 81.861 |
| sc108-55 | gmin | e3ed7dad | 0/12 | 1000000 | 24.381 |
| sc108-55 | gmin | a50f4b02 | 2/12 | 1000000 | 21.165 |
| sc96-55 | anneal | e3ed7dad | 12/12 | 17280 | 1.316 |
| sc96-55 | gmin | e3ed7dad | 0/12 | 1000000 | 21.334 |
| sc96-55 | gmin | a50f4b02 | 12/12 | 32261 | 0.652 |
| sc126-80 | anneal | e3ed7dad | 5/12 | 1000000 | 147.453 |
| sc126-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.538 |
| sc126-80 | gmin | a50f4b02 | 0/12 | 1000000 | 38.015 |
| sc108-80 | anneal | e3ed7dad | 4/12 | 1000000 | 155.295 |
| sc108-80 | gmin | e3ed7dad | 0/12 | 1000000 | 51.184 |
| sc108-80 | gmin | a50f4b02 | 0/12 | 1000000 | 42.888 |
| sc96-80 | anneal | e3ed7dad | 7/12 | 773712.5 | 117.332 |
| sc96-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.931 |
| sc96-80 | gmin | a50f4b02 | 0/12 | 1000000 | 40.355 |

Each rerun is tested against the anneal rows of its system. The wall ratio
is the GMIN median wall over the anneal median wall. On calls and wall the
code ahead is the one lower in more of the 144 seed pairs, the direction of
the Mann-Whitney test; on hits it is the one with more hits. n.s. marks a
lead at p >= 0.05.

| System | GMIN harness | Fisher p | Mann-Whitney p calls | Mann-Whitney p wall | Wall ratio | Hits ahead | Calls ahead | Wall ahead |
|---|---|---|---|---|---|---|---|---|
| sc126-38 | f9d1ac05 | 1 | 0.00426 | 0.885 | 0.911 | tie | anneal | GMIN, n.s. |
| sc108-38 | f9d1ac05 | 7.40e-07 | 1.03e-05 | 3.66e-05 | 8.535 | anneal | anneal | anneal |
| sc96-38 | 9149d1b0 | 1 | 0.000592 | 0.624 | 1.148 | tie | anneal | anneal, n.s. |
| sc126-55 | a50f4b02 | 1 | 0.214 | 0.000246 | 0.113 | tie | GMIN, n.s. | GMIN |
| sc108-55 | a50f4b02 | 0.371 | 0.184 | 0.0061 | 0.259 | anneal, n.s. | anneal, n.s. | GMIN |
| sc96-55 | a50f4b02 | 1 | 0.0999 | 0.0404 | 0.496 | tie | anneal, n.s. | GMIN |
| sc126-80 | a50f4b02 | 0.0373 | 0.0165 | 0.000592 | 0.258 | anneal | anneal | GMIN |
| sc108-80 | a50f4b02 | 0.0932 | 0.0367 | 0.175 | 0.276 | anneal, n.s. | anneal | GMIN, n.s. |
| sc96-80 | a50f4b02 | 0.00458 | 0.00286 | 0.0061 | 0.344 | anneal | anneal | GMIN |

GMIN ties anneal on hits on sc126-38, sc96-38, sc126-55 and sc96-55. Anneal
hits more seeds than GMIN on sc108-38, sc126-80 and sc96-80 at Fisher p <
0.05, and on sc108-55 and sc108-80 not significantly.

GMIN beats anneal on wall time on sc126-55 and sc96-55, at Mann-Whitney p <
0.05. GMIN's lead on wall time on sc126-38 is not significant. Anneal's lead
on wall time on sc96-38 is not significant. GMIN's lead on calls on sc126-55
is not significant. Anneal beats GMIN on calls on sc126-38 and sc96-38, at
Mann-Whitney p < 0.05. Anneal's lead on calls on sc96-55 is not significant.
On sc108-55, sc126-80, sc108-80 and sc96-80 GMIN's median wall is lower
while it hits fewer seeds, and its median calls are the full budget.

On sc108-38 every f9d1ac05 GMIN seed ends at -10108.5598. On sc126-80 the
a50f4b02 GMIN seeds end between -78402.0224 and -78238.3887, against
-78414.6271. On sc108-80 the a50f4b02 GMIN seeds end between -21811.8772 and
-21799.3447, against -21819.9952. On sc96-80 the a50f4b02 GMIN seeds end
between -19560.0322 and -19531.9554, against -19560.8966.

GMIN still quenches to an RMS force of 1e-4 (`SLOPPYCONV`) and 1e-5
(`TIGHTCONV`). Anneal stops its quench at its own default.
