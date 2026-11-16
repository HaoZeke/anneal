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
harness f9d1ac05, and sc96-38 under harness 9149d1b0. The GMIN rows of
sc126-80, sc96-80, sc108-80, sc126-55, sc108-55 and sc96-55 are those of
e3ed7dad, from the old file, and are not rerun. The anneal rows are not
rerun.

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

Each rerun is tested against the anneal rows of its system. The wall ratio
is the GMIN median wall over the anneal median wall.

| System | GMIN harness | Fisher p | Mann-Whitney p calls | Mann-Whitney p wall | Wall ratio |
|---|---|---|---|---|---|
| sc126-38 | f9d1ac05 | 1 | 0.00426 | 0.885 | 0.911 |
| sc108-38 | f9d1ac05 | 7.40e-07 | 1.03e-05 | 3.66e-05 | 8.535 |
| sc96-38 | 9149d1b0 | 1 | 0.000592 | 0.624 | 1.148 |

On sc108-38 every f9d1ac05 GMIN seed ends at -10108.5598.

GMIN still quenches to an RMS force of 1e-4 (`SLOPPYCONV`) and 1e-5
(`TIGHTCONV`). Anneal stops its quench at its own default.
