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

Only 9-6 at N=38 has been rerun, under harness 9149d1b0. The anneal rows
are not rerun. The other GMIN rows are those of e3ed7dad.

| System | Arm | Harness | Hits | Calls median | Wall median (s) |
|---|---|---|---|---|---|
| sc96-38 | anneal | e3ed7dad | 12/12 | 9177 | 0.399 |
| sc96-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.291 |
| sc96-38 | gmin | 9149d1b0 | 12/12 | 44974 | 0.458 |

Against the anneal rows, the 9149d1b0 GMIN rows give a Fisher p of 1 on
hits, a Mann-Whitney p of 0.000592 on calls and 0.624 on wall time. The
wall ratio, GMIN over anneal, is 1.148.

GMIN still quenches to an RMS force of 1e-4 (`SLOPPYCONV`) and 1e-5
(`TIGHTCONV`). Anneal stops its quench at its own default.
