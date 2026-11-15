Copyright (c) anneal developers

(H2O)16, seeds 15000-15005.
Both arms search GFN2 plus the one-sided O-H wall, onset 1.15 angstrom,
200 eV per square angstrom.
The anneal budget is 280305.
The shared rescore is
`xtb in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500`.
The reference is wdbase-1, -81.396292669509 Eh.
A hit is at or below that energy by 1e-6 Eh.

`seeds.csv` keeps every row.
The original seed 15000 row stays.
The rerun row is the one under hostname bench, from develop 8b4a2592.
The hygienic panel uses that rerun in place of the original row.
The other five seeds were already under hostname bench.

The hygienic gap, anneal minus CREST, has W+ 0, W- 21, exact two-sided p
0.03125, rank-biserial -1, median -0.003035313706 Eh.
The bootstrap of 10000 medians, seed 20261110, runs from -0.006444978869
to -0.002255136597 Eh.
Literature hits are 0/6 on both arms.
The original panel, which still contains the first seed 15000 row, has p
0.09375.

wdbase-1 is an open cage: oxygen radius of gyration 3.70 angstrom, eight
3-coordinate and eight 4-coordinate oxygens, furthest hydrogen 5.62
angstrom from the centre.
The six search minima are tighter, radius of gyration 3.14 to 3.44
angstrom, and none has that coordination.
The preset sphere is a covalent diameter, about 4.32 angstrom for 48
atoms, and the (H2O)16 starts sit outside it.

Widening that sphere by default does not land.
On develop, (H2O)8 seed 17000 at 4000 calls finishes at
-1106.9955472624 eV, and (NH3)8 seed 17000 at 4000 calls finishes at
-965.1559715057 eV.
Setting the sphere from the group contact, and widening it to a quench
that does not fit, both leave that opening quench and then finish
higher: (H2O)8 at -1106.9930881701 eV and (NH3)8 at -965.1325107976 eV.
C10 seed 2000, which does not use the molecular sphere, finishes at
-571.31695072 eV either way.
An LJ13 pin, seed 2000, budget 20000, has the same trace and the same
final geometry either way.
The search library is unchanged.

## Capped panel, seeds 2000-2011

`scripts/h2o16_crest/enter_capped.sh` runs both arms on one objective
with one call cap, serial on one core under hostname bench
(`snakemake -s workflow/Snakefile h2o16_crest_capped`).
The objective is GFN2 through rgpot's `libxtb_engine.so` plus the O-H wall
above.
Anneal is `examples/gfn2_water_h2h`.
CREST 3.0.2 is
`crest start.xyz --gfn2 -T 1 --legacy -notopo -nocross -shake 1 -tstep 2`,
with its xtb run through `xtb --driver` into
`examples/gfn2_water_crest_driver`, which evaluates the same objective in
one handle.
Both arms start from anneal's random start for the seed.
CREST's run also starts with one optimisation of it through the driver,
charged to CREST.
The cap is 280305 calls for both arms, the anneal (H2O)16 budget.
A hit is an objective call at or below wdbase-1 plus 0.001 eV,
-2214.904861 eV, and both arms stop at their first hit.
Calls to the hit count every objective call up to it; a miss counts the
cap.
Inside is the time in the handle and outside the rest of the wall.

Rows are appended to `capped-rows.csv` with the anneal revision, the host
and the CREST, xtb and engine sha256, and each run appends one line per
arm to `capped-summary.csv`, with Fisher's exact test on hits and
two-sided Mann-Whitney tests on calls to the hit and on the best energy
against anneal.
Neither file is rewritten; the Snakemake rule outputs a stamp under
`/work`.
