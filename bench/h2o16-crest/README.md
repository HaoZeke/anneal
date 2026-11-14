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
The original seed 15000 row stays, recorded on vm-a.
The rerun row is the one under hostname bench, from develop 8b4a2592.
Substituting that one row for the original moves the exact paired p from
0.09375 to 0.03125.
The one-host panel is the six rows named onehost.
All six seeds of both arms were rerun together under hostname bench, at
code revision ca6adf99d6d8c8e3d432c3ecc7e788242fc13f36.
The earlier rows are unchanged.

Neither arm hits wdbase-1: 0/6 and 0/6.
Fisher's exact two-sided test on those hit counts has p 1.
The two-sided Mann-Whitney test on the six rescored energies, the normal
approximation with the tie correction and the continuity correction, has
p 0.2297662705.
The paired gap, anneal minus CREST, has W+ 4, W- 17, exact two-sided p
0.21875, rank-biserial -0.619047619, median -0.001204749507 Eh.
The bootstrap of 10000 medians, seed 20261110, runs from -0.003286605418
to 0.000687121629 Eh.
Anneal is lower on seeds 15000, 15001, 15002 and 15003, and higher on
15004 and 15005.
Anneal's lowest energy is seed 15003, -81.388562844846 Eh.
CREST's lowest energy is seed 15005, -81.388474624241 Eh.
Anneal's lowest is below CREST's lowest.

Anneal's time inside the potential, the six `potential_s` values, has
median 7982.2247655 s.
Time outside the potential, the six `outside_s` values, has median
6.6909455 s.
CREST's own runtime is the process wall in `crest_wall_s`.
That log does not split the wall into time inside the potential and time
outside it, so those two cells are empty.

de Souza, Angewandte Chemie International Edition 64, e202500393 (2025),
doi:10.1002/anie.202500393, reports an ice-derived (H2O)16 minimum on the
GFN2-xTB surface that CREST misses by 1.45 kcal/mol over three runs.
The supplement page lists that geometry in
anie202500393-sup-0001-misc_information.pdf, 248.1 KB.
The file was not obtained, so the structure was not relaxed and was not
used as a start or a seed.
Unpaywall and OpenAlex name only
https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/anie.202500393
and no repository copy.
Crossref links the same publisher PDF.
Semantic Scholar marks the paper closed and gives no PDF.
Europe PMC has no PMCID.
Requests for the pdfdirect URL, the epdf URL, the chemistry-europe and
aces hosts, and
`downloadSupplement` for anie202500393-sup-0001-misc_information.pdf
returned Cloudflare HTTP 403.
The Wayback Machine has only 403 captures and no copy of that file.
A fetch of the supplement page listed the filename.
A fetch of the PDF itself timed out.
archive.ph timed out, and a reader proxy received the Cloudflare challenge.

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
`crest start.xyz --gfn2 -T 1 --legacy -notopo -shake 1`, with
`-xnam` naming a wrapper that runs its xtb through `xtb --driver` into
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
