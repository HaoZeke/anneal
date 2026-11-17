Copyright (c) anneal developers

LJ98 and LJ104 at 4,000,000 calls. A hit is a quenched energy within
0.001 of the reference, -543.665361 and -582.086642.

The GMIN panels and the ASE-panel anneal arm do not use the same preset.

`h2h_gmin_proof` is `Config::recommended` with no further change.
`bench/lj-n12-500aaf65/summary.csv` records that driver at
`500aaf651dcb613c644c202a5519417b872b0e6e`, seeds 2000-2011, host bench:
LJ98 10/12, LJ104 9/12.
`bench/gmin-wall-v8`, the record in
`27f6ee98827e09caed21bd100e019886e9585c00`, names anneal `ec64b77b`
and the same driver. Seeds 2000-2005: LJ98 5/6, LJ104 6/6.

The ASE-panel anneal arm is `examples/ase_panel`, `Config::for_cluster`,
library `c2de652dca0bc75e2b0cb4afffac91fd656404e4`. Seeds 2000-2011,
host bench: LJ98 4/12, LJ104 0/12. The harness commit that printed those
rows does not change `src/`.

`recommended` starts from the same screened scales as `for_cluster`,
then sets a lean-burst library, move allocation, a depth reward, tabu
and symmetrise on stall, flexible SOAP, orbit completion on the fraction
core, trail and escape on stall, lattice relocation, symmetry measured
every 10 quenches, and 1000 relax steps. `for_cluster` keeps 200 relax
steps and does not complete orbits. `recommended` also installs a
surface term when the point count is below 98. LJ98 and LJ104 are not
below that line, so that term is off for both sizes.

The budget, the hit tolerance, and the n=12 seed list match. The preset
does not. The hit gap is that protocol difference. No same-session
rerun, no bisect, and no search change.
