# Stock preset on the paper Lennard-Jones panel

The paper rows were produced with `Config::recommended` while the
structure priors were still on. This directory reruns that panel on
the stock preset: `Config::recommended` with no prior flag set.

Seeds are 2000 through 2011, twelve per cell, the seeds in the
committed rows. A hit is a finite best energy at or below the
reference plus 0.001. Budgets are 400000 calls at 38 atoms, 1000000
at 55, and 4000000 at 75 and above. The walk is one process on one
core, release build, locked dependencies. No prior environment
variable is set.

## Cells

The GMIN panel is the thirteen clusters in
`scripts/gmin_panel/run_panel.py`: LJ38, LJ55, LJ75, LJ98, LJ102,
LJ103, LJ104, LJ150, Au55, and Morse M55 and M80 at rho 6 and 14.
The ASE panel adds the five Lennard-Jones sizes LJ38, LJ55, LJ98,
LJ104 and LJ150. Those five use the same stock preset, the same
seeds and the same starts as the matching GMIN Lennard-Jones cells.
They are separate cells because the paper reports them separately.

Gold uses `Config::recommended` with the gold species list and a
basin stride of 8, which is what the metal driver does when no
stride variable is set. Morse uses `Config::recommended` and does
not set `long_ranged`. At the paper revision the Morse driver did
set that flag when the second-neighbour energy was below -0.05.

Starts are the sphere draw `random_cluster_in_radius` from the stock
container and minimum separation, seeded by the cell seed. Each
start is stored under `starts/` as little-endian hex bits of the
coordinates. The search reads that file.

`manifest.csv` freezes the cells, seeds, start digests, references,
and the paper hit and call counts. A paper miss leaves the call
count empty. Au55 has two stored sessions at the paper revision;
the later one is kept, and its hits and call counts match the
earlier one. The ASE recommended rows are revision
b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac, not that paper revision.
`settings-diff.txt` is the constructor comparison.

## Miss classification

These rules are fixed before any miss is read. They are not adjusted
to the structures this run finds.

The point group is read with `detect_all` on the whole cluster.
The orders are 2, 3, 4, 5 and 6, the orders the shell-core reader
tries, plus mirror planes. The tolerance is
`ShellCore::scaled(length_scale).tolerance`, which is 0.1 times the
preset length. Two axes of the same order count as one when the
absolute cosine of their directions exceeds 0.995. Inversion is
present when `inversion_rms` is below that same tolerance. The
label is the first match:

- six or more five-fold axes: Ih if a mirror or inversion is
  present, otherwise I
- one five-fold axis: D5h if a mirror is present, otherwise D5
- three or more four-fold axes: Oh if a mirror or inversion is
  present, otherwise O
- four or more three-fold axes: Td if a mirror is present,
  otherwise T
- one three-fold axis and three or more two-fold axes: D3h if a
  mirror is present, otherwise D3
- one three-fold axis and a mirror: C3v
- one three-fold axis: C3
- a two-fold axis and a mirror: C2v
- a two-fold axis: C2
- a mirror: Cs
- inversion only: Ci
- otherwise: C1

Common-neighbour analysis uses `cna` at `contact_cutoff`, which is
`RING_CUTOFF_SCALE` times the median nearest-neighbour distance.
The signature lists each triplet and its count, sorted, separated
by semicolons. The structure type is the first match, with f555,
f421 and f422 the shares of bonds labelled 555, 421 and 422:

- icosahedral when f555 is at least 0.40 and f421 is below 0.10
- decahedral when f555 is at least 0.05 and f421 is at least 0.10
- fcc/twin when f555 is below 0.05 and f421 + f422 is at least 0.25
- disordered otherwise

A miss writes `misses/<cell>-s<seed>.con` through readcon-core, and
one row of `misses.csv`.
