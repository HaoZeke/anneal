# GFN2 gradient-fix floor rows

`gfn2.csv` stores one run of each seed under the three labels
`floor-rerun-grad-fix`, `series-grad-fix` and `grad-grad-fix`, not
triplicates. Hit, calls to the hit, and best energy agree across the
three labels of a seed. Counts below are unique seeds.

The rows are scored against the stop target -599.34546694 eV. That is
a stop target, not a verified minimum, so hits may be inflated. The
electronic temperature is 3000 K. These rows are not comparable with
other GFN2 rows. They were produced with `764aa838` plus the force-sign
patch, which is not on develop. The `code_revision` field on the rows
is `a37896fdf9d098a879db18f6a5d5e30b95b65da5`.

## Five Si13 reruns

Seeds 2000, 2001 and 2008, and the controls 2002 and 2003, were rerun
at the stored budget, seed and settings. The final structures are
`si13-geometry-check/si13-<seed>.xyz`. None of the five reruns
reproduced both the stored best and the stored `evals_to_hit`. The
stored lines were kept as recorded.

| seed | stored best (eV) | stored evals | rerun best (eV) | rerun evals |
|---|---:|---:|---:|---:|
| 2000 | -641.40367700 | 1499 | -599.34546694 | 1271 |
| 2001 | -618.15035402 | 1777 | -617.45116244 | 1906 |
| 2008 | -618.20745111 | 8192 | -601.87623818 | 4443 |
| 2002 | -599.34546694 | 1898 | -599.34546694 | 2039 |
| 2003 | -599.34546694 | 1417 | -599.34546694 | 5143 |

Seeds 2002 and 2003 match the stored best and differ in `evals_to_hit`.

## Geometry check

A structure is broken if the single-point SCC fails to converge at
300 K or at 3000 K, it is not one fragment, any pair is under 1.0
angstrom, or its 300 K relaxed energy is below the Si13 ring reference
by more than 0.01 eV. The ring reference is -624.70446339 eV in
`bench/covalent-h2h/gfn2/minima.csv`, with the structure in
`bench/covalent-h2h/gfn2/minima/si13-ring.xyz`.

The covalent cutoff is 2.775 angstrom, 1.25 times the sum of the
Cordero silicon radii (1.11 angstrom). Neighbours are linkcell pairs
and pair distances are minimage distances. Single points are xtb 6.7.1
GFN2 at accuracy 0.01 with a ceiling of 500 iterations. Free energy and
the electronic entropy term are not exposed. The re-relaxation is 200
steps of the search LBFGS through the rgpot GFN2 engine at 300 K.
The per-seed table is `si13-geometry-check/geometry-check.csv`.

- Seed 2000 is one fragment. The minimum pair is 2.2371335482 angstrom
  and no pair is under 1.0 angstrom. The SCC converged in 30 iterations
  at 300 K and in 22 iterations at 3000 K. The 300 K relaxed energy is
  -597.1589389798 eV. The structure is not broken.
- Seed 2001 is three fragments. The minimum pair is 2.1819563547
  angstrom and no pair is under 1.0 angstrom. The SCC converged in 196
  iterations at 300 K and in 55 iterations at 3000 K. The 300 K relaxed
  energy is -823.4410036118 eV, below the ring reference. The structure
  is broken.
- Seed 2008 is one fragment. The minimum pair is 1.5583987414 angstrom
  and no pair is under 1.0 angstrom. The SCC did not converge within
  500 iterations at 300 K or at 3000 K. The 300 K relaxed energy is
  -2357.8823610488 eV, below the ring reference. The structure is broken.
- Seed 2002 is one fragment. The minimum pair is 2.2371336866 angstrom
  and no pair is under 1.0 angstrom. The SCC converged in 27 iterations
  at 300 K and in 21 iterations at 3000 K. The 300 K relaxed energy is
  -597.1589364660 eV. The structure is not broken.
- Seed 2003 is one fragment. The minimum pair is 2.2371337085 angstrom
  and no pair is under 1.0 angstrom. The SCC converged in 28 iterations
  at 300 K and in 22 iterations at 3000 K. The 300 K relaxed energy is
  -597.1589389942 eV. The structure is not broken.

## Hit counts

A script counted unique seeds from `gfn2.csv` and treated every broken
seed in the geometry check as a miss. C20 is 0/12, as stored. Si13 is
12/12 as stored. Seeds 2001 and 2008 are broken, so the Si13 count is
10/12.
