# G3 funnel classification rules

Declared and committed before any benchmark trace was produced. The same
rules apply to every system and every method. `scripts/g3_funnel/classify.py`
implements them.

## Signatures

Bond-based common neighbour analysis from OVITO 3.17.2
(`CreateBondsModifier` with a fixed pair cutoff, then
`CommonNeighborAnalysisModifier` in `BondBased` mode). Every bond gets the
triplet (common neighbours, bonds among them, longest chain). Each minimum is
reduced to four shares of its bonds:

| share | signature |
|-------|-----------|
| `f421` | (4,2,1) |
| `f422` | (4,2,2) |
| `f555` | (5,5,5) |
| `fother` | every other triplet, so the four shares sum to 1 |

A minimum with no bonds has `f421 = f422 = f555 = 0`, `fother = 1`.

## Cutoffs

The cutoff sits between the first and second neighbour shells of the pair
minimum distance `r_e`:

| potential | `r_e` | cutoff |
|-----------|-------|--------|
| Lennard-Jones, sigma = 1 | 2^(1/6) | 1.39 |
| Morse, r0 = 1 (any rho) | 1 | 1.39 / 2^(1/6) = 1.238330 |

## Funnel labels

Applied in this order; the first match wins.

1. `other` if `f421 + f422 + f555 < 0.25` (no ordered packing).
2. `icosahedral` if `f555 >= 0.035`.
3. `decahedral` if `f555 > 0`.
4. `fcc` (fcc / close-packed, including twinned and hcp-stacked packings)
   otherwise, i.e. `f555 = 0`.

## Calibration

The thresholds were set on reference structures only: the Cambridge Cluster
Database global minima for LJ38, LJ75, LJ98, LJ150 and Morse M80 at rho 6, and
relaxed Mackay icosahedra, Ino/Marks decahedra and truncated octahedra built
with `ase.cluster`, with and without atoms stripped to 38, 75, 98 and 150.

| structure | f421 | f422 | f555 | label |
|-----------|------|------|------|-------|
| LJ38 GM (truncated octahedron) | 0.417 | 0 | 0 | fcc |
| LJ75 GM (Marks decahedron) | 0.282 | 0.204 | 0.013 | decahedral |
| LJ98 GM (Leary tetrahedron) | 0.153 | 0.361 | 0.042 | icosahedral |
| LJ150 GM | 0.186 | 0.381 | 0.051 | icosahedral |
| M80 rho 6 GM | 0 | 0.424 | 0.136 | icosahedral |
| stripped icosahedra, n = 38-98 | 0-0.16 | 0.23-0.34 | 0.053-0.116 | icosahedral |
| stripped decahedra, n = 38-150 | 0.14-0.42 | 0.12-0.23 | 0-0.021 | decahedral or fcc |
| stripped fcc, n = 38-98 | 0.42-0.57 | 0 | 0 | fcc |

## Known limitation

The Leary tetrahedron (the LJ98 global minimum) is built around fivefold
disclinations and its four shares fall inside the icosahedral range, so these
rules label it `icosahedral`. For LJ98 the icosahedral label therefore covers
both the Leary and the Mackay-type funnels; the energy column separates the
global minimum from the rest.

## Transitions

A transition is a pair of consecutive accepted minima in one run whose labels
differ. It is attributed to the move recorded on the second minimum.
