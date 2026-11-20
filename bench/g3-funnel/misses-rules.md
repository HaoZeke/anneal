# G3 miss and first-best rules

Declared and committed before any miss structure was written or read.
`scripts/g3_funnel/replay.py` and `scripts/g3_funnel/misses.py` implement
them. Energy is never used as structural identity here.

## Final structure

Every row is replayed with the same binary or harness, seed and calls:
anneal by rerunning the seed, minima hopping with `run_mh.py --calls
<charged>`, which stops where the wall-capped seed stopped. A replay counts
as matching when its accepted-minimum trace has the sha256 stored on the row
and it prints the same `best`. The final structure is the last frame of the
replay's incumbent trace, which is the reported best. For every miss it is
written to `misses/<system>-<method>-<seed>.con` by `xyz_to_con`
(readcon-core 0.14.8, 17 decimals, reduced units, every atom `Ar`).

## Point group

The `.con` file is read back with ASE's `eon` reader and passed to pymatgen
2026.10.24 `PointGroupAnalyzer(tolerance=0.05, eigen_tolerance=0.01,
matrix_tolerance=0.1)`, lengths in sigma (Lennard-Jones) or r0 (Morse). On the
Cambridge global minima it returns Oh (LJ38), D5h (LJ75), Td (LJ98), C3v
(LJ150) and Cs (M80 rho 6) at every tolerance from 0.01 to 0.3.

## CNA class

The funnel label of `classify-rules.md` (icosahedral, decahedral, fcc,
other) computed on the `.con` structure.

## Miss class and structure identity

The miss class is the pair (CNA class, point group). Two final structures of
one system are the same structure when their sorted interatomic distance
lists differ by at most 0.01 (sigma or r0) in every entry; this is invariant
to rotation, translation, permutation and reflection. Structures are numbered
S1, S2, ... per system in order of increasing energy, for naming only.

The dominant trap of a system is its most frequent miss class, pooled over
methods; a tie goes to the class holding the lowest-energy miss.

## hop_first_best

The first frame of a replay's incumbent trace whose sorted distance list is
within 0.01 of the final frame's, by the same test. Its hop is
`hop_first_best`: completed hops for anneal, the minima hopping step (0 for
the opening quench) for minima hopping.
