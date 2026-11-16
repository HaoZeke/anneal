A stored `nlist::Neighbours` list is kept when the caller wraps atoms back into the periodic cell.
The half-skin test measures each atom's move to the nearest image of its reference position, so an atom that crossed a face no longer counts as having moved a cell vector, and the bins are redone after such a wrap.
The potentials that keep such a list (`LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot`, `EAMPot`, `StillingerWeberSi`, `TersoffSiC` and `EAMTablePot`) rebuild it less often on wrapped input: on the gather path of `pair_style rgpot`, 3 instead of 40 builds in 112 calls for 16384 Lennard-Jones atoms.
One thread walks the pairs in index order, so its results keep their bits; with `OMP_NUM_THREADS` above 1 the order follows the bins and results agree to rounding.
`nlist::Builder`, whose table stores image shifts, is unchanged.
