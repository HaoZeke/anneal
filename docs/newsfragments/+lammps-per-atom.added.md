`compute pe/atom`, `compute stress/atom` and `compute centroid/stress/atom`
work with `pair_style rgpot` for LJ, Morse, SuttonChen and EAMTable, on any
number of ranks; they match `lj/cut`, `morse` and `eam/alloy` atom by atom.
`LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot` and
`EAMTablePot` gain `forcePerAtom`, which also writes per-atom energies (pair
terms split half to each atom, EAM and Sutton-Chen embedding to its own atom)
and per-atom virials; `rgpot::PerAtomPotential` in `rgpot/PerAtom.hpp`
detects it. No virtual or data member changes, so the C++ layout is
unchanged. Potentials without the split stop the run with an error naming
the potential.
