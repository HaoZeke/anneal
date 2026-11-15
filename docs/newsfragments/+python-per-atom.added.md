`LJPot`, `LJClusterPot` and `MorsePot` gain `evaluate_per_atom`, which
returns everything `evaluate` does plus per-atom `energies` (shape `(N,)`, eV,
summing to the energy) and `virials` (shape `(N, 6)`, eV, Voigt order, summing
to minus the volume times the stress). It exists only on potentials whose C++
kernel has `forcePerAtom`.
