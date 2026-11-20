An empty-memory L-BFGS step no longer opens at one coordinate unit when
that step would stretch a declared bond past the bond check. The trial is
the longest fraction of the steepest step that leaves the bond inside
``bond_tolerance`` times the covalent radii.
