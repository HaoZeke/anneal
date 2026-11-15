`rgpot_external_potential_energy_forces` returns 1 with its usual message for a
null `box9` when `n` is 0, instead of reading through the null pointer. The
LAMMPS, GROMACS and OpenMM constructors also reject an infinite `u0`, `cutoff`
or `psi`, which they used to accept.
