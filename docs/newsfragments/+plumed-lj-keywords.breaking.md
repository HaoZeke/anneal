`RGPOT` no longer takes `U0`, `CUTOFF` and `PSI` in PLUMED units.
Write `RGPOT ATOMS=... POTENTIAL=LJ OPTIONS={u0 1.0 psi 1.0 cutoff 8.0} TYPES=Ar` instead, with the options in eV and Angstrom: `U0=96.48533212331002 CUTOFF=0.8 PSI=0.1` under GROMACS units is `u0 1.0 cutoff 8.0 psi 1.0`.
`TYPES` is required.
