`pair_style rgpot` evaluates finite-range potentials by domain on more than one rank.
For LJ, the `u0 psi` form, Morse, SuttonChen with a positive cutoff and EAMTable, each rank evaluates its own atoms plus the LAMMPS ghost atoms within the force range (the cutoff for a pair potential, twice the cutoff for a many-body one) and keeps the forces, energies and virials of its own atoms, so the force call no longer runs on rank 0 alone.
Energies, forces, the global virial, `pe/atom` and `stress/atom` match the gather path and `lj/cut` and `eam/alloy` on 1, 2 and 4 ranks, including ranks that own no atoms.
Engines, potentials without a finite range or a per-atom split, `virial fd` and one-rank runs keep the gather on rank 0.
The new `pair_style rgpot cutoff mode auto|gather|domain` keyword forces either path, and `init` logs which one runs and why.
In C++, `rgpot/Locality.hpp` adds `rgpot::Locality` and the concept `rgpot::LocalPotential`, a non-virtual `locality()` on `LJPot`, `MorsePot`, `SuttonChenPot` and `EAMTablePot`; `PotCaps` and the C++ layout are unchanged.
