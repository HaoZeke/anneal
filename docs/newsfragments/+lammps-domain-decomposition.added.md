`pair_style rgpot` evaluates finite-range potentials by domain on more than one rank, the way `pair_style eam` does.
For LJ, the `u0 psi` form, Morse, SuttonChen with a positive cutoff and EAMTable, each rank runs the density and force stages of the potential on the half neighbour list of LAMMPS, whose ghosts reach one cutoff plus the skin, and the densities of a many-body potential pass between the ranks between the two stages, so the force call no longer runs on rank 0 alone.
On one rank with `mode domain`, the 16384-atom Lennard-Jones box of `bench_pair_rgpot.py` takes 8.27 ms per step (`lj/cut` 8.38, the gather path 15.10) and the 32000-atom CuNi box 46.84 ms (`eam/alloy` 44.47, the gather path 70.10).
Energies, forces, the global virial, `pe/atom` and `stress/atom` match the gather path and `lj/cut`, `eam/alloy` and `morse` on 1, 2 and 4 ranks, under `newton on` and `off`, including ranks that own no atoms.
Engines, potentials without a finite range or the two stages, `virial fd` and one-rank runs keep the gather on rank 0.
The new `pair_style rgpot cutoff mode auto|gather|domain` keyword forces either path, and `init` logs which one runs and why.
In C++, `rgpot/Locality.hpp` adds `rgpot::Locality` and the concept `rgpot::LocalPotential`, a non-virtual `locality()` on `LJPot`, `MorsePot`, `SuttonChenPot` and `EAMTablePot`; `PotCaps` and the C++ layout are unchanged.
