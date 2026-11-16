`EAMTablePot` keeps its pair list between force calls, with a Verlet skin of `EAMTableConfig::skin` (1 Angstrom by default, `skin` in `pair_coeff * * EAMTable ...` and the other hosts of `rgpot::spec::build`), instead of building a list with no skin on every call.
Energies, forces, stress and per-atom terms keep their bits for any skin.
The spline tables were and are made once, in the constructor.
Because the list is per-instance scratch, `caps().reentrancy` is now `Reentrancy::PerInstance`: a caller that evaluates one `EAMTablePot` from several threads at once needs one instance per thread, as for `LJPot`.
On the gather path of `pair_style rgpot` a 32000-atom CuNi step went from 338.63 to 70.74 ms on one rank, with one list build in 62 calls (`docs/orgmode/howto/engine-lammps.org`).
