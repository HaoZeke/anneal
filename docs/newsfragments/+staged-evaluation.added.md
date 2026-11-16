The header `rgpot/Staged.hpp` lets a host that builds its own neighbour list, such as LAMMPS, evaluate a finite-range potential on that list in two stages, the scheme of `pair_style eam`: `densityStage` sums each atom's densities, the host completes them across its copies of neighbouring atoms (`Locality::densities` values per atom), and `forceStage` adds the forces.
`HalfList` takes a half list in the layout of a LAMMPS `NeighList`, with newton on or off, and `StageTally` takes the host's share of the energy and virial and the per-atom split.
`LJPot`, `MorsePot`, `SuttonChenPot` and `EAMTablePot` have the stages, the concept `rgpot::StagedPotential` detects them, and `rgpot::spec::Built` carries them for a potential built by name.
`PotCaps` and the C++ layout are unchanged.
