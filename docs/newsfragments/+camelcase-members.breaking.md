C++ member functions of the potentials now use camelCase, like the other members of those classes.
Rename the calls; no old name remains.

- `PotentialBase::get_type()` is `PotentialBase::getType()`.
- `PotentialBase::set_cache()` and `Potential::set_cache()` are `setCache()`.
  An override in a derived class must use the new name.
  The Python method stays `set_cache`.
- `CP2KPot::probe_available()`, `CPMDPot::probe_available()`, `MOPACPot::probe_available()` and `NWChemPot::probe_available()` are `probeAvailable()`.
- `CPMDPot::abi_available()`, `MOPACPot::abi_available()` and `NWChemPot::abi_available()` are `abiAvailable()`.
- `lr::CoulombPot::set_neighbours()`, `lr::DispersionPot::set_neighbours()` and `lr::NlhPot::set_neighbours()` are `setNeighbours()`.
- `lr::SumPot::share_neighbours()` is `shareNeighbours()`.
- The namespace `rgpot::details` in `include/rgpot/errors.hpp` is `rgpot::detail`, like every other internal namespace.
