`rgpot_sutton_chen_energy_forces` and `rgpot_lj_eval` keep the potential of their last call on each thread instead of building one per call, so the next call with the same parameters and atom count, and for `rgpot_lj_eval` the same atomic numbers, reuses its neighbour update and scratch.
Any other call, and one that throws, drops it and builds a new one.
Energies and forces keep the bits of a potential built for the call.
`rgpot::Kept` in `rgpot/KeptPotential.hpp` holds one potential per thread and potential type, keyed by the bits of its parameters.
