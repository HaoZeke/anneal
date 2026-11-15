`CP2KPot`, `CPMDPot`, `MOPACPot` and `NWChemPot` skip an engine variable that
is set but empty, as the other engine-backed potentials do. Before, the empty
path opened the running program and stopped the search with a missing-symbol
error.
