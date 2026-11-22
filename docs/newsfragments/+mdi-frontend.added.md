<!-- Copyright 2023--present rgpot developers -->

An opt-in MDI compatibility frontend now serves every `rgpot_potential_t` to drivers such as LAMMPS `fix mdi/qm`, with atomic-unit wire conversion, shared energy/force evaluations, and TCP low-latency mode. Direct in-process calls and Cap'n Proto RPC remain the default paths.
