<!-- Copyright 2023--present rgpot developers -->

Constructing and freeing a LAMMPS, GROMACS or OpenMM potential no longer takes another dynamic-linker reference for an engine file that is already open. An open that does not become the resident engine is closed, and the resident engine stays mapped.
