<!-- Copyright 2023--present rgpot developers -->

`rgpot.lammps.FixExternal` now runs any Python rgpot potential through LAMMPS `fix external`, including unit conversion, triclinic cells, image unwrapping, analytic virials, deadlock-safe MPI error propagation, and complete callback teardown, without importing LAMMPS or mpi4py until needed.
